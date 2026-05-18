import json
from pathlib import Path
import tempfile
from typing import Any

from gist.gateway.openai_vision import sample_evidence_frames


DEFAULT_HF_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


class HuggingFaceVisionGatewayError(RuntimeError):
    """Raised when a local Hugging Face VLM cannot produce an answer."""


def answer_from_gateway_payload(
    payload: dict[str, Any],
    model: str = DEFAULT_HF_MODEL,
    max_frames: int = 8,
    max_new_tokens: int = 128,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    trust_remote_code: bool = False,
    ffmpeg_bin: str = "ffmpeg",
) -> dict[str, str]:
    try:
        from PIL import Image
        import torch
        from transformers import AutoProcessor
    except ImportError as exc:
        raise HuggingFaceVisionGatewayError(
            "Hugging Face VLM gateway requires optional vision dependencies. "
            "Install with: pip install -e '.[sota]'"
        ) from exc

    with tempfile.TemporaryDirectory(prefix="gist-hf-frames-") as temp_dir:
        frame_paths = sample_evidence_frames(
            evidence=payload.get("evidence", []),
            output_dir=Path(temp_dir),
            max_frames=max_frames,
            ffmpeg_bin=ffmpeg_bin,
        )
        images = [Image.open(path).convert("RGB") for path in frame_paths]
        messages = build_messages(payload, images)
        processor = AutoProcessor.from_pretrained(
            model,
            trust_remote_code=trust_remote_code,
        )
        vlm = _load_model(
            model=model,
            device_map=device_map,
            dtype=_resolve_dtype(torch, torch_dtype),
            trust_remote_code=trust_remote_code,
        )
        device = _model_device(vlm)
        prompt = processor.apply_chat_template(
            _messages_with_image_placeholders(payload, images),
            add_generation_prompt=True,
        )
        inputs = processor(
            text=prompt,
            images=images or None,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            generated_ids = vlm.generate(**inputs, max_new_tokens=max_new_tokens)
        input_length = inputs["input_ids"].shape[-1]
        output = processor.batch_decode(
            generated_ids[:, input_length:],
            skip_special_tokens=True,
        )[0]

    return {
        "answer": extract_pipeline_text(output),
        "provider": f"hf:{model}",
    }


def build_messages(payload: dict[str, Any], images: list[Any]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {"type": "image", "image": image}
        for image in images
    ]
    content.append(
        {
            "type": "text",
            "text": _prompt(payload),
        }
    )
    return [
        {
            "role": "user",
            "content": content,
        }
    ]


def _messages_with_image_placeholders(
    payload: dict[str, Any],
    images: list[Any],
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "image"} for _image in images]
    content.append({"type": "text", "text": _prompt(payload)})
    return [{"role": "user", "content": content}]


def _load_model(
    model: str,
    device_map: str,
    dtype: Any,
    trust_remote_code: bool,
) -> Any:
    try:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(
            model,
            device_map=device_map,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
    except (ImportError, ValueError, OSError):
        from transformers import AutoModelForVision2Seq

        return AutoModelForVision2Seq.from_pretrained(
            model,
            device_map=device_map,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )


def _resolve_dtype(torch: Any, torch_dtype: str) -> Any:
    if torch_dtype == "auto":
        return "auto"
    dtype = getattr(torch, torch_dtype, None)
    if dtype is None:
        raise HuggingFaceVisionGatewayError(f"unsupported torch dtype: {torch_dtype}")
    return dtype


def _model_device(model: Any) -> Any:
    if hasattr(model, "device"):
        return model.device
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise HuggingFaceVisionGatewayError("model has no parameters") from exc


def extract_pipeline_text(output: Any) -> str:
    if isinstance(output, str) and output.strip():
        return output.strip()
    if isinstance(output, list) and output:
        first = output[0]
        if isinstance(first, str) and first.strip():
            return first.strip()
        if isinstance(first, dict):
            generated = first.get("generated_text")
            if isinstance(generated, str) and generated.strip():
                return generated.strip()
            if isinstance(generated, list):
                text = _text_from_messages(generated)
                if text:
                    return text
    raise HuggingFaceVisionGatewayError(
        f"could not extract generated text from pipeline output: {json.dumps(_safe_json(output))}"
    )


def _text_from_messages(messages: list[Any]) -> str | None:
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            parts = [
                item.get("text", "").strip()
                for item in content
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            ]
            text = " ".join(part for part in parts if part)
            if text:
                return text
    return None


def _prompt(payload: dict[str, Any]) -> str:
    query = str(payload.get("query", "")).strip()
    context = str(payload.get("context", "")).strip()
    return (
        "Answer the video question using only the provided Gist evidence frames and "
        "evidence context. If the question is multiple choice, answer with the best "
        "choice letter or text. Keep the answer concise.\n\n"
        f"Question: {query}\n\n"
        f"Evidence context:\n{context}"
    )


def _safe_json(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)
