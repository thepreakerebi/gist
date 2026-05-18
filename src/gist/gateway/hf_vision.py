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
        from transformers import pipeline
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
        vlm = pipeline(
            "image-text-to-text",
            model=model,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        output = vlm(text=messages, max_new_tokens=max_new_tokens)

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
