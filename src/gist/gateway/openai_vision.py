import base64
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any
from urllib import error, request


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_MODEL = "gpt-4.1-mini"


class OpenAIVisionGatewayError(RuntimeError):
    """Raised when the OpenAI vision gateway cannot produce an answer."""


def answer_from_gateway_payload(
    payload: dict[str, Any],
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    max_frames: int = 8,
    detail: str = "low",
    timeout_seconds: float = 120.0,
    ffmpeg_bin: str = "ffmpeg",
) -> dict[str, str]:
    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_api_key:
        raise OpenAIVisionGatewayError("OPENAI_API_KEY is required for OpenAI vision gateway")

    with tempfile.TemporaryDirectory(prefix="gist-openai-frames-") as temp_dir:
        frame_paths = sample_evidence_frames(
            evidence=payload.get("evidence", []),
            output_dir=Path(temp_dir),
            max_frames=max_frames,
            ffmpeg_bin=ffmpeg_bin,
        )
        response_payload = create_responses_payload(
            gateway_payload=payload,
            frame_paths=frame_paths,
            model=model,
            detail=detail,
        )
        response = post_openai_response(
            payload=response_payload,
            api_key=resolved_api_key,
            timeout_seconds=timeout_seconds,
        )

    return {
        "answer": extract_output_text(response),
        "provider": f"openai:{model}",
    }


def sample_evidence_frames(
    evidence: list[Any],
    output_dir: Path,
    max_frames: int,
    ffmpeg_bin: str = "ffmpeg",
) -> list[Path]:
    if max_frames <= 0:
        return []

    clip_paths = [
        Path(item["clip_path"])
        for item in evidence
        if isinstance(item, dict)
        and isinstance(item.get("clip_path"), str)
        and Path(item["clip_path"]).exists()
    ]
    if not clip_paths:
        return []

    per_clip = max(1, max_frames // len(clip_paths))
    output_dir.mkdir(parents=True, exist_ok=True)
    sampled: list[Path] = []
    for clip_index, clip_path in enumerate(clip_paths):
        pattern = output_dir / f"clip-{clip_index:03d}-%03d.jpg"
        subprocess.run(
            [
                ffmpeg_bin,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(clip_path),
                "-vf",
                "fps=1,scale='min(768,iw)':-2",
                "-frames:v",
                str(per_clip),
                str(pattern),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        sampled.extend(sorted(output_dir.glob(f"clip-{clip_index:03d}-*.jpg")))
        if len(sampled) >= max_frames:
            break
    return sampled[:max_frames]


def create_responses_payload(
    gateway_payload: dict[str, Any],
    frame_paths: list[Path],
    model: str,
    detail: str,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": _prompt(gateway_payload),
        }
    ]
    for frame_path in frame_paths:
        content.append(
            {
                "type": "input_image",
                "image_url": _data_url(frame_path),
                "detail": detail,
            }
        )

    return {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "max_output_tokens": 256,
    }


def post_openai_response(
    payload: dict[str, Any],
    api_key: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    body = json.dumps(payload).encode()
    http_request = request.Request(
        OPENAI_RESPONSES_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode())
    except error.HTTPError as exc:
        message = exc.read().decode(errors="replace")
        raise OpenAIVisionGatewayError(f"OpenAI API error {exc.code}: {message}") from exc
    except error.URLError as exc:
        raise OpenAIVisionGatewayError(f"OpenAI API request failed: {exc}") from exc


def extract_output_text(response: dict[str, Any]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    text_parts: list[str] = []
    for output_item in response.get("output", []):
        if not isinstance(output_item, dict):
            continue
        for content_item in output_item.get("content", []):
            if not isinstance(content_item, dict):
                continue
            text = content_item.get("text")
            if isinstance(text, str) and text.strip():
                text_parts.append(text.strip())
    if text_parts:
        return "\n".join(text_parts)
    raise OpenAIVisionGatewayError("OpenAI response did not include output text")


def _prompt(gateway_payload: dict[str, Any]) -> str:
    query = str(gateway_payload.get("query", "")).strip()
    context = str(gateway_payload.get("context", "")).strip()
    return (
        "Answer the video question using only the provided Gist evidence frames and "
        "evidence context. If choices are visible in the question, answer with the "
        "best choice text or letter. Keep the answer concise.\n\n"
        f"Question: {query}\n\n"
        f"Evidence context:\n{context}"
    )


def _data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/jpeg;base64,{encoded}"
