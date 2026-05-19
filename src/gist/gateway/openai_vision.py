import base64
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any
from urllib import error, request

from gist.gateway.prompt import build_video_answer_prompt


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

    clips = _evidence_clips(evidence)
    if not clips:
        return []

    allocations = _frame_allocations(clips, max_frames)
    output_dir.mkdir(parents=True, exist_ok=True)
    sampled: list[Path] = []

    for clip_index, (clip, frame_count) in enumerate(allocations):
        offsets = _sample_offsets(
            duration_seconds=clip["duration_seconds"],
            anchor_offset_seconds=clip["anchor_offset_seconds"],
            frame_count=frame_count,
        )
        for frame_index, offset in enumerate(offsets):
            frame_path = output_dir / f"clip-{clip_index:03d}-{frame_index:03d}.jpg"
            _extract_frame(
                ffmpeg_bin=ffmpeg_bin,
                clip_path=clip["path"],
                output_path=frame_path,
                offset_seconds=offset,
            )
            sampled.append(frame_path)

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
            "text": build_video_answer_prompt(gateway_payload),
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


def _data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/jpeg;base64,{encoded}"


def _evidence_clips(evidence: list[Any]) -> list[dict[str, Any]]:
    clips: list[dict[str, Any]] = []
    for item in evidence:
        if not isinstance(item, dict) or not isinstance(item.get("clip_path"), str):
            continue
        clip_path = Path(item["clip_path"])
        if not clip_path.exists():
            continue

        clip_start = _number(item.get("clip_start_seconds"))
        clip_end = _number(item.get("clip_end_seconds"))
        timestamp = _number(item.get("timestamp_seconds"))
        duration = max(0.2, (clip_end - clip_start) if clip_start is not None and clip_end is not None else 1.0)
        anchor = (
            timestamp - clip_start
            if timestamp is not None and clip_start is not None
            else duration / 2
        )
        clips.append(
            {
                "path": clip_path,
                "duration_seconds": duration,
                "anchor_offset_seconds": _clamp(anchor, 0.0, duration),
            }
        )
    return clips


def _frame_allocations(
    clips: list[dict[str, Any]],
    max_frames: int,
) -> list[tuple[dict[str, Any], int]]:
    selected = clips[:max_frames] if len(clips) >= max_frames else clips
    if not selected:
        return []

    base = max(1, max_frames // len(selected))
    remainder = max(0, max_frames - base * len(selected))
    return [
        (clip, base + (1 if index < remainder else 0))
        for index, clip in enumerate(selected)
    ]


def _sample_offsets(
    duration_seconds: float,
    anchor_offset_seconds: float,
    frame_count: int,
) -> list[float]:
    if frame_count <= 0:
        return []
    if frame_count == 1:
        return [_safe_offset(anchor_offset_seconds, duration_seconds)]

    step = duration_seconds / (frame_count + 1)
    offsets = [step * (index + 1) for index in range(frame_count)]
    offsets[frame_count // 2] = anchor_offset_seconds
    return [_safe_offset(offset, duration_seconds) for offset in offsets]


def _extract_frame(
    ffmpeg_bin: str,
    clip_path: Path,
    output_path: Path,
    offset_seconds: float,
) -> None:
    subprocess.run(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{offset_seconds:.3f}",
            "-i",
            str(clip_path),
            "-frames:v",
            "1",
            "-vf",
            "scale='min(768,iw)':-2",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _safe_offset(offset_seconds: float, duration_seconds: float) -> float:
    upper = max(0.0, duration_seconds - 0.05)
    return _clamp(offset_seconds, 0.0, upper)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))
