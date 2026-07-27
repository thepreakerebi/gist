import base64
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from urllib import error, request

from gist.gateway.openai_vision import sample_evidence_frames
from gist.gateway.prompt import build_video_answer_prompt

ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-sonnet-5"


class AnthropicVisionGatewayError(RuntimeError):
    """Raised when the Anthropic vision gateway cannot produce an answer."""


def answer_from_gateway_payload(
    payload: dict[str, Any],
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    max_frames: int = 8,
    timeout_seconds: float = 120.0,
    ffmpeg_bin: str = "ffmpeg",
    frame_sampling: str = "start",
    prompt_strategy: str = "default",
) -> dict[str, str]:
    resolved_api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not resolved_api_key:
        raise AnthropicVisionGatewayError(
            "ANTHROPIC_API_KEY is required for Anthropic vision gateway"
        )

    with tempfile.TemporaryDirectory(prefix="gist-anthropic-frames-") as temp_dir:
        frame_paths = sample_evidence_frames(
            evidence=payload.get("evidence", []),
            output_dir=Path(temp_dir),
            max_frames=max_frames,
            ffmpeg_bin=ffmpeg_bin,
            strategy=frame_sampling,
        )
        message_payload = create_messages_payload(
            gateway_payload=payload,
            frame_paths=frame_paths,
            model=model,
            prompt_strategy=prompt_strategy,
        )
        response = post_anthropic_response(
            payload=message_payload,
            api_key=resolved_api_key,
            timeout_seconds=timeout_seconds,
        )

    return {
        "answer": extract_output_text(response),
        "provider": f"anthropic:{model}",
    }


def create_messages_payload(
    gateway_payload: dict[str, Any],
    frame_paths: list[Path],
    model: str,
    prompt_strategy: str = "default",
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": build_video_answer_prompt(gateway_payload, strategy=prompt_strategy),
        }
    ]
    for frame_path in frame_paths:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": _base64_jpeg(frame_path),
                },
            }
        )

    return {
        "model": model,
        "max_tokens": 256,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
    }


def post_anthropic_response(
    payload: dict[str, Any],
    api_key: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    body = json.dumps(payload).encode()
    http_request = request.Request(
        ANTHROPIC_MESSAGES_URL,
        data=body,
        headers={
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode())
    except error.HTTPError as exc:
        message = exc.read().decode(errors="replace")
        raise AnthropicVisionGatewayError(f"Anthropic API error {exc.code}: {message}") from exc
    except error.URLError as exc:
        raise AnthropicVisionGatewayError(f"Anthropic API request failed: {exc}") from exc


def extract_output_text(response: dict[str, Any]) -> str:
    text_parts: list[str] = []
    for content_item in response.get("content", []):
        if not isinstance(content_item, dict):
            continue
        if content_item.get("type") != "text":
            continue
        text = content_item.get("text")
        if isinstance(text, str) and text.strip():
            text_parts.append(text.strip())
    if text_parts:
        return "\n".join(text_parts)
    raise AnthropicVisionGatewayError("Anthropic response did not include output text")


def _base64_jpeg(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()
