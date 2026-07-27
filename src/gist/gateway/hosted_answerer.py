"""Model-agnostic answerer over hosted multimodal LLMs.

Both OpenAI and Anthropic consume the identical Gist evidence payload built by
`build_gateway_payload`, so the answerer can be swapped by name. This is the
"plug any LLM behind Gist" path used by the live web demo. The paper's measured
efficiency/FLOP claims come from Qwen2.5-Omni-7B offline, not from these hosted
APIs.
"""

from typing import Any

from gist.gateway import anthropic_vision, openai_vision
from gist.gateway.schemas import GatewayRequest, GatewayResponse
from gist.gateway.subprocess import build_gateway_payload

SUPPORTED_ANSWERERS = ("openai", "claude")


class HostedAnswererError(RuntimeError):
    """Raised when the requested hosted answerer is unknown or misconfigured."""


def answer_with_hosted_llm(
    request: GatewayRequest,
    answerer: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    max_frames: int = 8,
    timeout_seconds: float = 120.0,
    ffmpeg_bin: str = "ffmpeg",
    frame_sampling: str = "start",
    prompt_strategy: str = "default",
) -> GatewayResponse:
    """Answer a Gist compression using the named hosted multimodal LLM."""
    normalized = answerer.strip().lower()
    context, payload = build_gateway_payload(request)

    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "max_frames": max_frames,
        "timeout_seconds": timeout_seconds,
        "ffmpeg_bin": ffmpeg_bin,
        "frame_sampling": frame_sampling,
        "prompt_strategy": prompt_strategy,
    }

    if normalized == "openai":
        result = openai_vision.answer_from_gateway_payload(
            payload,
            model=model or openai_vision.DEFAULT_MODEL,
            **kwargs,
        )
    elif normalized == "claude":
        result = anthropic_vision.answer_from_gateway_payload(
            payload,
            model=model or anthropic_vision.DEFAULT_MODEL,
            **kwargs,
        )
    else:
        raise HostedAnswererError(
            f"unknown answerer {answerer!r}; expected one of {SUPPORTED_ANSWERERS}"
        )

    return GatewayResponse(
        answer=result["answer"],
        context=context,
        provider=result.get("provider", normalized),
    )
