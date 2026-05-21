from dataclasses import dataclass
from enum import StrEnum
import math


class ProcessingMode(StrEnum):
    AUTO = "auto"
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


@dataclass(frozen=True, slots=True)
class IngestionPlan:
    mode: ProcessingMode
    sample_count: int
    audio_window_seconds: float
    audio_context_window_count: int
    max_audio_windows: int
    reason: str


SHORT_VIDEO_SECONDS = 10 * 60
MEDIUM_VIDEO_SECONDS = 60 * 60

SHORT_DEFAULT_FRAMES = 128
MEDIUM_DEFAULT_FRAMES = 256
LONG_DEFAULT_FRAMES = 512

SHORT_AUDIO_WINDOW_SECONDS = 2.0
MEDIUM_AUDIO_WINDOW_SECONDS = 10.0
LONG_AUDIO_WINDOW_SECONDS = 30.0
MAX_LONG_AUDIO_WINDOWS = 240


def plan_ingestion(
    duration_seconds: float,
    mode: ProcessingMode = ProcessingMode.AUTO,
    sample_count: int | None = None,
    audio_window_seconds: float | None = None,
) -> IngestionPlan:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be greater than zero")

    resolved_mode = _resolve_mode(duration_seconds, mode)
    default_frames, default_audio_window, context_count = _defaults_for(resolved_mode)
    resolved_sample_count = sample_count or default_frames
    resolved_audio_window = audio_window_seconds or default_audio_window

    if resolved_mode == ProcessingMode.LONG:
        resolved_audio_window = max(
            resolved_audio_window,
            math.ceil(duration_seconds / MAX_LONG_AUDIO_WINDOWS),
        )

    return IngestionPlan(
        mode=resolved_mode,
        sample_count=resolved_sample_count,
        audio_window_seconds=float(resolved_audio_window),
        audio_context_window_count=context_count,
        max_audio_windows=math.ceil(duration_seconds / resolved_audio_window),
        reason=_plan_reason(
            duration_seconds=duration_seconds,
            mode=resolved_mode,
            sample_count=sample_count,
            audio_window_seconds=audio_window_seconds,
        ),
    )


def _resolve_mode(duration_seconds: float, mode: ProcessingMode) -> ProcessingMode:
    if mode != ProcessingMode.AUTO:
        return mode
    if duration_seconds <= SHORT_VIDEO_SECONDS:
        return ProcessingMode.SHORT
    if duration_seconds <= MEDIUM_VIDEO_SECONDS:
        return ProcessingMode.MEDIUM
    return ProcessingMode.LONG


def _defaults_for(mode: ProcessingMode) -> tuple[int, float, int]:
    if mode == ProcessingMode.SHORT:
        return SHORT_DEFAULT_FRAMES, SHORT_AUDIO_WINDOW_SECONDS, 1
    if mode == ProcessingMode.MEDIUM:
        return MEDIUM_DEFAULT_FRAMES, MEDIUM_AUDIO_WINDOW_SECONDS, 1
    if mode == ProcessingMode.LONG:
        return LONG_DEFAULT_FRAMES, LONG_AUDIO_WINDOW_SECONDS, 0
    raise ValueError(f"unsupported processing mode: {mode}")


def _plan_reason(
    duration_seconds: float,
    mode: ProcessingMode,
    sample_count: int | None,
    audio_window_seconds: float | None,
) -> str:
    override_parts: list[str] = []
    if sample_count is not None:
        override_parts.append("caller-provided frame count")
    if audio_window_seconds is not None:
        override_parts.append("caller-provided audio window")

    override_suffix = ""
    if override_parts:
        override_suffix = f" with {' and '.join(override_parts)}"

    return (
        f"{mode.value} processing selected for "
        f"{duration_seconds / 60:.1f} minute video{override_suffix}"
    )
