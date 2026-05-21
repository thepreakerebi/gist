from pathlib import Path

from pydantic import BaseModel, Field


class VideoMetadata(BaseModel):
    duration_seconds: float = Field(gt=0)
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    frame_rate: float | None = Field(default=None, gt=0)
    has_audio: bool = False


class ExtractedFrame(BaseModel):
    index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)
    path: Path


class AudioWindow(BaseModel):
    index: int = Field(ge=0)
    start_seconds: float = Field(ge=0)
    duration_seconds: float = Field(gt=0)
    path: Path


class IngestionSettings(BaseModel):
    processing_mode: str = "manual"
    sample_count: int = Field(gt=0)
    audio_window_seconds: float = Field(gt=0)
    audio_context_window_count: int = Field(ge=0)
    max_audio_windows: int = Field(gt=0)
    reason: str = ""


class IngestedVideo(BaseModel):
    video_id: str = Field(min_length=1)
    source_path: Path
    metadata: VideoMetadata
    frames: list[ExtractedFrame]
    audio_windows: list[AudioWindow]
    settings: IngestionSettings | None = None
