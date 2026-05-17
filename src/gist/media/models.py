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

