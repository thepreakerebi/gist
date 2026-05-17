from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field


class MediaIngestionRequest(BaseModel):
    video_path: Path
    output_root: Path = Path(".gist/ingestions")
    sample_count: Annotated[int, Field(gt=0, le=512)] = 128
    audio_window_seconds: Annotated[float, Field(gt=0, le=30)] = 1.0
