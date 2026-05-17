from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field

from gist.core.modes import AudioScoringMode, VisualScoringMode
from gist.core.presets import CompressionPreset
from gist.core.schemas import CompressionResponse
from gist.core.token_estimation import TokenEstimatorProfile
from gist.media.models import IngestedVideo


class MediaIngestionRequest(BaseModel):
    video_path: Path
    output_root: Path = Path(".gist/ingestions")
    sample_count: Annotated[int, Field(gt=0, le=512)] = 128
    audio_window_seconds: Annotated[float, Field(gt=0, le=30)] = 1.0


class LocalVideoCompressionRequest(MediaIngestionRequest):
    query: Annotated[str, Field(min_length=1)]
    preset: CompressionPreset = CompressionPreset.BALANCED
    visual_scorer: VisualScoringMode = VisualScoringMode.BASELINE
    audio_scorer: AudioScoringMode = AudioScoringMode.BASELINE
    adaptive_budget: bool = False
    decompose_query: bool = False
    token_estimator: TokenEstimatorProfile = TokenEstimatorProfile.GENERIC


class LocalVideoCompressionResponse(BaseModel):
    ingestion: IngestedVideo
    compression: CompressionResponse
