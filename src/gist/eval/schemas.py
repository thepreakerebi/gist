from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field

from gist.core.modes import AudioScoringMode, VisualScoringMode
from gist.core.presets import CompressionPreset
from gist.core.schemas import Candidate, CompressionResponse, Modality, SelectedCandidate
from gist.core.token_estimation import TokenEstimatorProfile


class EvalExample(BaseModel):
    id: Annotated[str, Field(min_length=1)]
    video_id: Annotated[str, Field(min_length=1)]
    query: Annotated[str, Field(min_length=1)]
    duration_seconds: Annotated[float, Field(gt=0)]
    video_path: Path | None = None
    sample_count: Annotated[int, Field(gt=0, le=512)] = 128
    audio_window_seconds: Annotated[float, Field(gt=0, le=30)] = 1.0
    relevant_timestamps: list[float] = Field(default_factory=list)
    timestamp_tolerance_seconds: Annotated[float, Field(gt=0)] = 5.0
    expected_answer: str | None = None
    choices: list[str] = Field(default_factory=list)
    visual_candidates: list[Candidate] = Field(default_factory=list)
    audio_candidates: list[Candidate] = Field(default_factory=list)


class EvalSettings(BaseModel):
    preset: CompressionPreset = CompressionPreset.BALANCED
    visual_scorer: VisualScoringMode = VisualScoringMode.BASELINE
    audio_scorer: AudioScoringMode = AudioScoringMode.BASELINE
    decompose_query: bool = False
    adaptive_budget: bool = False
    token_estimator: TokenEstimatorProfile = TokenEstimatorProfile.GENERIC
    spatial_pruning: bool = False
    spatial_retention_ratio: Annotated[float, Field(gt=0, le=1)] = 0.35
    spatial_grid_size: Annotated[int, Field(gt=0)] = 14


class EvalVariant(BaseModel):
    name: Annotated[str, Field(min_length=1)]
    preset: CompressionPreset = CompressionPreset.BALANCED
    visual_scorer: VisualScoringMode = VisualScoringMode.BASELINE
    audio_scorer: AudioScoringMode = AudioScoringMode.BASELINE
    decompose_query: bool = False
    adaptive_budget: bool = False
    token_estimator: TokenEstimatorProfile = TokenEstimatorProfile.GENERIC
    spatial_pruning: bool = False
    spatial_retention_ratio: Annotated[float, Field(gt=0, le=1)] = 0.35
    spatial_grid_size: Annotated[int, Field(gt=0)] = 14


class BaselineResult(BaseModel):
    name: str
    selected: list[SelectedCandidate]
    selected_candidates: int
    reduction_percent: float
    timestamp_hit_rate: float
    modality_coverage: dict[Modality, int]


class GistVariantResult(BaseModel):
    name: str
    settings: EvalVariant
    response: CompressionResponse
    timestamp_hit_rate: float
    latency_ms: float
    predicted_answer: str | None = None
    answer_score: float | None = None
    answer_provider: str | None = None


class EvalExampleResult(BaseModel):
    id: str
    query: str
    variants: list[GistVariantResult]
    baselines: list[BaselineResult]


class EvalSummary(BaseModel):
    examples: int
    variants: dict[str, "EvalVariantSummary"]


class EvalVariantSummary(BaseModel):
    avg_reduction_percent: float
    avg_token_reduction_percent: float
    avg_timestamp_hit_rate: float
    avg_latency_ms: float
    avg_answer_score: float | None = None


class EvalReport(BaseModel):
    settings: EvalSettings | None = None
    variants: list[EvalVariant]
    summary: EvalSummary
    results: list[EvalExampleResult]

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))
