from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field

from gist.core.presets import CompressionPreset
from gist.core.schemas import Candidate, CompressionResponse, Modality, SelectedCandidate


class EvalExample(BaseModel):
    id: Annotated[str, Field(min_length=1)]
    video_id: Annotated[str, Field(min_length=1)]
    query: Annotated[str, Field(min_length=1)]
    duration_seconds: Annotated[float, Field(gt=0)]
    relevant_timestamps: list[float] = Field(default_factory=list)
    timestamp_tolerance_seconds: Annotated[float, Field(gt=0)] = 5.0
    visual_candidates: list[Candidate] = Field(default_factory=list)
    audio_candidates: list[Candidate] = Field(default_factory=list)


class EvalSettings(BaseModel):
    preset: CompressionPreset = CompressionPreset.BALANCED
    decompose_query: bool = False
    adaptive_budget: bool = False


class BaselineResult(BaseModel):
    name: str
    selected: list[SelectedCandidate]
    selected_candidates: int
    reduction_percent: float
    timestamp_hit_rate: float
    modality_coverage: dict[Modality, int]


class EvalExampleResult(BaseModel):
    id: str
    query: str
    gist: CompressionResponse
    gist_timestamp_hit_rate: float
    baselines: list[BaselineResult]
    latency_ms: float


class EvalSummary(BaseModel):
    examples: int
    avg_gist_reduction_percent: float
    avg_gist_timestamp_hit_rate: float
    avg_latency_ms: float


class EvalReport(BaseModel):
    settings: EvalSettings
    summary: EvalSummary
    results: list[EvalExampleResult]

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))
