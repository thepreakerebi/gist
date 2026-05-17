from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from gist.core.decomposition import QueryAspect
from gist.core.presets import CompressionPreset


class Modality(StrEnum):
    VISUAL = "visual"
    AUDIO = "audio"


class Candidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: Annotated[str, Field(min_length=1)]
    timestamp_seconds: Annotated[float, Field(ge=0)]
    text: str = ""
    saliency_score: float | None = None


class SelectedCandidate(BaseModel):
    id: str
    modality: Modality
    timestamp_seconds: float
    text: str
    selection_rank: int
    relevance_score: float
    normalized_score: float
    mmr_score: float
    source_score_type: str
    reason: str


class CompressionRequest(BaseModel):
    video_id: Annotated[str, Field(min_length=1)]
    query: Annotated[str, Field(min_length=1)]
    duration_seconds: Annotated[float, Field(gt=0)]
    preset: CompressionPreset = CompressionPreset.BALANCED
    decompose_query: bool = False
    visual_candidates: list[Candidate] = Field(default_factory=list)
    audio_candidates: list[Candidate] = Field(default_factory=list)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("query must not be blank")
        return normalized


class CompressionMetrics(BaseModel):
    input_candidates: int
    selected_candidates: int
    visual_selected: int
    audio_selected: int
    estimated_candidate_reduction_ratio: float
    estimated_candidate_reduction_percent: float
    dropped_candidates: int


class CompressionResponse(BaseModel):
    video_id: str
    query: str
    preset: CompressionPreset
    query_aspects: list[QueryAspect] = Field(default_factory=list)
    selected: list[SelectedCandidate]
    metrics: CompressionMetrics
