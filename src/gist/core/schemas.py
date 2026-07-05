from enum import StrEnum
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from gist.core.decomposition import QueryAspect
from gist.core.modes import AudioScoringMode
from gist.core.presets import CompressionPreset
from gist.core.query_intent import QueryIntent
from gist.core.token_estimation import TokenEstimatorProfile


class Modality(StrEnum):
    VISUAL = "visual"
    AUDIO = "audio"


class Candidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: Annotated[str, Field(min_length=1)]
    timestamp_seconds: Annotated[float, Field(ge=0)]
    text: str = ""
    saliency_score: float | None = None
    asset_path: Path | None = None
    segment_id: str | None = None
    scene_start_seconds: float | None = None
    scene_end_seconds: float | None = None
    spatial_mask_path: Path | None = None
    temporal_anchor_score: float | None = None
    temporal_target_score: float | None = None
    temporal_direction: str | None = None


class SelectedCandidate(BaseModel):
    id: str
    modality: Modality
    timestamp_seconds: float
    text: str
    asset_path: Path | None = None
    clip_path: Path | None = None
    clip_start_seconds: float | None = None
    clip_end_seconds: float | None = None
    segment_id: str | None = None
    scene_start_seconds: float | None = None
    scene_end_seconds: float | None = None
    spatial_mask_path: Path | None = None
    spatial_mask_preview_path: Path | None = None
    spatial_mask_overlay_path: Path | None = None
    audio_anchor_timestamp_seconds: float | None = None
    audio_anchor_score: float = 0.0
    temporal_anchor_score: float | None = None
    temporal_target_score: float | None = None
    temporal_direction: str | None = None
    selection_rank: int
    relevance_score: float
    normalized_score: float
    mmr_score: float
    answer_support_score: float | None = None
    query_support_score: float | None = None
    evidence_support_score: float | None = None
    audio_support_score: float | None = None
    ocr_support_score: float | None = None
    visual_support_score: float | None = None
    cross_modal_support_score: float | None = None
    support_label: str | None = None
    grounding_label: str | None = None
    grounding_reason: str | None = None
    source_score_type: str
    reason: str


class CompressionRequest(BaseModel):
    video_id: Annotated[str, Field(min_length=1)]
    query: Annotated[str, Field(min_length=1)]
    duration_seconds: Annotated[float, Field(gt=0)]
    preset: CompressionPreset = CompressionPreset.BALANCED
    adaptive_budget: bool = False
    decompose_query: bool = False
    token_estimator: TokenEstimatorProfile = TokenEstimatorProfile.GENERIC
    task_aware_selection: bool = False
    # Per-intent coverage post-processors (_ensure_* in the compressor) beyond the
    # structurally-justified opening + temporal-pair preservation. On by default to
    # preserve behaviour; set False to A/B whether they carry signal or just overfit.
    coverage_heuristics: bool = True
    query_intent: QueryIntent | None = None
    routing_reason: str | None = None
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
    raw_input_candidates: int | None = None
    fused_input_candidates: int | None = None
    estimated_candidate_reduction_ratio: float
    estimated_candidate_reduction_percent: float
    dropped_candidates: int
    budget_mode: str = "fixed"
    budget_preset_used: CompressionPreset
    budget_expanded: bool = False
    expansion_reason: str | None = None
    estimated_baseline_tokens: int = 0
    estimated_raw_baseline_tokens: int | None = None
    estimated_compressed_tokens: int = 0
    estimated_saved_tokens: int = 0
    estimated_token_reduction_ratio: float = 0.0
    estimated_token_reduction_percent: float = 0.0
    token_estimator: TokenEstimatorProfile = TokenEstimatorProfile.GENERIC
    estimated_spatial_visual_tokens: int = 0
    estimated_retained_spatial_visual_tokens: int = 0
    estimated_spatial_token_reduction_percent: float = 0.0


class QualityWarning(BaseModel):
    code: str
    message: str
    severity: str = "warning"


class TranscriptMetadata(BaseModel):
    quality: str | None = None
    model_size: str | None = None
    device: str | None = None
    compute_type: str | None = None
    beam_size: int | None = None
    auto_retry_enabled: bool = False
    retry_attempted: bool = False
    retry_from_quality: str | None = None
    retry_to_quality: str | None = None


class CompressionResponse(BaseModel):
    video_id: str
    query: str
    answer: str | None = None
    answer_provider: str | None = None
    preset: CompressionPreset
    query_intent: QueryIntent | None = None
    routing_reason: str | None = None
    audio_scorer_used: AudioScoringMode | None = None
    transcript_metadata: TranscriptMetadata | None = None
    query_aspects: list[QueryAspect] = Field(default_factory=list)
    selected: list[SelectedCandidate]
    metrics: CompressionMetrics
    quality_warnings: list[QualityWarning] = Field(default_factory=list)
