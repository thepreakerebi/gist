from dataclasses import dataclass
from enum import StrEnum


class TokenEstimatorProfile(StrEnum):
    GENERIC = "generic"
    GEMINI_DEFAULT = "gemini_default"
    GEMINI_LOW_RES = "gemini_low_res"


@dataclass(frozen=True, slots=True)
class TokenEstimateConfig:
    visual_tokens_per_candidate: int = 258
    audio_tokens_per_candidate: int = 32


TOKEN_ESTIMATE_PROFILES: dict[TokenEstimatorProfile, TokenEstimateConfig] = {
    TokenEstimatorProfile.GENERIC: TokenEstimateConfig(
        visual_tokens_per_candidate=258,
        audio_tokens_per_candidate=32,
    ),
    TokenEstimatorProfile.GEMINI_DEFAULT: TokenEstimateConfig(
        visual_tokens_per_candidate=258,
        audio_tokens_per_candidate=32,
    ),
    TokenEstimatorProfile.GEMINI_LOW_RES: TokenEstimateConfig(
        visual_tokens_per_candidate=66,
        audio_tokens_per_candidate=32,
    ),
}


@dataclass(frozen=True, slots=True)
class TokenEstimate:
    profile: TokenEstimatorProfile
    baseline_tokens: int
    compressed_tokens: int
    saved_tokens: int
    reduction_ratio: float
    reduction_percent: float


def estimate_tokens(
    input_visual_candidates: int,
    input_audio_candidates: int,
    selected_modalities: list[object],
    profile: TokenEstimatorProfile = TokenEstimatorProfile.GENERIC,
    config: TokenEstimateConfig | None = None,
) -> TokenEstimate:
    resolved_config = config or TOKEN_ESTIMATE_PROFILES[profile]
    baseline_tokens = (
        input_visual_candidates * resolved_config.visual_tokens_per_candidate
        + input_audio_candidates * resolved_config.audio_tokens_per_candidate
    )
    compressed_tokens = sum(
        resolved_config.visual_tokens_per_candidate
        if _is_visual_modality(modality)
        else resolved_config.audio_tokens_per_candidate
        for modality in selected_modalities
    )
    saved_tokens = max(baseline_tokens - compressed_tokens, 0)
    reduction_ratio = 0.0 if baseline_tokens == 0 else compressed_tokens / baseline_tokens
    reduction_percent = (1.0 - reduction_ratio) * 100 if baseline_tokens else 0.0

    return TokenEstimate(
        profile=profile,
        baseline_tokens=baseline_tokens,
        compressed_tokens=compressed_tokens,
        saved_tokens=saved_tokens,
        reduction_ratio=reduction_ratio,
        reduction_percent=reduction_percent,
    )


def _is_visual_modality(modality: object) -> bool:
    return str(modality).lower().endswith("visual")
