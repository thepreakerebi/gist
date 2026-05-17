from dataclasses import dataclass

from gist.core.schemas import Modality


@dataclass(frozen=True, slots=True)
class TokenEstimateConfig:
    visual_tokens_per_candidate: int = 258
    audio_tokens_per_candidate: int = 32


@dataclass(frozen=True, slots=True)
class TokenEstimate:
    baseline_tokens: int
    compressed_tokens: int
    saved_tokens: int
    reduction_ratio: float
    reduction_percent: float


def estimate_tokens(
    input_visual_candidates: int,
    input_audio_candidates: int,
    selected_modalities: list[Modality],
    config: TokenEstimateConfig = TokenEstimateConfig(),
) -> TokenEstimate:
    baseline_tokens = (
        input_visual_candidates * config.visual_tokens_per_candidate
        + input_audio_candidates * config.audio_tokens_per_candidate
    )
    compressed_tokens = sum(
        config.visual_tokens_per_candidate
        if modality == Modality.VISUAL
        else config.audio_tokens_per_candidate
        for modality in selected_modalities
    )
    saved_tokens = max(baseline_tokens - compressed_tokens, 0)
    reduction_ratio = 0.0 if baseline_tokens == 0 else compressed_tokens / baseline_tokens
    reduction_percent = (1.0 - reduction_ratio) * 100 if baseline_tokens else 0.0

    return TokenEstimate(
        baseline_tokens=baseline_tokens,
        compressed_tokens=compressed_tokens,
        saved_tokens=saved_tokens,
        reduction_ratio=reduction_ratio,
        reduction_percent=reduction_percent,
    )
