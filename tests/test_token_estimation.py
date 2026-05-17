from gist.core.schemas import Modality
from gist.core.token_estimation import (
    TokenEstimateConfig,
    TokenEstimatorProfile,
    estimate_tokens,
)


def test_estimate_tokens_uses_modality_specific_weights() -> None:
    estimate = estimate_tokens(
        input_visual_candidates=2,
        input_audio_candidates=2,
        selected_modalities=[Modality.VISUAL, Modality.AUDIO],
        config=TokenEstimateConfig(
            visual_tokens_per_candidate=100,
            audio_tokens_per_candidate=10,
        ),
    )

    assert estimate.baseline_tokens == 220
    assert estimate.compressed_tokens == 110
    assert estimate.saved_tokens == 110
    assert estimate.reduction_percent == 50


def test_estimate_tokens_supports_gemini_low_res_profile() -> None:
    estimate = estimate_tokens(
        input_visual_candidates=1,
        input_audio_candidates=1,
        selected_modalities=[Modality.VISUAL],
        profile=TokenEstimatorProfile.GEMINI_LOW_RES,
    )

    assert estimate.profile == TokenEstimatorProfile.GEMINI_LOW_RES
    assert estimate.baseline_tokens == 98
    assert estimate.compressed_tokens == 66
