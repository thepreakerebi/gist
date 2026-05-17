from gist.core.compressor import GistCompressor
from gist.core.presets import CompressionPreset
from gist.core.schemas import Candidate, CompressionRequest, Modality


def test_compressor_selects_query_relevant_audio_and_visual_candidates() -> None:
    request = CompressionRequest(
        video_id="demo",
        query="pricing",
        duration_seconds=120,
        preset=CompressionPreset.AGGRESSIVE,
        visual_candidates=[
            Candidate(id="v-title", timestamp_seconds=1, text="intro title slide"),
            Candidate(id="v-pricing", timestamp_seconds=42, text="pricing plan chart"),
        ],
        audio_candidates=[
            Candidate(id="a-pricing", timestamp_seconds=44, text="pricing starts at ten dollars"),
            Candidate(id="a-close", timestamp_seconds=100, text="thanks for watching"),
        ],
    )

    response = GistCompressor().compress(request)
    selected_ids = {item.id for item in response.selected}

    assert "v-pricing" in selected_ids
    assert "a-pricing" in selected_ids
    assert response.metrics.input_candidates == 4
    assert response.metrics.selected_candidates == 4
    assert response.metrics.visual_selected == 2
    assert response.metrics.audio_selected == 2


def test_aggressive_preset_caps_selected_candidates() -> None:
    request = CompressionRequest(
        video_id="demo",
        query="goal",
        duration_seconds=300,
        preset=CompressionPreset.AGGRESSIVE,
        visual_candidates=[
            Candidate(id=f"v-{index}", timestamp_seconds=float(index), text="goal replay")
            for index in range(10)
        ],
        audio_candidates=[
            Candidate(id=f"a-{index}", timestamp_seconds=float(index + 20), text="crowd goal noise")
            for index in range(10)
        ],
    )

    response = GistCompressor().compress(request)

    assert response.metrics.selected_candidates == 6
    assert response.metrics.estimated_candidate_reduction_ratio == 0.3


def test_cross_modal_selection_keeps_modality_metadata() -> None:
    request = CompressionRequest(
        video_id="demo",
        query="speaker",
        duration_seconds=60,
        visual_candidates=[Candidate(id="v-1", timestamp_seconds=3, text="speaker on stage")],
        audio_candidates=[Candidate(id="a-1", timestamp_seconds=4, text="speaker says hello")],
    )

    response = GistCompressor().compress(request)
    modalities = {item.modality for item in response.selected}

    assert modalities == {Modality.VISUAL, Modality.AUDIO}

