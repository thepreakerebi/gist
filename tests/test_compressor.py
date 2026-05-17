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
    assert response.metrics.dropped_candidates == 0
    assert response.metrics.estimated_candidate_reduction_percent == 0
    assert response.metrics.budget_mode == "fixed"
    assert response.metrics.budget_preset_used == CompressionPreset.AGGRESSIVE
    assert response.metrics.budget_expanded is False
    assert all(item.reason for item in response.selected)
    assert {item.source_score_type for item in response.selected} == {"lexical_overlap"}


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
    assert response.metrics.estimated_candidate_reduction_percent == 70
    assert response.metrics.dropped_candidates == 14


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


def test_model_saliency_candidates_report_source_score_type() -> None:
    request = CompressionRequest(
        video_id="demo",
        query="applause",
        duration_seconds=10,
        audio_candidates=[
            Candidate(
                id="a-1",
                timestamp_seconds=1,
                text="audio event",
                saliency_score=0.82,
            )
        ],
    )

    response = GistCompressor().compress(request)

    assert response.selected[0].source_score_type == "model_saliency"
    assert response.selected[0].selection_rank == 1
    assert response.selected[0].mmr_score == 0


def test_decomposed_query_reports_query_aspects_and_reasons() -> None:
    request = CompressionRequest(
        video_id="demo",
        query="show the person in red shirt and what does the speaker say",
        duration_seconds=60,
        decompose_query=True,
        visual_candidates=[
            Candidate(id="v-1", timestamp_seconds=3, text="person in red shirt"),
        ],
        audio_candidates=[
            Candidate(id="a-1", timestamp_seconds=4, text="speaker says pricing details"),
        ],
    )

    response = GistCompressor().compress(request)

    assert [aspect.text for aspect in response.query_aspects] == [
        "show the person in red shirt",
        "what does the speaker say",
    ]
    assert all("aspect" in item.reason for item in response.selected)


def test_adaptive_budget_uses_aggressive_preset_when_evidence_is_good() -> None:
    request = CompressionRequest(
        video_id="demo",
        query="pricing",
        duration_seconds=120,
        preset=CompressionPreset.BALANCED,
        adaptive_budget=True,
        visual_candidates=[
            Candidate(id=f"v-{index}", timestamp_seconds=float(index), text="pricing slide")
            for index in range(10)
        ],
        audio_candidates=[
            Candidate(
                id=f"a-{index}",
                timestamp_seconds=float(index + 20),
                text="speaker explains pricing",
            )
            for index in range(10)
        ],
    )

    response = GistCompressor().compress(request)

    assert response.preset == CompressionPreset.AGGRESSIVE
    assert response.metrics.budget_mode == "adaptive"
    assert response.metrics.budget_expanded is False
    assert response.metrics.selected_candidates == 6


def test_adaptive_budget_expands_when_aggressive_selection_has_low_relevance() -> None:
    request = CompressionRequest(
        video_id="demo",
        query="pricing",
        duration_seconds=120,
        preset=CompressionPreset.BALANCED,
        adaptive_budget=True,
        visual_candidates=[
            Candidate(id=f"v-{index}", timestamp_seconds=float(index), text="unrelated scene")
            for index in range(10)
        ],
        audio_candidates=[
            Candidate(id=f"a-{index}", timestamp_seconds=float(index + 20), text="ambient noise")
            for index in range(10)
        ],
    )

    response = GistCompressor().compress(request)

    assert response.preset == CompressionPreset.BALANCED
    assert response.metrics.budget_expanded is True
    assert response.metrics.expansion_reason == "low best relevance at aggressive budget"
    assert response.metrics.selected_candidates == 12
