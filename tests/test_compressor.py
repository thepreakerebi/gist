from gist.core.compressor import GistCompressor
from gist.core.presets import CompressionPreset
from gist.core.query_intent import QueryIntent
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
    assert response.metrics.estimated_baseline_tokens > 0
    assert response.metrics.estimated_compressed_tokens > 0
    assert response.metrics.estimated_saved_tokens == 0
    assert response.query_intent == QueryIntent.MIXED_AV
    assert response.routing_reason
    assert all(item.reason for item in response.selected)
    assert {item.source_score_type for item in response.selected} == {"lexical_overlap"}


def test_compressor_preserves_scene_metadata_on_selected_candidates() -> None:
    request = CompressionRequest(
        video_id="demo",
        query="show the person",
        duration_seconds=120,
        preset=CompressionPreset.AGGRESSIVE,
        visual_candidates=[
            Candidate(
                id="v-person",
                timestamp_seconds=42,
                text="person enters the frame",
                segment_id="scene-2",
                scene_start_seconds=36,
                scene_end_seconds=50,
            ),
        ],
    )

    response = GistCompressor().compress(request)

    assert response.query_intent == QueryIntent.VISUAL_OBJECT_ACTION
    assert response.selected[0].segment_id == "scene-2"
    assert response.selected[0].scene_start_seconds == 36
    assert response.selected[0].scene_end_seconds == 50


def test_aggressive_preset_caps_selected_candidates() -> None:
    request = CompressionRequest(
        video_id="demo",
        query="goal",
        duration_seconds=300,
        preset=CompressionPreset.AGGRESSIVE,
        visual_candidates=[
            Candidate(id=f"v-{index}", timestamp_seconds=float(index * 20), text="goal replay")
            for index in range(10)
        ],
        audio_candidates=[
            Candidate(
                id=f"a-{index}",
                timestamp_seconds=float(220 + (index * 20)),
                text="crowd goal noise",
            )
            for index in range(10)
        ],
    )

    response = GistCompressor().compress(request)

    assert response.metrics.selected_candidates == 6
    assert response.metrics.estimated_candidate_reduction_ratio == 0.3
    assert response.metrics.estimated_candidate_reduction_percent == 70
    assert response.metrics.dropped_candidates == 14
    assert response.metrics.estimated_token_reduction_percent > 0


def test_scene_aware_selection_preserves_relevant_scene_coverage() -> None:
    request = CompressionRequest(
        video_id="demo",
        query="goal replay",
        duration_seconds=300,
        preset=CompressionPreset.AGGRESSIVE,
        visual_candidates=[
            Candidate(
                id=f"v-a-{index}",
                timestamp_seconds=float(index),
                text="goal replay close angle",
                segment_id="scene-a",
                scene_start_seconds=0,
                scene_end_seconds=20,
            )
            for index in range(8)
        ]
        + [
            Candidate(
                id="v-b-0",
                timestamp_seconds=140,
                text="goal replay wide angle",
                segment_id="scene-b",
                scene_start_seconds=130,
                scene_end_seconds=150,
            )
        ],
    )

    response = GistCompressor().compress(request)

    assert {item.segment_id for item in response.selected} >= {"scene-a", "scene-b"}


def test_compressor_suppresses_redundant_neighboring_audio_evidence() -> None:
    request = CompressionRequest(
        video_id="demo",
        query="architecture missions",
        duration_seconds=120,
        preset=CompressionPreset.BALANCED,
        audio_candidates=[
            Candidate(
                id="a-1",
                timestamp_seconds=50,
                text="to expand what is possible the architecture",
            ),
            Candidate(
                id="a-2",
                timestamp_seconds=54,
                text="the architecture these missions are taking shape",
            ),
            Candidate(
                id="a-3",
                timestamp_seconds=58,
                text="these missions are taking shape with new systems",
            ),
            Candidate(
                id="a-4",
                timestamp_seconds=100,
                text="closing remarks about exploration",
            ),
        ],
    )

    response = GistCompressor().compress(request)
    selected_ids = {item.id for item in response.selected}

    assert "a-2" in selected_ids
    assert not {"a-1", "a-2", "a-3"}.issubset(selected_ids)


def test_compressor_boosts_visual_evidence_near_relevant_audio() -> None:
    request = CompressionRequest(
        video_id="demo",
        query="architecture missions",
        duration_seconds=120,
        preset=CompressionPreset.AGGRESSIVE,
        visual_candidates=[
            Candidate(id="v-start", timestamp_seconds=0, text="visual frame sampled at 0 seconds"),
            Candidate(id="v-near", timestamp_seconds=58, text="visual frame sampled at 58 seconds"),
            Candidate(
                id="v-end",
                timestamp_seconds=118,
                text="visual frame sampled at 118 seconds",
            ),
        ],
        audio_candidates=[
            Candidate(
                id="a-hit",
                timestamp_seconds=58,
                text="the architecture for these missions is taking shape",
            ),
            Candidate(id="a-end", timestamp_seconds=118, text="closing remarks"),
        ],
    )

    response = GistCompressor().compress(request)
    selected_by_id = {item.id: item for item in response.selected}

    assert "v-near" in selected_by_id
    assert selected_by_id["v-near"].audio_anchor_timestamp_seconds == 58
    assert selected_by_id["v-near"].audio_anchor_score > 0.9
    assert "near relevant audio evidence" in selected_by_id["v-near"].reason


def test_cross_modal_anchor_pair_is_not_treated_as_redundant() -> None:
    request = CompressionRequest(
        video_id="demo",
        query="architecture missions",
        duration_seconds=120,
        preset=CompressionPreset.AGGRESSIVE,
        visual_candidates=[
            Candidate(
                id="v-near",
                timestamp_seconds=58,
                text="visual frame sampled near the architecture moment",
            ),
            Candidate(
                id="v-far",
                timestamp_seconds=110,
                text="visual frame sampled far away",
            ),
        ],
        audio_candidates=[
            Candidate(
                id="a-hit",
                timestamp_seconds=58,
                text="the architecture for these missions is taking shape",
            ),
        ],
    )

    response = GistCompressor().compress(request)
    selected_ids = {item.id for item in response.selected}

    assert {"a-hit", "v-near"}.issubset(selected_ids)


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


def test_counting_query_keeps_neighboring_visual_frames() -> None:
    request = CompressionRequest(
        video_id="demo",
        query="how many red socks are above the fireplace",
        duration_seconds=100,
        preset=CompressionPreset.AGGRESSIVE,
        adaptive_budget=True,
        task_aware_selection=True,
        visual_candidates=[
            Candidate(id="v-0", timestamp_seconds=0, text="intro"),
            Candidate(id="v-1", timestamp_seconds=30, text="fireplace with socks"),
            Candidate(id="v-2", timestamp_seconds=38, text="fireplace close view"),
            Candidate(id="v-3", timestamp_seconds=70, text="closing"),
        ],
        audio_candidates=[
            Candidate(id="a-1", timestamp_seconds=31, text="look at the red socks"),
            Candidate(id="a-2", timestamp_seconds=80, text="closing remarks"),
        ],
    )

    response = GistCompressor().compress(request)
    selected_ids = {item.id for item in response.selected}

    assert response.query_intent == QueryIntent.COUNTING_COMPARISON
    assert {"v-1", "v-2"}.issubset(selected_ids)
    assert response.metrics.visual_selected >= 3


def test_visual_query_reserves_budget_for_direct_visual_evidence() -> None:
    request = CompressionRequest(
        video_id="demo",
        query="show the robot hand on screen",
        duration_seconds=120,
        preset=CompressionPreset.AGGRESSIVE,
        task_aware_selection=True,
        visual_candidates=[
            Candidate(id="v-hand", timestamp_seconds=10, text="robot hand close-up"),
            Candidate(id="v-person", timestamp_seconds=20, text="person reacting"),
            Candidate(id="v-room", timestamp_seconds=30, text="room wide shot"),
        ],
        audio_candidates=[
            Candidate(
                id=f"a-{index}",
                timestamp_seconds=float(40 + index),
                text="speaker says robot hand on screen",
            )
            for index in range(8)
        ],
    )

    response = GistCompressor().compress(request)

    assert response.query_intent == QueryIntent.VISUAL_OBJECT_ACTION
    assert response.metrics.visual_selected >= 3
    assert "v-hand" in {item.id for item in response.selected}


def test_opening_visual_query_keeps_earliest_ocr_frame() -> None:
    request = CompressionRequest(
        video_id="demo",
        query="What course title appears on the opening lecture slide?",
        duration_seconds=4500,
        preset=CompressionPreset.AGGRESSIVE,
        task_aware_selection=True,
        visual_candidates=[
            Candidate(
                id="opening-title",
                timestamp_seconds=0,
                text="on-screen text near 0 seconds: Bio-Inspired Motor Control",
                saliency_score=0.1,
                segment_id="opening",
            ),
            Candidate(
                id="later-title",
                timestamp_seconds=638,
                text="on-screen text near 638 seconds: Legged Locomotion in Nature",
                saliency_score=0.95,
                segment_id="later",
            ),
            Candidate(
                id="closing-title",
                timestamp_seconds=4400,
                text="on-screen text near 4400 seconds: Summary",
                saliency_score=0.8,
                segment_id="closing",
            ),
        ],
    )

    response = GistCompressor().compress(request)

    assert response.query_intent == QueryIntent.VISUAL_OBJECT_ACTION
    assert "opening-title" in {item.id for item in response.selected}


def test_temporal_query_keeps_anchor_and_directional_target() -> None:
    request = CompressionRequest(
        video_id="demo",
        query="What title appears after the editing interface?",
        duration_seconds=3600,
        preset=CompressionPreset.AGGRESSIVE,
        task_aware_selection=True,
        visual_candidates=[
            Candidate(
                id="opening-title",
                timestamp_seconds=10,
                text="KINECT opening title",
                saliency_score=0.95,
                temporal_anchor_score=0.1,
                temporal_target_score=0.95,
                temporal_direction="after",
            ),
            Candidate(
                id="editing-interface",
                timestamp_seconds=3500,
                text="video editing interface",
                temporal_anchor_score=0.95,
                temporal_target_score=0.1,
                temporal_direction="after",
            ),
            Candidate(
                id="successor-title",
                timestamp_seconds=3510,
                text="KINECT for Windows",
                temporal_anchor_score=0.1,
                temporal_target_score=0.9,
                temporal_direction="after",
            ),
        ],
    )

    response = GistCompressor().compress(request)

    assert response.query_intent == QueryIntent.TEMPORAL_BEFORE_AFTER
    assert {"editing-interface", "successor-title"}.issubset(
        {item.id for item in response.selected}
    )


def test_negative_query_prefers_audio_coverage() -> None:
    request = CompressionRequest(
        video_id="demo",
        query="which of these is not discussed in the video? Choices: inkstone niche jade table",
        duration_seconds=100,
        preset=CompressionPreset.AGGRESSIVE,
        adaptive_budget=True,
        task_aware_selection=True,
        visual_candidates=[
            Candidate(id="v-1", timestamp_seconds=10, text="tomb chamber"),
            Candidate(id="v-2", timestamp_seconds=50, text="stone table"),
            Candidate(id="v-3", timestamp_seconds=90, text="closing title"),
        ],
        audio_candidates=[
            Candidate(id="a-1", timestamp_seconds=11, text="the inkstone was found"),
            Candidate(id="a-2", timestamp_seconds=30, text="a niche was visible"),
            Candidate(id="a-3", timestamp_seconds=70, text="a sacrificial table was preserved"),
        ],
    )

    response = GistCompressor().compress(request)

    assert response.query_intent == QueryIntent.NEGATIVE_EVIDENCE
    assert response.metrics.audio_selected >= 3


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


def test_adaptive_budget_expands_when_anchored_visuals_crowd_out_audio() -> None:
    request = CompressionRequest(
        video_id="demo",
        query="architecture missions",
        duration_seconds=120,
        preset=CompressionPreset.BALANCED,
        adaptive_budget=True,
        visual_candidates=[
            Candidate(
                id=f"v-{index}",
                timestamp_seconds=float(50 + index),
                text=f"architecture missions visual frame sampled at {50 + index} seconds",
            )
            for index in range(8)
        ],
        audio_candidates=[
            Candidate(
                id="a-hit",
                timestamp_seconds=54,
                text="the architecture for these missions is taking shape",
            ),
            Candidate(id="a-context", timestamp_seconds=90, text="mission context continues"),
        ],
    )

    response = GistCompressor().compress(request)

    assert response.preset == CompressionPreset.BALANCED
    assert response.metrics.budget_expanded is True
    assert (
        response.metrics.expansion_reason
        == "aggressive budget underrepresented source audio evidence"
    )
    assert response.metrics.audio_selected >= 2


def test_mixed_av_selection_keeps_nearby_audio_and_visual_evidence() -> None:
    request = CompressionRequest(
        video_id="demo",
        query=(
            "What does the presenter say about the demonstration "
            "showing a person and an on-screen skeleton?"
        ),
        duration_seconds=3600,
        preset=CompressionPreset.BALANCED,
        task_aware_selection=True,
        query_intent=QueryIntent.MIXED_AV,
        visual_candidates=[
            Candidate(
                id=f"visual-{index}",
                timestamp_seconds=float(100 + index * 100),
                text="visual frame sampled during the body tracking demonstration",
                saliency_score=1.0 - index * 0.01,
            )
            for index in range(12)
        ],
        audio_candidates=[
            Candidate(
                id="audio-near",
                timestamp_seconds=105,
                text="the system tracks the joints of the person in real time",
            ),
            Candidate(
                id="audio-far",
                timestamp_seconds=3000,
                text="unrelated closing remarks",
            ),
        ],
    )

    response = GistCompressor().compress(request)

    assert response.metrics.visual_selected >= 2
    assert response.metrics.audio_selected >= 1
    assert any(item.id == "audio-near" for item in response.selected)


def test_adaptive_budget_keeps_aggressive_for_grounded_transcript_moments() -> None:
    request = CompressionRequest(
        video_id="demo",
        query="why is he afraid of the robot hand",
        duration_seconds=120,
        preset=CompressionPreset.BALANCED,
        adaptive_budget=True,
        audio_candidates=[
            Candidate(
                id=f"demo:audio:{index}+demo:visual:{index}",
                timestamp_seconds=float(index * 20),
                text="he is freaked out by the robot hand",
                asset_path=f"frame-{index}.jpg",
            )
            for index in range(8)
        ],
    )

    response = GistCompressor().compress(request)

    assert response.preset == CompressionPreset.AGGRESSIVE
    assert response.metrics.budget_expanded is False
    assert response.metrics.selected_candidates == 6
