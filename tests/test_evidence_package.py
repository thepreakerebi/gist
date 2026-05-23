from pathlib import Path

from gist.core.presets import CompressionPreset
from gist.core.schemas import CompressionMetrics, CompressionResponse, Modality, SelectedCandidate
from gist.core.token_estimation import TokenEstimatorProfile
from gist.gateway.evidence_package import EVIDENCE_PACKAGE_VERSION, build_evidence_package
from gist.media.models import IngestedVideo, VideoMetadata


def test_build_evidence_package_exports_model_ready_contract(tmp_path: Path) -> None:
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"fake")
    ingestion = IngestedVideo(
        video_id="video-1",
        source_path=tmp_path / "video.mp4",
        metadata=VideoMetadata(duration_seconds=60, width=640, height=360, has_audio=True),
        frames=[],
        audio_windows=[],
    )
    compression = CompressionResponse(
        video_id="video-1",
        query="What happened?",
        answer="The speaker explains the event.",
        answer_provider="local-text-evidence",
        preset=CompressionPreset.BALANCED,
        selected=[
            SelectedCandidate(
                id="a-1+v-1",
                modality=Modality.AUDIO,
                timestamp_seconds=12,
                text="The speaker explains the event.",
                clip_path=clip_path,
                clip_start_seconds=10,
                clip_end_seconds=20,
                selection_rank=1,
                relevance_score=0.8,
                normalized_score=1.0,
                mmr_score=0.7,
                answer_support_score=1.0,
                query_support_score=0.5,
                evidence_support_score=0.85,
                support_label="strong",
                source_score_type="lexical_overlap",
                reason="test",
            )
        ],
        metrics=CompressionMetrics(
            input_candidates=1,
            selected_candidates=1,
            visual_selected=0,
            audio_selected=1,
            estimated_candidate_reduction_ratio=1,
            estimated_candidate_reduction_percent=0,
            dropped_candidates=0,
            budget_preset_used=CompressionPreset.BALANCED,
            token_estimator=TokenEstimatorProfile.GENERIC,
        ),
    )

    package = build_evidence_package(ingestion, compression)

    assert package["schema"] == EVIDENCE_PACKAGE_VERSION
    assert package["query"] == "What happened?"
    assert package["answer_hint"] == "The speaker explains the event."
    assert package["answer_provider"] == "local-text-evidence"
    assert package["evidence"][0]["clip_path"] == str(clip_path)
    assert package["evidence"][0]["transcript"] == "The speaker explains the event."
    assert package["evidence"][0]["support_label"] == "strong"
    assert package["evidence"][0]["evidence_support_score"] == 0.85
    assert "Initial answer hint" not in package["prompt"]
    assert "Return a concise answer" in package["prompt"]


def test_evidence_prompt_marks_visual_only_clips_as_non_textual(tmp_path: Path) -> None:
    ingestion = IngestedVideo(
        video_id="video-1",
        source_path=tmp_path / "video.mp4",
        metadata=VideoMetadata(duration_seconds=60, width=640, height=360, has_audio=True),
        frames=[],
        audio_windows=[],
    )
    compression = CompressionResponse(
        video_id="video-1",
        query="What is shown?",
        preset=CompressionPreset.BALANCED,
        selected=[
            SelectedCandidate(
                id="v-1",
                modality=Modality.VISUAL,
                timestamp_seconds=12,
                text="visual frame sampled at 12.00 seconds",
                clip_start_seconds=10,
                clip_end_seconds=20,
                selection_rank=1,
                relevance_score=0.8,
                normalized_score=1.0,
                mmr_score=0.7,
                source_score_type="lexical_overlap",
                reason="test",
            )
        ],
        metrics=CompressionMetrics(
            input_candidates=1,
            selected_candidates=1,
            visual_selected=1,
            audio_selected=0,
            estimated_candidate_reduction_ratio=1,
            estimated_candidate_reduction_percent=0,
            dropped_candidates=0,
            budget_preset_used=CompressionPreset.BALANCED,
            token_estimator=TokenEstimatorProfile.GENERIC,
        ),
    )

    package = build_evidence_package(ingestion, compression)

    assert "[visual-only clip; no transcript text available]" in package["prompt"]
    assert "cannot inspect pixels" in package["prompt"]


def test_evidence_prompt_omits_visual_only_clips_when_transcripts_exist(
    tmp_path: Path,
) -> None:
    ingestion = IngestedVideo(
        video_id="video-1",
        source_path=tmp_path / "video.mp4",
        metadata=VideoMetadata(duration_seconds=60, width=640, height=360, has_audio=True),
        frames=[],
        audio_windows=[],
    )
    compression = CompressionResponse(
        video_id="video-1",
        query="What happened?",
        preset=CompressionPreset.BALANCED,
        selected=[
            SelectedCandidate(
                id="v-1",
                modality=Modality.VISUAL,
                timestamp_seconds=12,
                text="visual frame sampled at 12.00 seconds",
                selection_rank=1,
                relevance_score=0.8,
                normalized_score=1.0,
                mmr_score=0.7,
                source_score_type="lexical_overlap",
                reason="test",
            ),
            SelectedCandidate(
                id="a-1",
                modality=Modality.AUDIO,
                timestamp_seconds=20,
                text="The founder uses AI to review plans.",
                selection_rank=2,
                relevance_score=0.9,
                normalized_score=1.0,
                mmr_score=0.8,
                source_score_type="lexical_overlap",
                reason="test",
            ),
        ],
        metrics=CompressionMetrics(
            input_candidates=2,
            selected_candidates=2,
            visual_selected=1,
            audio_selected=1,
            estimated_candidate_reduction_ratio=1,
            estimated_candidate_reduction_percent=0,
            dropped_candidates=0,
            budget_preset_used=CompressionPreset.BALANCED,
            token_estimator=TokenEstimatorProfile.GENERIC,
        ),
    )

    package = build_evidence_package(ingestion, compression)

    assert "1. 12.00s" not in package["prompt"]
    assert "2. 20.00s" in package["prompt"]
    assert "The founder uses AI to review plans." in package["prompt"]


def test_evidence_prompt_treats_ocr_visual_text_as_non_transcript_when_audio_exists(
    tmp_path: Path,
) -> None:
    ingestion = IngestedVideo(
        video_id="video-1",
        source_path=tmp_path / "video.mp4",
        metadata=VideoMetadata(duration_seconds=60, width=640, height=360, has_audio=True),
        frames=[],
        audio_windows=[],
    )
    compression = CompressionResponse(
        video_id="video-1",
        query="What does the speaker say?",
        preset=CompressionPreset.BALANCED,
        selected=[
            SelectedCandidate(
                id="v-ocr",
                modality=Modality.VISUAL,
                timestamp_seconds=12,
                text="on-screen text near 12.00 seconds: noisy OCR tokens",
                selection_rank=1,
                relevance_score=0.8,
                normalized_score=1.0,
                mmr_score=0.7,
                source_score_type="lexical_overlap",
                reason="test",
            ),
            SelectedCandidate(
                id="a-1",
                modality=Modality.AUDIO,
                timestamp_seconds=20,
                text="The architecture for these missions is taking shape.",
                selection_rank=2,
                relevance_score=0.9,
                normalized_score=1.0,
                mmr_score=0.8,
                source_score_type="lexical_overlap",
                reason="test",
            ),
        ],
        metrics=CompressionMetrics(
            input_candidates=2,
            selected_candidates=2,
            visual_selected=1,
            audio_selected=1,
            estimated_candidate_reduction_ratio=1,
            estimated_candidate_reduction_percent=0,
            dropped_candidates=0,
            budget_preset_used=CompressionPreset.BALANCED,
            token_estimator=TokenEstimatorProfile.GENERIC,
        ),
    )

    package = build_evidence_package(ingestion, compression)

    assert "1. 12.00s" not in package["prompt"]
    assert "2. 20.00s" in package["prompt"]
    assert "noisy OCR tokens" not in package["prompt"]
