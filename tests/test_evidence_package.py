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
    assert package["evidence"][0]["clip_path"] == str(clip_path)
    assert package["evidence"][0]["transcript"] == "The speaker explains the event."
    assert "Return a concise answer" in package["prompt"]
