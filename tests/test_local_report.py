from pathlib import Path

from gist.core.presets import CompressionPreset
from gist.core.schemas import (
    CompressionMetrics,
    CompressionResponse,
    Modality,
    SelectedCandidate,
)
from gist.core.token_estimation import TokenEstimatorProfile
from gist.media.models import IngestedVideo, IngestionSettings, VideoMetadata
from gist.reports.local import render_local_compression_report


def test_render_local_compression_report_includes_plan_and_video_clip(tmp_path: Path) -> None:
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"fake")
    ingestion = IngestedVideo(
        video_id="video-1",
        source_path=tmp_path / "video.mp4",
        metadata=VideoMetadata(duration_seconds=3600, has_audio=True),
        frames=[],
        audio_windows=[],
        settings=IngestionSettings(
            processing_mode="long",
            sample_count=512,
            audio_window_seconds=30,
            audio_context_window_count=0,
            max_audio_windows=120,
            reason="long processing selected",
        ),
    )
    compression = CompressionResponse(
        video_id="video-1",
        query="refund policy",
        preset=CompressionPreset.BALANCED,
        selected=[
            SelectedCandidate(
                id="a-1",
                modality=Modality.AUDIO,
                timestamp_seconds=120,
                text="refund policy explained",
                clip_path=clip_path,
                clip_start_seconds=116,
                clip_end_seconds=124,
                segment_id="long-segment-0001",
                selection_rank=1,
                relevance_score=0.8,
                normalized_score=1.0,
                mmr_score=0.7,
                source_score_type="lexical_overlap",
                reason="query terms matched",
            )
        ],
        metrics=CompressionMetrics(
            input_candidates=100,
            selected_candidates=1,
            visual_selected=0,
            audio_selected=1,
            estimated_candidate_reduction_ratio=0.01,
            estimated_candidate_reduction_percent=99,
            dropped_candidates=99,
            budget_preset_used=CompressionPreset.BALANCED,
            estimated_baseline_tokens=1000,
            estimated_compressed_tokens=32,
            estimated_saved_tokens=968,
            estimated_token_reduction_ratio=0.032,
            estimated_token_reduction_percent=96.8,
            token_estimator=TokenEstimatorProfile.GENERIC,
        ),
    )

    html = render_local_compression_report(ingestion, compression)

    assert "<html" in html
    assert "refund policy" in html
    assert "long processing selected" in html
    assert "evidence" in html
    assert "<video" in html
    assert clip_path.resolve().as_uri() in html
