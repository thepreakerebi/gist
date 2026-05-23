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
        answer="Refunds are available for eligible plans.",
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
                answer_support_score=0.8,
                query_support_score=0.7,
                evidence_support_score=0.77,
                audio_support_score=0.77,
                ocr_support_score=0.0,
                visual_support_score=0.0,
                cross_modal_support_score=0.0,
                support_label="strong",
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
    assert "Refunds are available for eligible plans." in html
    assert "long processing selected" in html
    assert "strong support" in html
    assert "answer_support=0.800" in html
    assert "audio_support=0.770" in html
    assert "evidence" in html
    assert "<video" in html
    assert clip_path.resolve().as_uri() in html


def test_render_local_compression_report_groups_audio_and_visual_into_video_moment(
    tmp_path: Path,
) -> None:
    audio_clip_path = tmp_path / "audio-source-video.mp4"
    visual_clip_path = tmp_path / "visual-source-video.mp4"
    audio_clip_path.write_bytes(b"fake")
    visual_clip_path.write_bytes(b"fake")
    ingestion = IngestedVideo(
        video_id="video-1",
        source_path=tmp_path / "video.mp4",
        metadata=VideoMetadata(duration_seconds=3600, has_audio=True),
        frames=[],
        audio_windows=[],
        settings=None,
    )
    compression = CompressionResponse(
        video_id="video-1",
        query="why is he afraid of the robot hand",
        preset=CompressionPreset.BALANCED,
        selected=[
            SelectedCandidate(
                id="v-1",
                modality=Modality.VISUAL,
                timestamp_seconds=344,
                text="visual frame sampled at 344 seconds",
                clip_path=visual_clip_path,
                clip_start_seconds=338,
                clip_end_seconds=350,
                audio_anchor_timestamp_seconds=345,
                audio_anchor_score=0.98,
                selection_rank=1,
                relevance_score=0.2,
                normalized_score=1.0,
                mmr_score=0.7,
                source_score_type="lexical_overlap",
                reason="anchored visual",
            ),
            SelectedCandidate(
                id="a-1",
                modality=Modality.AUDIO,
                timestamp_seconds=345,
                text="He is freaked out by my robot hand.",
                clip_path=audio_clip_path,
                clip_start_seconds=340,
                clip_end_seconds=350,
                selection_rank=2,
                relevance_score=0.8,
                normalized_score=1.2,
                mmr_score=0.9,
                source_score_type="lexical_overlap",
                reason="query terms matched",
            ),
        ],
        metrics=CompressionMetrics(
            input_candidates=100,
            selected_candidates=2,
            visual_selected=1,
            audio_selected=1,
            estimated_candidate_reduction_ratio=0.02,
            estimated_candidate_reduction_percent=98,
            dropped_candidates=98,
            budget_preset_used=CompressionPreset.BALANCED,
            estimated_baseline_tokens=1000,
            estimated_compressed_tokens=64,
            estimated_saved_tokens=936,
            estimated_token_reduction_ratio=0.064,
            estimated_token_reduction_percent=93.6,
            token_estimator=TokenEstimatorProfile.GENERIC,
        ),
    )

    html = render_local_compression_report(ingestion, compression)

    assert "1</div><div class=\"muted\">video evidence moments" in html
    assert "Video evidence 1" in html
    assert "<strong>video</strong>" in html
    assert "<strong>audio</strong>" not in html
    assert audio_clip_path.resolve().as_uri() in html
    assert visual_clip_path.resolve().as_uri() not in html
    assert "He is freaked out by my robot hand." in html


def test_render_local_compression_report_hides_visual_only_and_caps_moments(
    tmp_path: Path,
) -> None:
    ingestion = IngestedVideo(
        video_id="video-1",
        source_path=tmp_path / "video.mp4",
        metadata=VideoMetadata(duration_seconds=3600, has_audio=True),
        frames=[],
        audio_windows=[],
        settings=None,
    )
    selected = []
    transcripts = [
        "refund policy overview",
        "launch sequence explanation",
        "battery replacement steps",
        "patient intake workflow",
        "warehouse safety checklist",
        "robot hand fear answer",
        "invoice approval process",
        "thermal camera calibration",
    ]
    for index in range(8):
        clip_path = tmp_path / f"audio-{index}.mp4"
        clip_path.write_bytes(b"fake")
        selected.append(
            SelectedCandidate(
                id=f"a-{index}",
                modality=Modality.AUDIO,
                timestamp_seconds=100 + (index * 60),
                text=transcripts[index],
                clip_path=clip_path,
                clip_start_seconds=96 + (index * 60),
                clip_end_seconds=104 + (index * 60),
                selection_rank=index + 1,
                relevance_score=0.1 + index,
                normalized_score=1.0,
                mmr_score=0.7,
                source_score_type="lexical_overlap",
                reason="query terms matched",
            )
        )
    visual_clip_path = tmp_path / "visual-only.mp4"
    visual_clip_path.write_bytes(b"fake")
    selected.append(
        SelectedCandidate(
            id="v-only",
            modality=Modality.VISUAL,
            timestamp_seconds=900,
            text="visual frame sampled",
            clip_path=visual_clip_path,
            clip_start_seconds=896,
            clip_end_seconds=904,
            selection_rank=9,
            relevance_score=99,
            normalized_score=1.0,
            mmr_score=0.7,
            source_score_type="lexical_overlap",
            reason="visual only",
        )
    )
    compression = CompressionResponse(
        video_id="video-1",
        query="question",
        preset=CompressionPreset.BALANCED,
        selected=selected,
        metrics=CompressionMetrics(
            input_candidates=100,
            selected_candidates=len(selected),
            visual_selected=1,
            audio_selected=8,
            estimated_candidate_reduction_ratio=0.09,
            estimated_candidate_reduction_percent=91,
            dropped_candidates=91,
            budget_preset_used=CompressionPreset.BALANCED,
            estimated_baseline_tokens=1000,
            estimated_compressed_tokens=64,
            estimated_saved_tokens=936,
            estimated_token_reduction_ratio=0.064,
            estimated_token_reduction_percent=93.6,
            token_estimator=TokenEstimatorProfile.GENERIC,
        ),
    )

    html = render_local_compression_report(ingestion, compression)

    assert "6</div><div class=\"muted\">video evidence moments" in html
    assert html.count("Video evidence") == 6
    assert "visual frame sampled" not in html
    assert visual_clip_path.resolve().as_uri() not in html
    assert transcripts[0] not in html
    assert transcripts[7] in html
