import json
from argparse import Namespace
from pathlib import Path

import gist.cli as cli
from gist.audio.whisper import TranscriptQuality
from gist.cli import (
    _attach_spatial_masks,
    _clear_previous_clips,
    _should_retry_transcripts,
    _with_retry_metadata,
)
from gist.core.modes import AudioScoringMode
from gist.core.presets import CompressionPreset
from gist.core.query_intent import QueryIntent
from gist.core.schemas import (
    CompressionMetrics,
    CompressionResponse,
    Modality,
    QualityWarning,
    SelectedCandidate,
    TranscriptMetadata,
)
from gist.core.token_estimation import TokenEstimatorProfile
from gist.media.models import IngestedVideo, VideoMetadata


def test_clear_previous_clips_removes_stale_mp4_files_only(tmp_path: Path) -> None:
    clips = tmp_path / "clips"
    clips.mkdir()
    stale = clips / "old.mp4"
    keep = clips / "notes.txt"
    stale.write_bytes(b"old")
    keep.write_text("keep")

    _clear_previous_clips(clips)

    assert stale.exists() is False
    assert keep.exists() is True


def test_attach_spatial_masks_writes_masks_for_visual_evidence(tmp_path: Path) -> None:
    compression = CompressionResponse(
        video_id="demo",
        query="show robot hand",
        preset=CompressionPreset.BALANCED,
        selected=[
            SelectedCandidate(
                id="visual-1",
                modality=Modality.VISUAL,
                timestamp_seconds=4,
                text="visual frame sampled at 4.00 seconds",
                asset_path=tmp_path / "frame.jpg",
                selection_rank=1,
                relevance_score=0.5,
                normalized_score=1,
                mmr_score=0.7,
                source_score_type="test",
                reason="selected",
            ),
            SelectedCandidate(
                id="audio-1",
                modality=Modality.AUDIO,
                timestamp_seconds=8,
                text="robot hand mentioned",
                selection_rank=2,
                relevance_score=0.4,
                normalized_score=0.8,
                mmr_score=0.6,
                source_score_type="test",
                reason="selected",
            ),
        ],
        metrics=CompressionMetrics(
            input_candidates=10,
            selected_candidates=2,
            visual_selected=1,
            audio_selected=1,
            estimated_candidate_reduction_ratio=0.2,
            estimated_candidate_reduction_percent=80,
            dropped_candidates=8,
            budget_preset_used=CompressionPreset.BALANCED,
            token_estimator=TokenEstimatorProfile.GENERIC,
        ),
    )
    (tmp_path / "frame.jpg").write_bytes(b"fake-image")

    with_masks = _attach_spatial_masks(
        compression=compression,
        output_dir=tmp_path / "spatial",
        grid_size=4,
        retention_ratio=0.25,
    )

    assert with_masks.selected[0].spatial_mask_path is not None
    assert with_masks.selected[0].spatial_mask_path.exists()
    assert with_masks.selected[0].spatial_mask_preview_path is not None
    assert with_masks.selected[0].spatial_mask_preview_path.exists()
    assert with_masks.selected[0].spatial_mask_overlay_path is not None
    assert with_masks.selected[0].spatial_mask_overlay_path.exists()
    assert with_masks.selected[1].spatial_mask_path is None
    assert with_masks.selected[1].spatial_mask_preview_path is None
    assert with_masks.selected[1].spatial_mask_overlay_path is None
    assert with_masks.metrics.estimated_spatial_visual_tokens == 16
    assert with_masks.metrics.estimated_retained_spatial_visual_tokens == 4
    assert with_masks.metrics.estimated_spatial_token_reduction_percent == 75


def test_main_cli_accepts_builtin_extraction_schema_name(tmp_path: Path, monkeypatch) -> None:
    class FakePipeline:
        def __init__(self, output_root: Path) -> None:
            self.output_root = output_root

        def run(self, **_kwargs):
            ingestion = IngestedVideo(
                video_id="video",
                source_path=tmp_path / "video.mp4",
                metadata=VideoMetadata(duration_seconds=60, has_audio=True),
                frames=[],
                audio_windows=[],
            )
            compression = CompressionResponse(
                video_id="video",
                query="find customer objections",
                preset=CompressionPreset.BALANCED,
                selected=[
                    SelectedCandidate(
                        id="a-1",
                        modality=Modality.AUDIO,
                        timestamp_seconds=30,
                        text="The buyer says pricing is too expensive.",
                        clip_start_seconds=25,
                        clip_end_seconds=45,
                        selection_rank=1,
                        relevance_score=1,
                        normalized_score=1,
                        mmr_score=1,
                        source_score_type="test",
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
                ),
            )
            return ingestion, compression

    monkeypatch.setattr(cli, "LocalCompressionPipeline", FakePipeline)
    output_root = tmp_path / "runs"
    extraction_output = tmp_path / "extraction.json"
    extraction_csv_output = tmp_path / "extraction.csv"

    exit_code = cli.main(
        [
            str(tmp_path / "video.mp4"),
            "--query",
            "find customer objections",
            "--output-root",
            str(output_root),
            "--no-clips",
            "--no-answer-prune",
            "--quiet",
            "--extraction-preset",
            "customer-objections",
            "--extraction-output",
            str(extraction_output),
            "--extraction-csv-output",
            str(extraction_csv_output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(extraction_output.read_text())
    assert payload["schema_name"] == "customer_objections"
    assert payload["items"][0]["label"] == "pricing objection"
    assert "pricing objection" in extraction_csv_output.read_text()


def test_should_retry_transcripts_requires_noisy_whisper_warning() -> None:
    args = Namespace(
        auto_transcript_retry=True,
        whisper_model_size=None,
        whisper_device=None,
        whisper_compute_type=None,
        whisper_beam_size=None,
    )
    compression = _compression_with_warning("noisy_transcript_evidence")

    assert _should_retry_transcripts(
        args=args,
        compression=compression,
        current_quality=TranscriptQuality.FAST,
        retry_quality=TranscriptQuality.ACCURATE,
    )
    assert not _should_retry_transcripts(
        args=args,
        compression=compression.model_copy(update={"quality_warnings": []}),
        current_quality=TranscriptQuality.FAST,
        retry_quality=TranscriptQuality.ACCURATE,
    )


def test_should_retry_transcripts_skips_manual_whisper_overrides() -> None:
    args = Namespace(
        auto_transcript_retry=True,
        whisper_model_size="medium",
        whisper_device=None,
        whisper_compute_type=None,
        whisper_beam_size=None,
    )

    assert not _should_retry_transcripts(
        args=args,
        compression=_compression_with_warning("noisy_transcript_evidence"),
        current_quality=TranscriptQuality.FAST,
        retry_quality=TranscriptQuality.ACCURATE,
    )


def test_with_retry_metadata_records_retry_attempt() -> None:
    compression = _compression_with_warning(None)
    compression = compression.model_copy(
        update={
            "transcript_metadata": TranscriptMetadata(
                quality="accurate",
                model_size="small",
                device="cpu",
                compute_type="int8",
                beam_size=5,
            )
        }
    )

    updated = _with_retry_metadata(
        compression=compression,
        auto_retry_enabled=True,
        retry_attempted=True,
        retry_from_quality=TranscriptQuality.FAST,
        retry_to_quality=TranscriptQuality.ACCURATE,
    )

    assert updated.transcript_metadata is not None
    assert updated.transcript_metadata.auto_retry_enabled is True
    assert updated.transcript_metadata.retry_attempted is True
    assert updated.transcript_metadata.retry_from_quality == "fast"
    assert updated.transcript_metadata.retry_to_quality == "accurate"


def test_main_cli_auto_retries_noisy_transcripts(tmp_path: Path, monkeypatch) -> None:
    calls: list[TranscriptQuality] = []

    class FakePipeline:
        def __init__(self, output_root: Path) -> None:
            self.output_root = output_root

        def run(self, **kwargs):
            quality = kwargs["transcript_quality"]
            calls.append(quality)
            warning_code = (
                "noisy_transcript_evidence"
                if quality == TranscriptQuality.FAST
                else None
            )
            ingestion = IngestedVideo(
                video_id="video",
                source_path=tmp_path / "video.mp4",
                metadata=VideoMetadata(duration_seconds=3600, has_audio=True),
                frames=[],
                audio_windows=[],
            )
            return ingestion, _compression_with_warning(warning_code)

    monkeypatch.setattr(cli, "LocalCompressionPipeline", FakePipeline)

    exit_code = cli.main(
        [
            str(tmp_path / "video.mp4"),
            "--query",
            "What are the main topics covered throughout this lecture?",
            "--output-root",
            str(tmp_path / "runs"),
            "--audio-scorer",
            "whisper",
            "--transcript-quality",
            "fast",
            "--auto-transcript-retry",
            "--transcript-retry-quality",
            "accurate",
            "--no-clips",
            "--no-answer-prune",
            "--quiet",
        ]
    )

    assert exit_code == 0
    assert calls == [TranscriptQuality.FAST, TranscriptQuality.ACCURATE]
    payload = json.loads(
        (
            tmp_path
            / "runs"
            / "video"
            / "what-are-the-main-topics-covered-throughout-this-lecture"
            / "compression.json"
        ).read_text()
    )
    transcript_metadata = payload["compression"]["transcript_metadata"]
    assert transcript_metadata["retry_attempted"] is True
    assert transcript_metadata["retry_from_quality"] == "fast"
    assert transcript_metadata["retry_to_quality"] == "accurate"


def test_main_cli_rejects_schema_path_and_schema_name(tmp_path: Path) -> None:
    try:
        cli.main(
            [
                str(tmp_path / "video.mp4"),
                "--query",
                "find customer objections",
                "--extraction-schema",
                str(tmp_path / "schema.json"),
                "--extraction-preset",
                "customer-objections",
            ]
        )
    except SystemExit as exc:
        assert "Use only one of --extraction-schema" in str(exc)
    else:
        raise AssertionError("expected SystemExit")


def _compression_with_warning(warning_code: str | None) -> CompressionResponse:
    evidence_text = (
        "So this is a bit of a vlog in the place and we have a lot of courses."
        if warning_code == "noisy_transcript_evidence"
        else "sensors and control"
    )
    warnings = (
        [
            QualityWarning(
                code=warning_code,
                message="warning",
            )
        ]
        if warning_code
        else []
    )
    return CompressionResponse(
        video_id="video",
        query="What are the main topics covered throughout this lecture?",
        answer="The video covers: sensors and control.",
        preset=CompressionPreset.BALANCED,
        query_intent=QueryIntent.GLOBAL_SUMMARY,
        audio_scorer_used=AudioScoringMode.WHISPER,
        transcript_metadata={
            "quality": "fast",
            "model_size": "tiny",
            "device": "cpu",
            "compute_type": "int8",
            "beam_size": 1,
        },
        selected=[
            SelectedCandidate(
                id="a-1",
                modality=Modality.AUDIO,
                timestamp_seconds=30,
                text=evidence_text,
                selection_rank=1,
                relevance_score=1,
                normalized_score=1,
                mmr_score=1,
                source_score_type="test",
                reason="test",
                support_label="strong",
                grounding_label="direct",
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
            estimated_token_reduction_percent=99,
        ),
        quality_warnings=warnings,
    )
