import json
from pathlib import Path

from gist.core.modes import AudioScoringMode
from gist.eval.long_video_smoke import (
    evaluate_long_video_smoke,
    main,
    render_long_video_smoke_html,
    render_long_video_smoke_markdown,
)
from gist.eval.quality import QualityCase


def test_long_video_smoke_gates_duration_routing_and_quality(tmp_path: Path) -> None:
    compression_path = _write_run_artifact(tmp_path, duration_seconds=3700)
    quality_case = QualityCase(
        id="long-video",
        compression_path=compression_path,
        expected_answer_terms=["builders"],
        expected_evidence_terms=["builders"],
        relevant_ranges=[{"start_seconds": 350, "end_seconds": 380}],
        min_answer_term_recall=1,
        min_evidence_term_coverage=1,
        min_evidence_relevance_rate=1,
        min_timestamp_hit_rate=1,
        min_grounded_evidence_rate=1,
        min_token_reduction_percent=90,
        max_selected_evidence=1,
    )

    report = evaluate_long_video_smoke(
        compression_path=compression_path,
        quality_case=quality_case,
        expected_audio_scorer=AudioScoringMode.WHISPER,
    )

    assert report.passed
    assert report.routing_passed
    assert "Status: pass" in render_long_video_smoke_markdown(report)
    assert "<strong>Status:</strong> pass" in render_long_video_smoke_html(report)


def test_long_video_smoke_reports_duration_and_routing_failures(tmp_path: Path) -> None:
    compression_path = _write_run_artifact(tmp_path, duration_seconds=1800)
    quality_case = QualityCase(
        id="short-video",
        compression_path=compression_path,
        expected_answer_terms=["builders"],
        expected_evidence_terms=["builders"],
        relevant_ranges=[{"start_seconds": 350, "end_seconds": 380}],
        min_grounded_evidence_rate=1,
        min_token_reduction_percent=90,
    )

    report = evaluate_long_video_smoke(
        compression_path=compression_path,
        quality_case=quality_case,
        expected_audio_scorer=AudioScoringMode.BASELINE,
    )

    assert not report.passed
    assert not report.routing_passed
    assert any("video duration" in failure for failure in report.failures)
    assert any("audio scorer" in failure for failure in report.failures)


def test_long_video_smoke_cli_replays_artifact_and_writes_reports(
    tmp_path: Path,
    capsys,
) -> None:
    compression_path = _write_run_artifact(tmp_path, duration_seconds=3700)
    report_dir = tmp_path / "reports"

    exit_code = main(
        [
            "--compression",
            str(compression_path),
            "--expected-answer-term",
            "builders",
            "--expected-evidence-term",
            "builders",
            "--relevant-range",
            "350:380",
            "--expect-audio-scorer",
            "whisper",
            "--report-dir",
            str(report_dir),
        ]
    )

    assert exit_code == 0
    assert "passed=yes" in capsys.readouterr().out
    assert (report_dir / "long-video-smoke.json").exists()
    assert (report_dir / "long-video-smoke.md").exists()
    assert (report_dir / "long-video-smoke.html").exists()


def _write_run_artifact(tmp_path: Path, duration_seconds: float) -> Path:
    path = tmp_path / "compression.json"
    path.write_text(
        json.dumps(
            {
                "ingestion": {
                    "video_id": "long-video",
                    "source_path": "video.mp4",
                    "metadata": {
                        "duration_seconds": duration_seconds,
                        "has_audio": True,
                    },
                    "frames": [],
                    "audio_windows": [],
                },
                "compression": {
                    "video_id": "long-video",
                    "query": "How do builders use AI?",
                    "answer": "Builders use AI to multiply their output.",
                    "preset": "balanced",
                    "audio_scorer_used": "whisper",
                    "selected": [
                        {
                            "id": "audio-1",
                            "modality": "audio",
                            "timestamp_seconds": 365,
                            "text": "builders use AI to multiply their output",
                            "clip_start_seconds": 350,
                            "clip_end_seconds": 380,
                            "selection_rank": 1,
                            "relevance_score": 1,
                            "normalized_score": 1,
                            "mmr_score": 1,
                            "answer_support_score": 1,
                            "query_support_score": 1,
                            "evidence_support_score": 1,
                            "audio_support_score": 1,
                            "support_label": "direct",
                            "grounding_label": "direct_transcript",
                            "source_score_type": "whisper",
                            "reason": "direct transcript support",
                        }
                    ],
                    "metrics": {
                        "input_candidates": 100,
                        "selected_candidates": 1,
                        "visual_selected": 0,
                        "audio_selected": 1,
                        "estimated_candidate_reduction_ratio": 0.01,
                        "estimated_candidate_reduction_percent": 99,
                        "dropped_candidates": 99,
                        "budget_preset_used": "balanced",
                        "estimated_baseline_tokens": 3200,
                        "estimated_compressed_tokens": 32,
                        "estimated_saved_tokens": 3168,
                        "estimated_token_reduction_ratio": 0.01,
                        "estimated_token_reduction_percent": 99,
                    },
                },
            }
        )
        + "\n"
    )
    return path
