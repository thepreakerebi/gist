import json
from pathlib import Path

from gist.eval.frame_sweep import (
    render_frame_sweep_markdown,
    summarize_frame_sweep,
    write_frame_sweep_summary,
)


def test_summarize_frame_sweep_picks_best_accuracy_then_tokens(tmp_path: Path) -> None:
    report_one = _write_report(
        tmp_path / "frames-1.json",
        {
            "gist_core": {
                "avg_answer_score": 0.5,
                "avg_token_reduction_percent": 50.0,
                "avg_latency_ms": 100.0,
            },
            "gist_scene": {
                "avg_answer_score": 0.5,
                "avg_token_reduction_percent": 75.0,
                "avg_latency_ms": 120.0,
            },
        },
    )
    report_four = _write_report(
        tmp_path / "frames-4.json",
        {
            "gist_core": {
                "avg_answer_score": 0.75,
                "avg_token_reduction_percent": 45.0,
                "avg_latency_ms": 180.0,
            }
        },
    )

    summary = summarize_frame_sweep({1: report_one, 4: report_four})

    assert summary["runs"][0]["best_variant"]["variant"] == "gist_scene"
    assert summary["best_overall"]["frame_count"] == 4
    assert summary["best_overall"]["variant"] == "gist_core"


def test_render_frame_sweep_markdown_includes_best_overall() -> None:
    markdown = render_frame_sweep_markdown(
        {
            "runs": [
                {
                    "frame_count": 4,
                    "report_path": "reports/frames-4/sota-report.json",
                    "best_variant": {
                        "frame_count": 4,
                        "variant": "gist_core",
                        "avg_answer_score": 0.75,
                        "avg_token_reduction_percent": 45.0,
                        "avg_latency_ms": 180.0,
                    },
                }
            ],
            "best_overall": {
                "frame_count": 4,
                "variant": "gist_core",
                "avg_answer_score": 0.75,
                "avg_token_reduction_percent": 45.0,
                "avg_latency_ms": 180.0,
            },
        }
    )

    assert "| 4 | gist_core | 0.750 | 45.00% | 180.00 ms |" in markdown
    assert "## Best Overall" in markdown


def test_write_frame_sweep_summary_writes_json_and_markdown(tmp_path: Path) -> None:
    report = _write_report(
        tmp_path / "frames-1.json",
        {
            "gist_core": {
                "avg_answer_score": 0.5,
                "avg_token_reduction_percent": 50.0,
                "avg_latency_ms": 100.0,
            }
        },
    )

    summary = write_frame_sweep_summary(
        report_paths={1: report},
        output_json=tmp_path / "summary.json",
        output_markdown=tmp_path / "summary.md",
    )

    assert summary["best_overall"]["variant"] == "gist_core"
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "summary.md").exists()


def _write_report(path: Path, variants: dict[str, dict[str, float]]) -> Path:
    path.write_text(json.dumps({"summary": {"variants": variants}}))
    return path
