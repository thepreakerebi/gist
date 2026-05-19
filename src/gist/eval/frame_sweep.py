import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class FrameSweepVariantSummary:
    frame_count: int
    variant: str
    avg_answer_score: float | None
    avg_token_reduction_percent: float
    avg_latency_ms: float


def summarize_frame_sweep(report_paths: dict[int, Path]) -> dict[str, Any]:
    runs = []
    for frame_count, report_path in sorted(report_paths.items()):
        report = _load_report(report_path)
        variant_summaries = [
            FrameSweepVariantSummary(
                frame_count=frame_count,
                variant=name,
                avg_answer_score=summary.get("avg_answer_score"),
                avg_token_reduction_percent=float(summary["avg_token_reduction_percent"]),
                avg_latency_ms=float(summary["avg_latency_ms"]),
            )
            for name, summary in report["summary"]["variants"].items()
        ]
        runs.append(
            {
                "frame_count": frame_count,
                "report_path": str(report_path),
                "best_variant": asdict(_best_variant(variant_summaries)),
                "variants": [asdict(summary) for summary in variant_summaries],
            }
        )

    return {
        "runs": runs,
        "best_overall": _best_overall(runs),
    }


def render_frame_sweep_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Gist Frame Density Sweep",
        "",
        "| Frames | Best Variant | Answer Score | Token Reduction | Latency | Report |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for run in summary["runs"]:
        best = run["best_variant"]
        lines.append(
            f"| {run['frame_count']} | {best['variant']} | "
            f"{_format_score(best['avg_answer_score'])} | "
            f"{best['avg_token_reduction_percent']:.2f}% | "
            f"{best['avg_latency_ms']:.2f} ms | {run['report_path']} |"
        )

    best_overall = summary.get("best_overall")
    if best_overall:
        lines.extend(
            [
                "",
                "## Best Overall",
                "",
                f"- Frames: {best_overall['frame_count']}",
                f"- Variant: {best_overall['variant']}",
                f"- Answer score: {_format_score(best_overall['avg_answer_score'])}",
                f"- Token reduction: {best_overall['avg_token_reduction_percent']:.2f}%",
                f"- Latency: {best_overall['avg_latency_ms']:.2f} ms",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def write_frame_sweep_summary(
    report_paths: dict[int, Path],
    output_json: Path,
    output_markdown: Path,
) -> dict[str, Any]:
    summary = summarize_frame_sweep(report_paths)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2))
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(render_frame_sweep_markdown(summary))
    return summary


def _load_report(report_path: Path) -> dict[str, Any]:
    if not report_path.exists():
        raise FileNotFoundError(f"report not found: {report_path}")
    payload = json.loads(report_path.read_text())
    if "summary" not in payload or "variants" not in payload["summary"]:
        raise ValueError(f"report does not look like a Gist eval report: {report_path}")
    return payload


def _best_variant(
    variant_summaries: list[FrameSweepVariantSummary],
) -> FrameSweepVariantSummary:
    if not variant_summaries:
        raise ValueError("frame sweep run has no variants")
    return max(
        variant_summaries,
        key=lambda summary: (
            summary.avg_answer_score if summary.avg_answer_score is not None else -1.0,
            summary.avg_token_reduction_percent,
            -summary.avg_latency_ms,
        ),
    )


def _best_overall(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not runs:
        return None
    best_by_run = [run["best_variant"] for run in runs]
    best = max(
        best_by_run,
        key=lambda summary: (
            summary["avg_answer_score"] if summary["avg_answer_score"] is not None else -1.0,
            summary["avg_token_reduction_percent"],
            -summary["avg_latency_ms"],
        ),
    )
    return best


def _format_score(score: float | None) -> str:
    return "n/a" if score is None else f"{score:.3f}"
