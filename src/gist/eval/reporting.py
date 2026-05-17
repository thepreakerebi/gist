from gist.eval.schemas import EvalReport


def render_markdown_report(report: EvalReport) -> str:
    lines = [
        "# Gist Evaluation Report",
        "",
        "## Summary",
        "",
        f"- Examples: {report.summary.examples}",
        "",
        "| Variant | Avg Reduction | Avg Timestamp Hit Rate | Avg Latency |",
        "|---|---:|---:|---:|",
    ]
    for name, summary in report.summary.variants.items():
        lines.append(
            f"| {name} | {summary.avg_reduction_percent:.2f}% | "
            f"{summary.avg_timestamp_hit_rate:.2f} | {summary.avg_latency_ms:.2f} ms |"
        )
    lines.extend(
        [
            "",
            "## Examples",
            "",
        ]
    )
    for result in report.results:
        lines.extend(
            [
                f"### {result.id}",
                "",
                f"- Query: {result.query}",
                "",
                "| Variant | Selected | Reduction | Timestamp Hit Rate | Latency |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for variant in result.variants:
            lines.append(
                f"| {variant.name} | {variant.response.metrics.selected_candidates} | "
                f"{variant.response.metrics.estimated_candidate_reduction_percent:.2f}% | "
                f"{variant.timestamp_hit_rate:.2f} | {variant.latency_ms:.2f} ms |"
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"
