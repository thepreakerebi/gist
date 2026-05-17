from gist.eval.schemas import EvalReport


def render_markdown_report(report: EvalReport) -> str:
    lines = [
        "# Gist Evaluation Report",
        "",
        "## Summary",
        "",
        f"- Examples: {report.summary.examples}",
        f"- Avg Gist reduction: {report.summary.avg_gist_reduction_percent:.2f}%",
        f"- Avg Gist timestamp hit rate: {report.summary.avg_gist_timestamp_hit_rate:.2f}",
        f"- Avg latency: {report.summary.avg_latency_ms:.2f} ms",
        "",
        "## Examples",
        "",
    ]
    for result in report.results:
        lines.extend(
            [
                f"### {result.id}",
                "",
                f"- Query: {result.query}",
                f"- Gist selected: {result.gist.metrics.selected_candidates}",
                f"- Gist reduction: {result.gist.metrics.estimated_candidate_reduction_percent:.2f}%",
                f"- Gist timestamp hit rate: {result.gist_timestamp_hit_rate:.2f}",
                f"- Latency: {result.latency_ms:.2f} ms",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"
