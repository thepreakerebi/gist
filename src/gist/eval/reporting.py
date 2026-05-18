from html import escape
from pathlib import Path

from gist.core.schemas import Modality
from gist.eval.schemas import EvalReport


def render_markdown_report(report: EvalReport) -> str:
    lines = [
        "# Gist Evaluation Report",
        "",
        "## Summary",
        "",
        f"- Examples: {report.summary.examples}",
        "",
        "| Variant | Avg Candidate Reduction | Avg Token Reduction | Avg Timestamp Hit Rate | Avg Latency |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, summary in report.summary.variants.items():
        lines.append(
            f"| {name} | {summary.avg_reduction_percent:.2f}% | "
            f"{summary.avg_token_reduction_percent:.2f}% | "
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
                "| Variant | Selected | Candidate Reduction | Token Reduction | Timestamp Hit Rate | Latency |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for variant in result.variants:
            lines.append(
                f"| {variant.name} | {variant.response.metrics.selected_candidates} | "
                f"{variant.response.metrics.estimated_candidate_reduction_percent:.2f}% | "
                f"{variant.response.metrics.estimated_token_reduction_percent:.2f}% | "
                f"{variant.timestamp_hit_rate:.2f} | {variant.latency_ms:.2f} ms |"
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_html_report(report: EvalReport) -> str:
    summary_rows = "\n".join(
        "<tr>"
        f"<td>{escape(name)}</td>"
        f"<td>{summary.avg_reduction_percent:.2f}%</td>"
        f"<td>{summary.avg_token_reduction_percent:.2f}%</td>"
        f"<td>{summary.avg_timestamp_hit_rate:.2f}</td>"
        f"<td>{summary.avg_latency_ms:.2f} ms</td>"
        "</tr>"
        for name, summary in report.summary.variants.items()
    )
    examples = "\n".join(_render_example(result) for result in report.results)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Gist Evaluation Report</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 32px; color: #172026; }}
    h1, h2, h3 {{ color: #0f2f2f; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{ border: 1px solid #d7dfdf; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #edf5f3; }}
    .evidence {{ background: #f8fbfa; border: 1px solid #d7dfdf; padding: 10px; margin: 8px 0; }}
    .evidence-frame {{ display: block; max-width: min(520px, 100%); height: auto; margin: 10px 0; border: 1px solid #d7dfdf; border-radius: 6px; }}
    .evidence-clip {{ display: block; width: min(720px, 100%); margin: 10px 0; border: 1px solid #d7dfdf; border-radius: 6px; background: #000; }}
    .muted {{ color: #5f6f6f; }}
    code {{ background: #eef3f2; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Gist Evaluation Report</h1>
  <p class="muted">Examples: {report.summary.examples}</p>
  <h2>Summary</h2>
  <table>
    <thead>
      <tr><th>Variant</th><th>Candidate Reduction</th><th>Token Reduction</th><th>Timestamp Hit Rate</th><th>Latency</th></tr>
    </thead>
    <tbody>{summary_rows}</tbody>
  </table>
  <h2>Examples</h2>
  {examples}
</body>
</html>
"""


def _render_example(result) -> str:
    variant_sections = "\n".join(
        f"""
        <h4>{escape(variant.name)}</h4>
        <p class="muted">Selected {variant.response.metrics.selected_candidates} evidence items;
        token reduction {variant.response.metrics.estimated_token_reduction_percent:.2f}%;
        timestamp hit rate {variant.timestamp_hit_rate:.2f};
        latency {variant.latency_ms:.2f} ms.</p>
        {_render_evidence(variant.response.selected)}
        """
        for variant in result.variants
    )
    return f"""
    <section>
      <h3>{escape(result.id)}</h3>
      <p><strong>Query:</strong> {escape(result.query)}</p>
      {variant_sections}
    </section>
    """


def _render_evidence(selected) -> str:
    if not selected:
        return "<p class='muted'>No evidence selected.</p>"
    return "\n".join(
        f"""
        <div class="evidence">
          <div><strong>{escape(item.modality.value)}</strong> at <code>{item.timestamp_seconds:.2f}s</code></div>
          {_render_asset(item)}
          <div>{escape(item.text)}</div>
          <div class="muted">{escape(item.reason)}</div>
        </div>
        """
        for item in selected
    )


def _render_asset(item) -> str:
    if item.clip_path is not None:
        clip_markup = _render_clip(item)
        if clip_markup:
            return clip_markup

    if item.modality != Modality.VISUAL or item.asset_path is None:
        return ""

    path = Path(item.asset_path)
    if not path.exists():
        return f"<div class='muted'>Frame asset missing: <code>{escape(str(path))}</code></div>"

    return (
        f'<img class="evidence-frame" src="{escape(path.resolve().as_uri())}" '
        f'alt="Selected visual evidence at {item.timestamp_seconds:.2f}s">'
    )


def _render_clip(item) -> str:
    path = Path(item.clip_path)
    if not path.exists():
        return f"<div class='muted'>Clip asset missing: <code>{escape(str(path))}</code></div>"

    return (
        f'<video class="evidence-clip" controls preload="metadata" '
        f'src="{escape(path.resolve().as_uri())}"></video>'
    )
