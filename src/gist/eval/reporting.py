from html import escape
from dataclasses import dataclass
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
    variant_sections = "\n".join(_render_variant(variant) for variant in result.variants)
    return f"""
    <section>
      <h3>{escape(result.id)}</h3>
      <p><strong>Query:</strong> {escape(result.query)}</p>
      {variant_sections}
    </section>
    """


def _render_variant(variant) -> str:
    evidence_cards = _merge_evidence_cards(variant.response.selected)
    return (
        f"""
        <h4>{escape(variant.name)}</h4>
        <p class="muted">Selected {variant.response.metrics.selected_candidates} internal evidence items;
        rendered {len(evidence_cards)} video evidence clips;
        token reduction {variant.response.metrics.estimated_token_reduction_percent:.2f}%;
        timestamp hit rate {variant.timestamp_hit_rate:.2f};
        latency {variant.latency_ms:.2f} ms.</p>
        {_render_evidence(evidence_cards)}
        """
    )


@dataclass(frozen=True, slots=True)
class EvidenceCard:
    timestamp_seconds: float
    modalities: tuple[str, ...]
    text: str
    reason: str
    clip_path: Path | None
    asset_path: Path | None


def _merge_evidence_cards(selected) -> list[EvidenceCard]:
    if not selected:
        return []

    groups: list[list] = []
    for item in sorted(selected, key=lambda evidence: evidence.timestamp_seconds):
        group = _matching_group(item, groups)
        if group is None:
            groups.append([item])
        else:
            group.append(item)

    return [_card_from_group(group) for group in groups]


def _matching_group(item, groups: list[list]):
    for group in groups:
        if any(_same_video_evidence(item, existing) for existing in group):
            return group
    return None


def _same_video_evidence(left, right) -> bool:
    if left.clip_path is not None and right.clip_path is not None:
        return abs(left.timestamp_seconds - right.timestamp_seconds) <= 4.5
    return False


def _card_from_group(group: list) -> EvidenceCard:
    primary = _primary_evidence(group)
    return EvidenceCard(
        timestamp_seconds=primary.timestamp_seconds,
        modalities=tuple(sorted({item.modality.value for item in group})),
        text=_combined_text(group),
        reason=_combined_reason(group),
        clip_path=primary.clip_path,
        asset_path=primary.asset_path,
    )


def _primary_evidence(group: list):
    with_clip = [item for item in group if item.clip_path is not None]
    candidates = with_clip or group
    audio_items = [item for item in candidates if item.modality == Modality.AUDIO]
    return max(
        audio_items or candidates,
        key=lambda item: (item.relevance_score, item.normalized_score),
    )


def _combined_text(group: list) -> str:
    texts: list[str] = []
    for item in sorted(group, key=lambda evidence: evidence.timestamp_seconds):
        if item.text and item.text not in texts:
            texts.append(item.text)
    return " ".join(texts)


def _combined_reason(group: list) -> str:
    if len(group) == 1:
        return group[0].reason

    modalities = ", ".join(sorted({item.modality.value for item in group}))
    timestamps = ", ".join(f"{item.timestamp_seconds:.2f}s" for item in group)
    return (
        f"Merged {len(group)} internal {modalities} evidence items into one "
        f"playable video clip because their timestamps overlap ({timestamps})."
    )


def _render_evidence(cards: list[EvidenceCard]) -> str:
    if not cards:
        return "<p class='muted'>No evidence selected.</p>"
    return "\n".join(
        f"""
        <div class="evidence">
          <div><strong>video</strong> at <code>{card.timestamp_seconds:.2f}s</code> <span class="muted">from {escape(", ".join(card.modalities))}</span></div>
          {_render_asset(card)}
          <div>{escape(card.text)}</div>
          <div class="muted">{escape(card.reason)}</div>
        </div>
        """
        for card in cards
    )


def _render_asset(card: EvidenceCard) -> str:
    if card.clip_path is not None:
        clip_markup = _render_clip(card)
        if clip_markup:
            return clip_markup

    if card.asset_path is None:
        return ""

    path = Path(card.asset_path)
    if not path.exists():
        return f"<div class='muted'>Frame asset missing: <code>{escape(str(path))}</code></div>"

    return (
        f'<img class="evidence-frame" src="{escape(path.resolve().as_uri())}" '
        f'alt="Selected visual evidence at {card.timestamp_seconds:.2f}s">'
    )


def _render_clip(card: EvidenceCard) -> str:
    path = Path(card.clip_path)
    if not path.exists():
        return f"<div class='muted'>Clip asset missing: <code>{escape(str(path))}</code></div>"

    return (
        f'<video class="evidence-clip" controls preload="metadata" '
        f'src="{escape(path.resolve().as_uri())}"></video>'
    )
