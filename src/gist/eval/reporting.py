from html import escape
from dataclasses import dataclass
from pathlib import Path

from gist.core.schemas import Modality
from gist.eval.schemas import EvalReport


MAX_RENDERED_EVIDENCE_CLIPS = 3
ANSWER_LIKELIHOOD_KEEP_RATIO = 0.55
PRE_CONTEXT_SECONDS = 30.0


def render_markdown_report(report: EvalReport) -> str:
    lines = [
        "# Gist Evaluation Report",
        "",
        "## Summary",
        "",
        f"- Examples: {report.summary.examples}",
        "",
        "| Variant | Avg Candidate Reduction | Avg Token Reduction | Avg Timestamp Hit Rate | Avg Answer Score | Avg Latency |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, summary in report.summary.variants.items():
        lines.append(
            f"| {name} | {summary.avg_reduction_percent:.2f}% | "
            f"{summary.avg_token_reduction_percent:.2f}% | "
            f"{summary.avg_timestamp_hit_rate:.2f} | "
            f"{_format_optional_score(summary.avg_answer_score)} | "
            f"{summary.avg_latency_ms:.2f} ms |"
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
                "#### Baselines",
                "",
                "| Baseline | Selected | Candidate Reduction | Timestamp Hit Rate | Answer Score |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for baseline in result.baselines:
            lines.append(
                f"| {baseline.name} | {baseline.selected_candidates} | "
                f"{baseline.reduction_percent:.2f}% | "
                f"{baseline.timestamp_hit_rate:.2f} | "
                f"{_format_optional_score(baseline.answer_score)} |"
            )
        lines.extend(
            [
                "",
                "#### Gist Variants",
                "",
                "| Variant | Selected | Candidate Reduction | Token Reduction | Timestamp Hit Rate | Answer Score | Latency |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for variant in result.variants:
            lines.append(
                f"| {variant.name} | {variant.response.metrics.selected_candidates} | "
                f"{variant.response.metrics.estimated_candidate_reduction_percent:.2f}% | "
                f"{variant.response.metrics.estimated_token_reduction_percent:.2f}% | "
                f"{variant.timestamp_hit_rate:.2f} | "
                f"{_format_optional_score(variant.answer_score)} | "
                f"{variant.latency_ms:.2f} ms |"
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
        f"<td>{_format_optional_score(summary.avg_answer_score)}</td>"
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
      <tr><th>Variant</th><th>Candidate Reduction</th><th>Token Reduction</th><th>Timestamp Hit Rate</th><th>Answer Score</th><th>Latency</th></tr>
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
    baseline_rows = "\n".join(
        "<tr>"
        f"<td>{escape(baseline.name)}</td>"
        f"<td>{baseline.selected_candidates}</td>"
        f"<td>{baseline.reduction_percent:.2f}%</td>"
        f"<td>{baseline.timestamp_hit_rate:.2f}</td>"
        f"<td>{_format_optional_score(baseline.answer_score)}</td>"
        "</tr>"
        for baseline in result.baselines
    )
    return f"""
    <section>
      <h3>{escape(result.id)}</h3>
      <p><strong>Query:</strong> {escape(result.query)}</p>
      <h4>Baselines</h4>
      <table>
        <thead>
          <tr><th>Baseline</th><th>Selected</th><th>Candidate Reduction</th><th>Timestamp Hit Rate</th><th>Answer Score</th></tr>
        </thead>
        <tbody>{baseline_rows}</tbody>
      </table>
      <h4>Gist Variants</h4>
      {variant_sections}
    </section>
    """


def _render_variant(variant) -> str:
    evidence_cards = _rank_evidence_cards(
        query=variant.response.query,
        cards=_merge_evidence_cards(variant.response.selected),
    )
    return (
        f"""
        <h4>{escape(variant.name)}</h4>
        <p class="muted">Selected {variant.response.metrics.selected_candidates} internal evidence items;
        rendered {len(evidence_cards)} video evidence clips;
        token reduction {variant.response.metrics.estimated_token_reduction_percent:.2f}%;
        timestamp hit rate {variant.timestamp_hit_rate:.2f};
        answer score {_format_optional_score(variant.answer_score)};
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
    relevance_score: float
    normalized_score: float
    support_label: str
    evidence_support_score: float


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
        relevance_score=max(item.relevance_score for item in group),
        normalized_score=max(item.normalized_score for item in group),
        support_label=_group_support_label(group),
        evidence_support_score=max(
            (item.evidence_support_score or 0.0 for item in group),
            default=0.0,
        ),
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


def _group_support_label(group: list) -> str:
    labels = [item.support_label for item in group if item.support_label]
    if "strong" in labels:
        return "strong"
    if "medium" in labels:
        return "medium"
    if "weak" in labels:
        return "weak"
    return "unscored"


def _rank_evidence_cards(query: str, cards: list[EvidenceCard]) -> list[EvidenceCard]:
    if len(cards) <= MAX_RENDERED_EVIDENCE_CLIPS:
        return cards

    scored = [
        (_answer_likelihood(query, card), card)
        for card in cards
    ]
    best_score = max(score for score, _card in scored)
    direct_cards = [
        card
        for score, card in scored
        if best_score > 0 and score >= best_score * ANSWER_LIKELIHOOD_KEEP_RATIO
    ]
    if not direct_cards:
        direct_cards = [max(scored, key=lambda item: item[0])[1]]

    earliest_direct = min(card.timestamp_seconds for card in direct_cards)
    pre_context = [
        card
        for score, card in scored
        if card.timestamp_seconds < earliest_direct
        and earliest_direct - card.timestamp_seconds <= PRE_CONTEXT_SECONDS
    ]
    pre_context.sort(
        key=lambda card: (
            abs(card.timestamp_seconds - earliest_direct),
            -_answer_likelihood(query, card),
        )
    )

    ranked = sorted(
        {*direct_cards, *pre_context[:1]},
        key=lambda card: (
            -_answer_likelihood(query, card),
            card.timestamp_seconds,
        ),
    )
    return sorted(ranked[:MAX_RENDERED_EVIDENCE_CLIPS], key=lambda card: card.timestamp_seconds)


def _format_optional_score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _answer_likelihood(query: str, card: EvidenceCard) -> float:
    query_text_score = _query_coverage(query, card.text)
    source_score = max(card.relevance_score, 0.0) * 0.25
    modality_bonus = 0.08 if "audio" in card.modalities else 0.0
    return query_text_score + source_score + modality_bonus


def _query_coverage(query: str, text: str) -> float:
    query_terms = _content_terms(query)
    text_terms = _content_terms(text)
    if not query_terms or not text_terms:
        return 0.0
    return len(query_terms & text_terms) / len(query_terms)


def _content_terms(value: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "at",
        "do",
        "does",
        "for",
        "in",
        "is",
        "of",
        "the",
        "they",
        "to",
        "when",
    }
    import re

    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if token not in stopwords
    }


def _render_evidence(cards: list[EvidenceCard]) -> str:
    if not cards:
        return "<p class='muted'>No evidence selected.</p>"
    return "\n".join(
        f"""
        <div class="evidence">
          <div><strong>video</strong> at <code>{card.timestamp_seconds:.2f}s</code> <span class="muted">from {escape(", ".join(card.modalities))}</span></div>
          <div class="muted">Support: {escape(card.support_label)} ({card.evidence_support_score:.3f})</div>
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
