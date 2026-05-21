from html import escape
from pathlib import Path

from gist.core.schemas import Modality
from gist.core.schemas import CompressionResponse, SelectedCandidate
from gist.core.answering import WHY_ANSWER_TERMS
from gist.core.scoring import text_similarity
from gist.media.models import IngestedVideo


MOMENT_GROUP_SECONDS = 15.0
MAX_DISPLAY_MOMENTS = 6
NEAR_DUPLICATE_TRANSCRIPT_THRESHOLD = 0.35
def render_local_compression_report(
    ingestion: IngestedVideo,
    compression: CompressionResponse,
) -> str:
    moments = _display_moments(
        _evidence_moments(compression.selected),
        query=compression.query,
        answer=compression.answer,
    )
    evidence = "\n".join(
        _render_evidence_moment(index, moment) for index, moment in enumerate(moments, start=1)
    )
    settings = ingestion.settings
    settings_rows = ""
    if settings is not None:
        settings_rows = f"""
          <tr><th>Processing mode</th><td>{escape(settings.processing_mode)}</td></tr>
          <tr><th>Plan</th><td>{escape(settings.reason)}</td></tr>
          <tr><th>Frame candidates</th><td>{settings.sample_count}</td></tr>
          <tr><th>Audio window</th><td>{settings.audio_window_seconds:.2f}s</td></tr>
          <tr><th>Audio windows</th><td>{len(ingestion.audio_windows)}</td></tr>
        """
    raw_rows = ""
    if compression.metrics.raw_input_candidates is not None:
        raw_rows = f"""
        <tr><th>Raw input candidates</th><td>{compression.metrics.raw_input_candidates}</td></tr>
        <tr><th>Fused candidate moments</th><td>{compression.metrics.fused_input_candidates or compression.metrics.input_candidates}</td></tr>
        """

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Gist Local Compression Report</title>
  <style>
    :root {{
      --ink: #17201c;
      --muted: #66746e;
      --line: #dce5df;
      --panel: #f7faf7;
      --accent: #145c43;
    }}
    body {{
      margin: 32px;
      color: var(--ink);
      font-family: Avenir Next, Gill Sans, ui-sans-serif, system-ui, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(20, 92, 67, 0.16), transparent 34rem),
        linear-gradient(180deg, #fbfcf9, #f0f5ef);
    }}
    h1, h2 {{ color: var(--accent); }}
    table {{ border-collapse: collapse; width: 100%; margin: 18px 0 28px; }}
    th, td {{ border: 1px solid var(--line); padding: 10px 12px; text-align: left; }}
    th {{ width: 220px; background: #edf4ee; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }}
    .card {{ background: rgba(255,255,255,0.82); border: 1px solid var(--line); border-radius: 14px; padding: 16px; box-shadow: 0 16px 40px rgba(20, 35, 28, 0.08); }}
    .metric {{ font-size: 28px; font-weight: 700; }}
    .muted {{ color: var(--muted); }}
    code {{ background: #e9f1eb; padding: 2px 5px; border-radius: 5px; }}
    video {{ display: block; width: min(760px, 100%); margin: 12px 0; background: #000; border-radius: 10px; }}
    img {{ display: block; max-width: min(520px, 100%); height: auto; margin: 12px 0; border-radius: 10px; border: 1px solid var(--line); }}
    .evidence {{ margin: 18px 0; }}
  </style>
</head>
<body>
  <h1>Gist Local Compression Report</h1>
  <p class="muted">{escape(str(ingestion.source_path))}</p>

  <section class="card">
    <h2>Query</h2>
    <p>{escape(compression.query)}</p>
    {_render_answer(compression.answer)}
    <p class="muted">Intent: {escape(str(compression.query_intent or "unknown"))}. {escape(compression.routing_reason or "")}</p>
  </section>

  <section class="grid">
    <div class="card"><div class="metric">{len(moments)}</div><div class="muted">video evidence moments</div></div>
    <div class="card"><div class="metric">{compression.metrics.estimated_candidate_reduction_percent:.1f}%</div><div class="muted">candidate reduction</div></div>
    <div class="card"><div class="metric">{compression.metrics.estimated_token_reduction_percent:.1f}%</div><div class="muted">estimated token reduction</div></div>
    <div class="card"><div class="metric">{ingestion.metadata.duration_seconds / 60:.1f}m</div><div class="muted">video duration</div></div>
  </section>

  <section class="card">
    <h2>Plan</h2>
    <table>
      <tbody>
        {settings_rows}
        {raw_rows}
        <tr><th>Compressor input candidates</th><td>{compression.metrics.input_candidates}</td></tr>
        <tr><th>Dropped candidates</th><td>{compression.metrics.dropped_candidates}</td></tr>
        <tr><th>Budget</th><td>{escape(compression.metrics.budget_mode)} / {escape(compression.metrics.budget_preset_used.value)}</td></tr>
      </tbody>
    </table>
  </section>

  <section>
    <h2>Evidence</h2>
    {evidence or "<p class='muted'>No evidence selected.</p>"}
  </section>
</body>
</html>
"""


def _render_answer(answer: str | None) -> str:
    if not answer:
        return ""
    return f"<p><strong>Answer:</strong> {escape(answer)}</p>"


def _evidence_moments(
    selected: list[SelectedCandidate],
    group_seconds: float = MOMENT_GROUP_SECONDS,
) -> list[list[SelectedCandidate]]:
    moments: list[list[SelectedCandidate]] = []
    for item in sorted(selected, key=lambda candidate: candidate.timestamp_seconds):
        if not moments:
            moments.append([item])
            continue
        previous = moments[-1]
        previous_center = sum(candidate.timestamp_seconds for candidate in previous) / len(previous)
        overlaps = any(_clip_ranges_overlap(item, candidate) for candidate in previous)
        if overlaps or abs(item.timestamp_seconds - previous_center) <= group_seconds:
            previous.append(item)
            continue
        moments.append([item])
    return moments


def _display_moments(
    moments: list[list[SelectedCandidate]],
    query: str,
    answer: str | None = None,
    max_moments: int = MAX_DISPLAY_MOMENTS,
) -> list[list[SelectedCandidate]]:
    transcript_backed = [moment for moment in moments if _has_transcript(moment)]
    ranked = sorted(
        transcript_backed,
        key=lambda moment: _display_quality(moment, query, answer),
        reverse=True,
    )
    deduped: list[list[SelectedCandidate]] = []
    for moment in ranked:
        transcript = _moment_transcript(moment)
        if any(
            text_similarity(transcript, _moment_transcript(existing))
            >= NEAR_DUPLICATE_TRANSCRIPT_THRESHOLD
            for existing in deduped
        ):
            continue
        deduped.append(moment)
        if len(deduped) >= max_moments:
            break
    return sorted(deduped, key=_moment_timestamp)


def _display_quality(
    moment: list[SelectedCandidate],
    query: str,
    answer: str | None,
) -> tuple[float, float, float]:
    transcript = _moment_transcript(moment).lower()
    best_relevance = max(item.relevance_score for item in moment)
    best_mmr = max(item.mmr_score for item in moment)
    answer_overlap = text_similarity(answer, transcript) if answer else 0.0
    answer_signal = 0.0
    if query.lower().strip().startswith("why"):
        answer_signal = sum(1 for term in WHY_ANSWER_TERMS if term in transcript) * 0.25
    transcript_length_bonus = min(len(transcript.split()) / 80, 1.0) * 0.1
    return (
        best_relevance + answer_signal + answer_overlap + transcript_length_bonus,
        best_relevance,
        best_mmr,
    )


def _has_transcript(moment: list[SelectedCandidate]) -> bool:
    return any(item.modality == Modality.AUDIO and item.text.strip() for item in moment)


def _clip_ranges_overlap(left: SelectedCandidate, right: SelectedCandidate) -> bool:
    if (
        left.clip_start_seconds is None
        or left.clip_end_seconds is None
        or right.clip_start_seconds is None
        or right.clip_end_seconds is None
    ):
        return False
    return left.clip_start_seconds <= right.clip_end_seconds and right.clip_start_seconds <= left.clip_end_seconds


def _render_evidence_moment(index: int, moment: list[SelectedCandidate]) -> str:
    representative = _representative_video(moment)
    asset = _render_asset(representative)
    transcript = _moment_transcript(moment)
    timestamp = _moment_timestamp(moment)
    item_ids = ", ".join(item.id for item in moment)
    score = max((item.relevance_score for item in moment), default=0.0)
    mmr = max((item.mmr_score for item in moment), default=0.0)
    segment_ids = ", ".join(
        sorted({item.segment_id for item in moment if item.segment_id is not None})
    ) or "n/a"
    clip_range = ""
    if representative.clip_start_seconds is not None and representative.clip_end_seconds is not None:
        clip_range = (
            f"clip {representative.clip_start_seconds:.2f}s-"
            f"{representative.clip_end_seconds:.2f}s"
        )
    return f"""
    <article class="card evidence">
      <h3>Video evidence {index}</h3>
      <p><strong>video</strong> around <code>{timestamp:.2f}s</code> <span class="muted">{escape(clip_range)}</span></p>
      {asset}
      <p><strong>Transcript/context:</strong> {escape(transcript)}</p>
      <p class="muted">Internal candidates grouped: {escape(item_ids)}</p>
      <p class="muted">segments={escape(segment_ids)}; best_score={score:.3f}; best_mmr={mmr:.3f}</p>
    </article>
    """


def _representative_video(moment: list[SelectedCandidate]) -> SelectedCandidate:
    audio_items = [
        item
        for item in moment
        if item.modality == Modality.AUDIO and item.clip_path is not None
    ]
    if audio_items:
        return max(audio_items, key=lambda item: (item.relevance_score, item.mmr_score))

    visual_items = [
        item
        for item in moment
        if item.modality == Modality.VISUAL and item.clip_path is not None
    ]
    if visual_items:
        return max(
            visual_items,
            key=lambda item: (item.audio_anchor_score, item.relevance_score, item.mmr_score),
        )
    with_clips = [item for item in moment if item.clip_path is not None]
    if with_clips:
        return max(with_clips, key=lambda item: (item.relevance_score, item.mmr_score))
    return max(moment, key=lambda item: (item.relevance_score, item.mmr_score))


def _moment_transcript(moment: list[SelectedCandidate]) -> str:
    snippets = []
    seen = set()
    for item in sorted(moment, key=lambda candidate: candidate.timestamp_seconds):
        if item.modality != Modality.AUDIO:
            continue
        text = item.text.strip()
        if not text or text in seen:
            continue
        snippets.append(text)
        seen.add(text)
    if snippets:
        return " ".join(snippets)
    return "Transcript unavailable for this visual-only moment."


def _moment_timestamp(moment: list[SelectedCandidate]) -> float:
    audio_items = [item for item in moment if item.modality == Modality.AUDIO]
    if audio_items:
        return min(audio_items, key=lambda item: item.timestamp_seconds).timestamp_seconds
    return min(moment, key=lambda item: item.timestamp_seconds).timestamp_seconds


def _render_asset(item: SelectedCandidate) -> str:
    if item.clip_path is not None:
        path = Path(item.clip_path)
        if path.exists():
            return f'<video controls preload="metadata" src="{escape(path.resolve().as_uri())}"></video>'
        return f"<p class='muted'>Missing clip: <code>{escape(str(path))}</code></p>"

    if item.asset_path is not None:
        path = Path(item.asset_path)
        if path.exists():
            return f'<img src="{escape(path.resolve().as_uri())}" alt="Evidence at {item.timestamp_seconds:.2f}s">'
        return f"<p class='muted'>Missing asset: <code>{escape(str(path))}</code></p>"

    return ""
