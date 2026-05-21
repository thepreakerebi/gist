from html import escape
from pathlib import Path

from gist.core.schemas import CompressionResponse, SelectedCandidate
from gist.media.models import IngestedVideo


def render_local_compression_report(
    ingestion: IngestedVideo,
    compression: CompressionResponse,
) -> str:
    evidence = "\n".join(_render_evidence(item) for item in compression.selected)
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
    <p class="muted">Intent: {escape(str(compression.query_intent or "unknown"))}. {escape(compression.routing_reason or "")}</p>
  </section>

  <section class="grid">
    <div class="card"><div class="metric">{compression.metrics.selected_candidates}</div><div class="muted">selected evidence items</div></div>
    <div class="card"><div class="metric">{compression.metrics.estimated_candidate_reduction_percent:.1f}%</div><div class="muted">candidate reduction</div></div>
    <div class="card"><div class="metric">{compression.metrics.estimated_token_reduction_percent:.1f}%</div><div class="muted">estimated token reduction</div></div>
    <div class="card"><div class="metric">{ingestion.metadata.duration_seconds / 60:.1f}m</div><div class="muted">video duration</div></div>
  </section>

  <section class="card">
    <h2>Plan</h2>
    <table>
      <tbody>
        {settings_rows}
        <tr><th>Input candidates</th><td>{compression.metrics.input_candidates}</td></tr>
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


def _render_evidence(item: SelectedCandidate) -> str:
    asset = _render_asset(item)
    clip_range = ""
    if item.clip_start_seconds is not None and item.clip_end_seconds is not None:
        clip_range = f"clip {item.clip_start_seconds:.2f}s-{item.clip_end_seconds:.2f}s"
    return f"""
    <article class="card evidence">
      <h3>{escape(item.id)}</h3>
      <p><strong>{escape(item.modality.value)}</strong> at <code>{item.timestamp_seconds:.2f}s</code> <span class="muted">{escape(clip_range)}</span></p>
      {asset}
      <p>{escape(item.text)}</p>
      <p class="muted">{escape(item.reason)}</p>
      <p class="muted">segment={escape(str(item.segment_id or "n/a"))}; score={item.relevance_score:.3f}; mmr={item.mmr_score:.3f}</p>
    </article>
    """


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
