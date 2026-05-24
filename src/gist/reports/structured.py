from __future__ import annotations

import csv
import json
from html import escape
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from gist.gateway.structured import StructuredExtractionResponse


def render_structured_extraction_markdown(
    extraction: StructuredExtractionResponse,
) -> str:
    lines = [
        "# Gist Structured Extraction Report",
        "",
        f"- Schema: `{extraction.schema_name}`",
        f"- Query: {extraction.query}",
        f"- Item type: `{extraction.item_type}`",
        f"- Provider: `{extraction.provider}`",
        f"- Items: {len(extraction.items)}",
        "",
        "| # | Label | Time | Confidence | Evidence | Description |",
        "|---:|---|---:|---:|---|---|",
    ]
    for index, item in enumerate(extraction.items, start=1):
        lines.append(
            f"| {index} | {item.label} | "
            f"{item.timestamp_start_seconds:.2f}s-{item.timestamp_end_seconds:.2f}s | "
            f"{item.confidence:.2f} | {item.evidence_id} | {_cell(item.description)} |"
        )
    return "\n".join(lines).strip() + "\n"


def render_structured_extraction_html(
    extraction: StructuredExtractionResponse,
) -> str:
    items = "\n".join(
        _render_item(index, item.model_dump(mode="json"))
        for index, item in enumerate(extraction.items, start=1)
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Gist Structured Extraction Report</title>
  <style>
    :root {{
      --ink: #18201d;
      --muted: #65736d;
      --line: #dce5df;
      --panel: #f7faf7;
      --accent: #164d3a;
    }}
    body {{
      margin: 32px;
      color: var(--ink);
      font-family: Avenir Next, Gill Sans, ui-sans-serif, system-ui, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(22, 77, 58, 0.14), transparent 34rem),
        linear-gradient(180deg, #fbfcf9, #f0f5ef);
    }}
    h1, h2, h3 {{ color: var(--accent); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
    }}
    .card {{
      background: rgba(255,255,255,0.86);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 16px 40px rgba(20, 35, 28, 0.08);
      margin: 16px 0;
    }}
    .metric {{ font-size: 28px; font-weight: 700; }}
    .muted {{ color: var(--muted); }}
    .label {{
      display: inline-block;
      padding: 4px 9px;
      border-radius: 999px;
      background: #e6f2ea;
      color: #145c43;
      font-weight: 700;
    }}
    code {{ background: #e9f1eb; padding: 2px 5px; border-radius: 5px; }}
    video {{
      display: block;
      width: min(760px, 100%);
      margin: 12px 0;
      background: #000;
      border-radius: 10px;
    }}
    pre {{
      white-space: pre-wrap;
      background: #edf4ee;
      padding: 12px;
      border-radius: 10px;
      overflow-x: auto;
    }}
  </style>
</head>
<body>
  <h1>Gist Structured Extraction Report</h1>
  <section class="grid">
    <div class="card">
      <div class="metric">{len(extraction.items)}</div>
      <div class="muted">items</div>
    </div>
    <div class="card">
      <div class="metric">{escape(extraction.schema_name)}</div>
      <div class="muted">schema</div>
    </div>
    <div class="card">
      <div class="metric">{escape(extraction.provider)}</div>
      <div class="muted">provider</div>
    </div>
  </section>
  <section class="card">
    <h2>Query</h2>
    <p>{escape(extraction.query)}</p>
    <p class="muted">Item type: <code>{escape(extraction.item_type)}</code></p>
  </section>
  <section>
    <h2>Extracted Items</h2>
    {items or "<p class='muted'>No items extracted.</p>"}
  </section>
</body>
</html>
"""


def render_structured_extraction_csv(
    extraction: StructuredExtractionResponse,
) -> str:
    value_keys = sorted(
        {
            key
            for item in extraction.items
            for key in item.values
        }
    )
    fieldnames = [
        "schema_name",
        "query",
        "item_type",
        "provider",
        "label",
        "description",
        "timestamp_start_seconds",
        "timestamp_end_seconds",
        "evidence_id",
        "evidence_rank",
        "confidence",
        "support_text",
        "clip_path",
        "values_json",
        *[f"value_{key}" for key in value_keys],
    ]
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for item in extraction.items:
        row = {
            "schema_name": extraction.schema_name,
            "query": extraction.query,
            "item_type": extraction.item_type,
            "provider": extraction.provider,
            "label": item.label,
            "description": item.description,
            "timestamp_start_seconds": f"{item.timestamp_start_seconds:.2f}",
            "timestamp_end_seconds": f"{item.timestamp_end_seconds:.2f}",
            "evidence_id": item.evidence_id,
            "evidence_rank": item.evidence_rank,
            "confidence": f"{item.confidence:.4f}",
            "support_text": item.support_text,
            "clip_path": item.clip_path or "",
            "values_json": json.dumps(item.values, sort_keys=True),
        }
        row.update({f"value_{key}": item.values.get(key, "") for key in value_keys})
        writer.writerow(row)
    return output.getvalue()


def _render_item(index: int, item: dict[str, Any]) -> str:
    clip = _render_clip(item.get("clip_path"))
    values = item.get("values") if isinstance(item.get("values"), dict) else {}
    return f"""
    <article class="card">
      <h3>Item {index}: <span class="label">{escape(str(item.get("label") or ""))}</span></h3>
      <p><strong>Time:</strong> <code>{_time_range(item)}</code></p>
      <p><strong>Confidence:</strong> {float(item.get("confidence") or 0):.2f}</p>
      <p><strong>Evidence:</strong> <code>{escape(str(item.get("evidence_id") or ""))}</code>
      rank {int(item.get("evidence_rank") or 0)}</p>
      {clip}
      <p><strong>Description:</strong> {escape(str(item.get("description") or ""))}</p>
      <p><strong>Support text:</strong> {escape(str(item.get("support_text") or ""))}</p>
      <pre>{escape(_pretty_json(values))}</pre>
    </article>
    """


def _render_clip(path_value: Any) -> str:
    if not path_value:
        return ""
    path = Path(str(path_value))
    if not path.exists():
        return f"<p class='muted'>Missing clip: <code>{escape(str(path))}</code></p>"
    return f'<video controls preload="metadata" src="{escape(path.resolve().as_uri())}"></video>'


def _time_range(item: dict[str, Any]) -> str:
    start = float(item.get("timestamp_start_seconds") or 0)
    end = float(item.get("timestamp_end_seconds") or 0)
    return f"{start:.2f}s-{end:.2f}s"


def _pretty_json(value: dict[str, Any]) -> str:
    return "{}" if not value else json.dumps(value, indent=2)


def _cell(value: str) -> str:
    text = value.replace("|", "\\|").replace("\n", " ").strip()
    return text[:180] + ("..." if len(text) > 180 else "")
