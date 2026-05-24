import argparse
import json
from html import escape
from pathlib import Path

from gist.cli import main as gist_main
from gist.gateway.structured import schema_name_for_extraction_preset, suggest_extraction_preset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Label a video with a suggested Gist extraction preset."
    )
    parser.add_argument("video_path", type=Path)
    parser.add_argument("--task", required=True, help="Natural-language labeling task.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--query", help="Override the compression query. Defaults to task.")
    parser.add_argument("--preset", help="Override the suggested extraction preset.")
    parser.add_argument("--processing-mode", default="auto")
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--audio-window-seconds", type=float)
    parser.add_argument("--visual-scorer", default="clip_scene")
    parser.add_argument("--audio-scorer", default="baseline")
    parser.add_argument(
        "--answer-with",
        choices=["extractive", "local-text", "ollama"],
        default="extractive",
    )
    parser.add_argument("--ollama-model")
    parser.add_argument("--ollama-url")
    parser.add_argument("--no-clips", action="store_true")
    parser.add_argument("--no-answer-prune", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    suggestion = suggest_extraction_preset(args.task)
    extraction_preset = args.preset or suggestion.recommended_preset
    schema_name = schema_name_for_extraction_preset(extraction_preset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extraction_json = args.output_dir / "extraction.json"
    extraction_csv = args.output_dir / "extraction.csv"
    report_html = args.output_dir / "report.html"
    output_root = args.output_dir / "runs"

    gist_args = [
        str(args.video_path),
        "--query",
        args.query or args.task,
        "--output-root",
        str(output_root),
        "--processing-mode",
        args.processing_mode,
        "--visual-scorer",
        args.visual_scorer,
        "--audio-scorer",
        args.audio_scorer,
        "--adaptive-budget",
        "--decompose-query",
        "--extraction-preset",
        extraction_preset,
        "--extraction-output",
        str(extraction_json),
        "--extraction-csv-output",
        str(extraction_csv),
    ]
    if args.sample_count is not None:
        gist_args.extend(["--sample-count", str(args.sample_count)])
    if args.audio_window_seconds is not None:
        gist_args.extend(["--audio-window-seconds", str(args.audio_window_seconds)])
    if args.answer_with:
        gist_args.extend(["--answer-with", args.answer_with])
    if args.ollama_model:
        gist_args.extend(["--ollama-model", args.ollama_model])
    if args.ollama_url:
        gist_args.extend(["--ollama-url", args.ollama_url])
    if args.no_clips:
        gist_args.append("--no-clips")
    if args.no_answer_prune:
        gist_args.append("--no-answer-prune")
    if args.quiet:
        gist_args.append("--quiet")

    exit_code = gist_main(gist_args)
    item_count = _extraction_item_count(extraction_json)
    report_html.write_text(
        render_label_report(
            task=args.task,
            query=args.query or args.task,
            recommended_preset=suggestion.recommended_preset,
            extraction_preset=extraction_preset,
            schema_name=schema_name,
            reason=suggestion.reason,
            extraction_json=extraction_json,
            extraction_csv=extraction_csv,
            output_root=output_root,
            item_count=item_count,
        )
    )
    print(f"recommended_preset={suggestion.recommended_preset}")
    print(f"extraction_preset={extraction_preset}")
    print(f"schema_name={schema_name}")
    print(f"reason={suggestion.reason}")
    print(f"extraction={extraction_json}")
    print(f"extraction_csv={extraction_csv}")
    print(f"report={report_html}")
    return exit_code


def render_label_report(
    task: str,
    query: str,
    recommended_preset: str,
    extraction_preset: str,
    schema_name: str,
    reason: str,
    extraction_json: Path,
    extraction_csv: Path,
    output_root: Path,
    item_count: int | None,
) -> str:
    item_label = "unknown" if item_count is None else str(item_count)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Gist Label Report</title>
  <style>
    body {{
      margin: 32px;
      color: #18201d;
      font-family: Avenir Next, Gill Sans, ui-sans-serif, system-ui, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(39, 89, 67, 0.16), transparent 34rem),
        linear-gradient(180deg, #fbfcf9, #eef5ef);
    }}
    h1, h2 {{ color: #174734; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin: 20px 0;
    }}
    .card {{
      background: rgba(255,255,255,0.88);
      border: 1px solid #dce5df;
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 16px 40px rgba(20, 35, 28, 0.08);
    }}
    .metric {{ font-size: 30px; font-weight: 800; }}
    .muted {{ color: #63736d; }}
    a {{ color: #165c42; font-weight: 700; }}
    code {{ background: #e9f1eb; padding: 2px 5px; border-radius: 5px; }}
  </style>
</head>
<body>
  <h1>Gist Label Report</h1>
  <section class="grid">
    <div class="card">
      <div class="metric">{escape(item_label)}</div>
      <div class="muted">extracted items</div>
    </div>
    <div class="card">
      <div class="metric">{escape(extraction_preset)}</div>
      <div class="muted">active preset</div>
    </div>
    <div class="card">
      <div class="metric">{escape(schema_name)}</div>
      <div class="muted">schema</div>
    </div>
  </section>
  <section class="card">
    <h2>Task</h2>
    <p>{escape(task)}</p>
    <p class="muted">Compression query: <code>{escape(query)}</code></p>
  </section>
  <section class="card">
    <h2>Preset Selection</h2>
    <p><strong>Recommended:</strong> {escape(recommended_preset)}</p>
    <p><strong>Used:</strong> {escape(extraction_preset)}</p>
    <p>{escape(reason)}</p>
  </section>
  <section class="card">
    <h2>Artifacts</h2>
    <p><a href="{escape(extraction_json.resolve().as_uri())}">Extraction JSON</a></p>
    <p><a href="{escape(extraction_csv.resolve().as_uri())}">Extraction CSV</a></p>
    <p><a href="{escape(output_root.resolve().as_uri())}">Run artifacts directory</a></p>
  </section>
</body>
</html>
"""


def _extraction_item_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    items = payload.get("items")
    return len(items) if isinstance(items, list) else None


if __name__ == "__main__":
    raise SystemExit(main())
