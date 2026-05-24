from __future__ import annotations

import argparse
import fnmatch
import json
import shlex
from html import escape
from pathlib import Path

from pydantic import BaseModel, Field

from gist.gateway.structured import (
    EXTRACTION_PRESETS,
    SubprocessStructuredExtractor,
    extract_from_compression_file,
    load_compression_response,
    resolve_extraction_schema,
    schema_name_for_extraction_preset,
    suggest_extraction_preset,
)
from gist.label import render_label_report
from gist.label_quality import (
    LabelQualityReport,
    evaluate_label_quality,
    render_label_quality_html,
    render_label_quality_markdown,
)
from gist.reports.structured import (
    render_structured_extraction_csv,
    render_structured_extraction_html,
    render_structured_extraction_markdown,
)


class BatchLabelCaseResult(BaseModel):
    id: str
    compression_path: str
    output_dir: str
    query: str
    schema_name: str
    extraction_preset: str
    item_count: int
    evidence_count: int
    quality_passed: bool
    warning_codes: list[str] = Field(default_factory=list)
    quality: LabelQualityReport


class BatchLabelReport(BaseModel):
    task: str
    extraction_preset: str
    schema_name: str
    case_count: int
    total_items: int
    total_evidence: int
    average_items_per_case: float
    pass_rate: float
    warning_count: int
    manifest_path: str | None = None
    results: list[BatchLabelCaseResult]

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Label many existing Gist compression.json files and aggregate quality."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--compression", type=Path, action="append", default=[])
    source.add_argument("--input-root", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--preset", choices=sorted(EXTRACTION_PRESETS))
    parser.add_argument("--max-cases", type=int)
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Glob pattern for paths to keep.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Glob pattern for paths to drop.",
    )
    parser.add_argument("--query-contains", action="append", default=[])
    parser.add_argument("--min-evidence", type=int, default=0)
    parser.add_argument(
        "--write-manifest",
        type=Path,
        help="Write selected compression paths as JSONL for reproducible reruns.",
    )
    parser.add_argument(
        "--extractor-command",
        help="Optional external extractor command. Receives the structured payload on stdin.",
    )
    parser.add_argument("--extractor-timeout", type=float, default=120.0)
    args = parser.parse_args(argv)

    compression_paths = _compression_paths(args)
    if not compression_paths:
        parser.error("no compression.json files found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.write_manifest or args.output_dir / "batch-manifest.jsonl"
    write_batch_manifest(compression_paths, manifest_path)

    suggestion = suggest_extraction_preset(args.task)
    extraction_preset = args.preset or suggestion.recommended_preset
    schema_name = schema_name_for_extraction_preset(extraction_preset)
    extractor = (
        SubprocessStructuredExtractor(
            command=shlex.split(args.extractor_command),
            timeout_seconds=args.extractor_timeout,
        )
        if args.extractor_command
        else None
    )

    results = [
        run_batch_case(
            compression_path=compression_path,
            task=args.task,
            output_dir=args.output_dir / _case_id(compression_path),
            extraction_preset=extraction_preset,
            schema_name=schema_name,
            recommended_preset=suggestion.recommended_preset,
            reason=suggestion.reason,
            extractor=extractor,
        )
        for compression_path in compression_paths
    ]
    report = build_batch_report(
        task=args.task,
        extraction_preset=extraction_preset,
        schema_name=schema_name,
        results=results,
        manifest_path=str(manifest_path),
    )
    report_json = args.output_dir / "batch-report.json"
    report_markdown = args.output_dir / "batch-report.md"
    report_html = args.output_dir / "batch-report.html"
    report.write_json(report_json)
    report_markdown.write_text(render_batch_label_markdown(report))
    report_html.write_text(render_batch_label_html(report))

    print(f"cases={report.case_count}")
    print(f"items={report.total_items}")
    print(f"pass_rate={report.pass_rate:.2%}")
    print(f"warnings={report.warning_count}")
    print(f"manifest={manifest_path}")
    print(f"report={report_json}")
    print(f"markdown={report_markdown}")
    print(f"html={report_html}")
    return 0


def run_batch_case(
    compression_path: Path,
    task: str,
    output_dir: Path,
    extraction_preset: str,
    schema_name: str,
    recommended_preset: str,
    reason: str,
    extractor: SubprocessStructuredExtractor | None = None,
) -> BatchLabelCaseResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    schema = resolve_extraction_schema(preset=extraction_preset)
    compression = load_compression_response(compression_path)
    extraction = extract_from_compression_file(
        compression_path=compression_path,
        preset=extraction_preset,
        extractor=extractor,
    )
    quality = evaluate_label_quality(
        extraction=extraction,
        schema=schema,
        evidence_count=len(compression.selected),
    )

    extraction_json = output_dir / "extraction.json"
    extraction_csv = output_dir / "extraction.csv"
    extraction_markdown = output_dir / "extraction.md"
    extraction_html = output_dir / "extraction.html"
    quality_json = output_dir / "quality.json"
    quality_markdown = output_dir / "quality.md"
    quality_html = output_dir / "quality.html"
    label_report = output_dir / "report.html"
    extraction.write_json(extraction_json)
    extraction_csv.write_text(render_structured_extraction_csv(extraction))
    extraction_markdown.write_text(render_structured_extraction_markdown(extraction))
    extraction_html.write_text(render_structured_extraction_html(extraction))
    quality.write_json(quality_json)
    quality_markdown.write_text(render_label_quality_markdown(quality))
    quality_html.write_text(render_label_quality_html(quality))
    label_report.write_text(
        render_label_report(
            task=task,
            query=extraction.query,
            recommended_preset=recommended_preset,
            extraction_preset=extraction_preset,
            schema_name=schema_name,
            reason=reason,
            extraction_json=extraction_json,
            extraction_csv=extraction_csv,
            output_root=compression_path.parent,
            item_count=len(extraction.items),
        )
    )

    return BatchLabelCaseResult(
        id=_case_id(compression_path),
        compression_path=str(compression_path),
        output_dir=str(output_dir),
        query=extraction.query,
        schema_name=schema_name,
        extraction_preset=extraction_preset,
        item_count=len(extraction.items),
        evidence_count=len(compression.selected),
        quality_passed=quality.passed,
        warning_codes=[warning.code for warning in quality.warnings],
        quality=quality,
    )


def build_batch_report(
    task: str,
    extraction_preset: str,
    schema_name: str,
    results: list[BatchLabelCaseResult],
    manifest_path: str | None = None,
) -> BatchLabelReport:
    case_count = len(results)
    total_items = sum(result.item_count for result in results)
    total_evidence = sum(result.evidence_count for result in results)
    passed = sum(1 for result in results if result.quality_passed)
    warning_count = sum(len(result.warning_codes) for result in results)
    return BatchLabelReport(
        task=task,
        extraction_preset=extraction_preset,
        schema_name=schema_name,
        case_count=case_count,
        total_items=total_items,
        total_evidence=total_evidence,
        average_items_per_case=total_items / case_count if case_count else 0.0,
        pass_rate=passed / case_count if case_count else 0.0,
        warning_count=warning_count,
        manifest_path=manifest_path,
        results=results,
    )


def render_batch_label_markdown(report: BatchLabelReport) -> str:
    rows = "\n".join(
        "| "
        f"{result.id} | {result.item_count} | {result.evidence_count} | "
        f"{'yes' if result.quality_passed else 'no'} | "
        f"{', '.join(result.warning_codes) or 'none'} |"
        for result in report.results
    )
    return f"""# Gist Batch Label Report

- Task: {report.task}
- Preset: `{report.extraction_preset}`
- Schema: `{report.schema_name}`
- Cases: {report.case_count}
- Total items: {report.total_items}
- Total evidence: {report.total_evidence}
- Average items/case: {report.average_items_per_case:.2f}
- Pass rate: {report.pass_rate:.2%}
- Warnings: {report.warning_count}
- Manifest: {report.manifest_path or "none"}

| Case | Items | Evidence | Passed | Warnings |
|---|---:|---:|---|---|
{rows}
"""


def render_batch_label_html(report: BatchLabelReport) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{escape(result.id)}</td>"
        f"<td>{result.item_count}</td>"
        f"<td>{result.evidence_count}</td>"
        f"<td>{'yes' if result.quality_passed else 'no'}</td>"
        f"<td>{escape(', '.join(result.warning_codes) or 'none')}</td>"
        f'<td><a href="{escape(Path(result.output_dir).resolve().as_uri())}">artifacts</a></td>'
        "</tr>"
        for result in report.results
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Gist Batch Label Report</title>
  <style>
    body {{
      margin: 32px;
      color: #18201d;
      font-family: Avenir Next, Gill Sans, ui-sans-serif, system-ui, sans-serif;
      background: linear-gradient(180deg, #fbfcf9, #eef5ef);
    }}
    h1, h2 {{ color: #174734; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
      margin: 20px 0;
    }}
    .card {{
      background: rgba(255,255,255,0.9);
      border: 1px solid #dce5df;
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 12px 30px rgba(20, 35, 28, 0.08);
    }}
    .metric {{ font-size: 28px; font-weight: 800; }}
    .muted {{ color: #63736d; }}
    table {{ border-collapse: collapse; width: 100%; background: white; }}
    th, td {{ border: 1px solid #dce5df; padding: 8px; text-align: left; }}
    th {{ background: #e9f1eb; }}
  </style>
</head>
<body>
  <h1>Gist Batch Label Report</h1>
  <section class="grid">
    {_metric("Cases", str(report.case_count))}
    {_metric("Items", str(report.total_items))}
    {_metric("Evidence", str(report.total_evidence))}
    {_metric("Pass Rate", f"{report.pass_rate:.1%}")}
    {_metric("Warnings", str(report.warning_count))}
  </section>
  <section class="card">
    <h2>Task</h2>
    <p>{escape(report.task)}</p>
    <p class="muted">Preset: {escape(report.extraction_preset)}</p>
  </section>
  <table>
    <thead>
      <tr>
        <th>Case</th>
        <th>Items</th>
        <th>Evidence</th>
        <th>Passed</th>
        <th>Warnings</th>
        <th>Artifacts</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""


def _compression_paths(args: argparse.Namespace) -> list[Path]:
    if args.input_root is not None:
        paths = sorted(args.input_root.rglob("compression.json"))
    elif args.manifest is not None:
        paths = load_batch_manifest(args.manifest)
    else:
        paths = sorted(set(args.compression))
    paths = _filter_compression_paths(
        paths=paths,
        include_patterns=args.include,
        exclude_patterns=args.exclude,
        query_terms=args.query_contains,
        min_evidence=args.min_evidence,
    )
    if args.max_cases is not None:
        return paths[: args.max_cases]
    return paths


def load_batch_manifest(path: Path) -> list[Path]:
    text = path.read_text().strip()
    if not text:
        return []
    if text.startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError("batch manifest JSON must be a list")
        return [_manifest_item_path(item) for item in payload]
    return [_manifest_item_path(json.loads(line)) for line in text.splitlines() if line.strip()]


def write_batch_manifest(paths: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "compression": str(path),
                "case_id": _case_id(path),
            },
            sort_keys=True,
        )
        for path in paths
    ]
    output.write_text("\n".join(lines) + ("\n" if lines else ""))


def _manifest_item_path(item: object) -> Path:
    if isinstance(item, str):
        return Path(item)
    if isinstance(item, dict):
        value = item.get("compression") or item.get("compression_path") or item.get("path")
        if isinstance(value, str):
            return Path(value)
    raise ValueError("manifest item must be a path string or object with compression path")


def _filter_compression_paths(
    paths: list[Path],
    include_patterns: list[str],
    exclude_patterns: list[str],
    query_terms: list[str],
    min_evidence: int,
) -> list[Path]:
    filtered = []
    for path in paths:
        normalized_path = str(path)
        if include_patterns and not _matches_any(normalized_path, include_patterns):
            continue
        if exclude_patterns and _matches_any(normalized_path, exclude_patterns):
            continue
        compression = load_compression_response(path)
        if len(compression.selected) < min_evidence:
            continue
        normalized_query = compression.query.lower()
        if query_terms and not all(term.lower() in normalized_query for term in query_terms):
            continue
        filtered.append(path)
    return filtered


def _matches_any(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _case_id(path: Path) -> str:
    parent_parts = [part for part in path.parent.parts if part not in {".", ""}]
    parts = parent_parts[-2:] if len(parent_parts) >= 2 else [path.parent.name]
    return _slug("-".join(parts))


def _slug(value: str) -> str:
    slug = "".join(character.lower() if character.isalnum() else "-" for character in value)
    return "-".join(part for part in slug.split("-") if part) or "case"


def _metric(label: str, value: str) -> str:
    return (
        '<div class="card">'
        f'<div class="metric">{escape(value)}</div>'
        f'<div class="muted">{escape(label)}</div>'
        "</div>"
    )


if __name__ == "__main__":
    raise SystemExit(main())
