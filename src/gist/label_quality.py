from __future__ import annotations

import re
from html import escape
from pathlib import Path

from pydantic import BaseModel, Field

from gist.gateway.structured import ExtractionSchema, StructuredExtractionResponse


class LabelQualityWarning(BaseModel):
    code: str
    message: str
    severity: str = "warning"


class LabelQualityReport(BaseModel):
    item_count: int
    evidence_count: int
    extraction_rate: float
    duplicate_rate: float
    weak_field_rate: float
    timestamp_coverage_rate: float
    average_confidence: float
    warnings: list[LabelQualityWarning] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(warning.severity == "error" for warning in self.warnings)

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n")


def evaluate_label_quality(
    extraction: StructuredExtractionResponse,
    schema: ExtractionSchema,
    evidence_count: int,
) -> LabelQualityReport:
    item_count = len(extraction.items)
    extraction_rate = _safe_ratio(item_count, evidence_count)
    duplicate_rate = _duplicate_rate(extraction)
    weak_field_rate = _weak_field_rate(extraction, schema)
    timestamp_coverage_rate = _timestamp_coverage_rate(extraction)
    average_confidence = _average_confidence(extraction)
    warnings = _quality_warnings(
        item_count=item_count,
        evidence_count=evidence_count,
        extraction_rate=extraction_rate,
        duplicate_rate=duplicate_rate,
        weak_field_rate=weak_field_rate,
        timestamp_coverage_rate=timestamp_coverage_rate,
        average_confidence=average_confidence,
    )
    return LabelQualityReport(
        item_count=item_count,
        evidence_count=evidence_count,
        extraction_rate=extraction_rate,
        duplicate_rate=duplicate_rate,
        weak_field_rate=weak_field_rate,
        timestamp_coverage_rate=timestamp_coverage_rate,
        average_confidence=average_confidence,
        warnings=warnings,
    )


def render_label_quality_markdown(report: LabelQualityReport) -> str:
    warnings = (
        "\n".join(
            f"- `{warning.severity}` `{warning.code}`: {warning.message}"
            for warning in report.warnings
        )
        or "- none"
    )
    return f"""# Gist Label Quality Report

- Items: {report.item_count}
- Evidence count: {report.evidence_count}
- Extraction rate: {report.extraction_rate:.2%}
- Duplicate rate: {report.duplicate_rate:.2%}
- Weak-field rate: {report.weak_field_rate:.2%}
- Timestamp coverage: {report.timestamp_coverage_rate:.2%}
- Average confidence: {report.average_confidence:.2f}
- Passed: {"yes" if report.passed else "no"}

## Warnings

{warnings}
"""


def render_label_quality_html(report: LabelQualityReport) -> str:
    warnings = "\n".join(
        f"<li><strong>{escape(warning.severity)}</strong> "
        f"<code>{escape(warning.code)}</code>: {escape(warning.message)}</li>"
        for warning in report.warnings
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Gist Label Quality Report</title>
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
    code {{ background: #e9f1eb; padding: 2px 5px; border-radius: 5px; }}
  </style>
</head>
<body>
  <h1>Gist Label Quality Report</h1>
  <section class="grid">
    {_metric("Items", str(report.item_count))}
    {_metric("Extraction Rate", f"{report.extraction_rate:.1%}")}
    {_metric("Duplicate Rate", f"{report.duplicate_rate:.1%}")}
    {_metric("Weak Fields", f"{report.weak_field_rate:.1%}")}
    {_metric("Timestamp Coverage", f"{report.timestamp_coverage_rate:.1%}")}
    {_metric("Avg Confidence", f"{report.average_confidence:.2f}")}
  </section>
  <section class="card">
    <h2>Status</h2>
    <p>{"Passed" if report.passed else "Needs review"}</p>
  </section>
  <section class="card">
    <h2>Warnings</h2>
    <ul>{warnings or "<li>none</li>"}</ul>
  </section>
</body>
</html>
"""


def _quality_warnings(
    item_count: int,
    evidence_count: int,
    extraction_rate: float,
    duplicate_rate: float,
    weak_field_rate: float,
    timestamp_coverage_rate: float,
    average_confidence: float,
) -> list[LabelQualityWarning]:
    warnings: list[LabelQualityWarning] = []
    if evidence_count > 0 and item_count == 0:
        warnings.append(
            LabelQualityWarning(
                code="no_items",
                message="No structured items were extracted from selected evidence.",
                severity="error",
            )
        )
    if extraction_rate > 0.9 and evidence_count >= 3:
        warnings.append(
            LabelQualityWarning(
                code="over_extraction",
                message="Almost every evidence item became a label; review for false positives.",
            )
        )
    if duplicate_rate > 0.25:
        warnings.append(
            LabelQualityWarning(
                code="high_duplicate_rate",
                message="Many extracted items appear duplicated or highly overlapping.",
            )
        )
    if weak_field_rate > 0.35:
        warnings.append(
            LabelQualityWarning(
                code="weak_fields",
                message="Many required or important fields are missing or empty.",
            )
        )
    if item_count > 0 and timestamp_coverage_rate < 0.95:
        warnings.append(
            LabelQualityWarning(
                code="weak_timestamps",
                message="Some extracted items have weak or zero-duration timestamps.",
            )
        )
    if item_count > 0 and average_confidence < 0.45:
        warnings.append(
            LabelQualityWarning(
                code="low_confidence",
                message="Average extraction confidence is low.",
            )
        )
    return warnings


def _duplicate_rate(extraction: StructuredExtractionResponse) -> float:
    items = extraction.items
    if len(items) < 2:
        return 0.0
    duplicate_count = 0
    for index, item in enumerate(items):
        for other in items[:index]:
            if item.label != other.label:
                continue
            if _jaccard(_terms(item.support_text), _terms(other.support_text)) >= 0.72:
                duplicate_count += 1
                break
    return duplicate_count / len(items)


def _weak_field_rate(
    extraction: StructuredExtractionResponse,
    schema: ExtractionSchema,
) -> float:
    field_names = [field.name for field in schema.fields if field.required]
    if not field_names:
        field_names = [field.name for field in schema.fields]
    if not extraction.items or not field_names:
        return 0.0
    missing = 0
    total = len(extraction.items) * len(field_names)
    for item in extraction.items:
        for field_name in field_names:
            if not _has_value(item.values.get(field_name)):
                missing += 1
    return missing / total if total else 0.0


def _timestamp_coverage_rate(extraction: StructuredExtractionResponse) -> float:
    if not extraction.items:
        return 0.0
    valid = sum(
        1
        for item in extraction.items
        if item.timestamp_end_seconds > item.timestamp_start_seconds
    )
    return valid / len(extraction.items)


def _average_confidence(extraction: StructuredExtractionResponse) -> float:
    if not extraction.items:
        return 0.0
    return sum(item.confidence for item in extraction.items) / len(extraction.items)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _has_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def _metric(label: str, value: str) -> str:
    return (
        '<div class="card">'
        f'<div class="metric">{escape(value)}</div>'
        f'<div class="muted">{escape(label)}</div>'
        "</div>"
    )


def _terms(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
