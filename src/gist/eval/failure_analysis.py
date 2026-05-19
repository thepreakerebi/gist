import argparse
import json
from pathlib import Path
from typing import Any


def render_failure_analysis(report_path: Path, dataset_path: Path) -> str:
    report = json.loads(report_path.read_text())
    expected_by_id = _load_expected(dataset_path)
    lines = [
        "# Gist Failure Analysis",
        "",
        f"- Report: `{report_path}`",
        f"- Dataset: `{dataset_path}`",
        f"- Examples: {len(report.get('results', []))}",
        "",
        "## Summary",
        "",
        "| Variant | Accuracy | Failures | Avg Selected | Avg Token Reduction |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in _variant_names(report):
        rows = [variant for result in report["results"] for variant in result["variants"] if variant["name"] == name]
        scores = [row.get("answer_score") or 0.0 for row in rows]
        selected = [row["response"]["metrics"]["selected_candidates"] for row in rows]
        reductions = [row["response"]["metrics"]["estimated_token_reduction_percent"] for row in rows]
        accuracy = sum(scores) / len(scores) if scores else 0.0
        lines.append(
            f"| {name} | {accuracy:.2%} | {sum(score < 1.0 for score in scores)} | "
            f"{_avg(selected):.2f} | {_avg(reductions):.2f}% |"
        )

    lines.extend(["", "## Failures", ""])
    for result in report.get("results", []):
        expected = expected_by_id.get(result["id"], {})
        failed_variants = [
            variant
            for variant in result.get("variants", [])
            if (variant.get("answer_score") or 0.0) < 1.0
        ]
        if not failed_variants:
            continue
        lines.extend(
            [
                f"### {result['id']}",
                "",
                f"- Query: {result['query']}",
                f"- Expected: {expected.get('answer', 'unknown')}",
                f"- Choices: {_format_choices(expected.get('choices', []))}",
                "",
                "#### Baselines",
                "",
                "| Baseline | Score | Answer | Selected | Top Evidence |",
                "|---|---:|---|---:|---|",
            ]
        )
        for baseline in result.get("baselines", []):
            lines.append(
                f"| {baseline['name']} | {_score(baseline)} | "
                f"{_cell(baseline.get('predicted_answer'))} | "
                f"{baseline.get('selected_candidates', 0)} | "
                f"{_cell(_top_evidence_summary(baseline.get('selected', [])))} |"
            )
        lines.extend(["", "#### Variant Failures", ""])
        for variant in failed_variants:
            metrics = variant["response"]["metrics"]
            lines.extend(
                [
                    f"##### {variant['name']}",
                    "",
                    f"- Score: {_score(variant)}",
                    f"- Answer: {variant.get('predicted_answer') or 'none'}",
                    f"- Selected: {metrics['selected_candidates']}",
                    f"- Token reduction: {metrics['estimated_token_reduction_percent']:.2f}%",
                    f"- Query intent: {variant['response'].get('query_intent')}",
                    f"- Routing reason: {variant['response'].get('routing_reason')}",
                    "",
                    "| Rank | Modality | Time | Score | Text | Reason |",
                    "|---:|---|---:|---:|---|---|",
                ]
            )
            for evidence in sorted(
                variant["response"].get("selected", []),
                key=lambda item: item.get("selection_rank") or 999,
            )[:8]:
                lines.append(
                    f"| {evidence.get('selection_rank', '')} | "
                    f"{evidence.get('modality', '')} | "
                    f"{float(evidence.get('timestamp_seconds') or 0):.2f}s | "
                    f"{float(evidence.get('normalized_score') or 0):.3f} | "
                    f"{_cell(evidence.get('text'))} | "
                    f"{_cell(evidence.get('reason'))} |"
                )
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate a failure-analysis markdown report.")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_failure_analysis(args.report, args.dataset))


def main() -> None:
    run()


def _load_expected(dataset_path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for line in dataset_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        identifier = record.get("id") or record.get("question_id")
        if isinstance(identifier, str):
            records[identifier] = record
    return records


def _variant_names(report: dict[str, Any]) -> list[str]:
    return [variant["name"] for variant in report.get("variants", [])]


def _avg(values: list[float | int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _score(row: dict[str, Any]) -> str:
    score = row.get("answer_score")
    return "n/a" if score is None else f"{float(score):.2f}"


def _cell(value: Any) -> str:
    text = str(value or "").replace("|", "\\|").replace("\n", " ").strip()
    return text[:220] + ("..." if len(text) > 220 else "")


def _format_choices(choices: Any) -> str:
    if not isinstance(choices, list):
        return ""
    return "; ".join(str(choice) for choice in choices)


def _top_evidence_summary(selected: list[dict[str, Any]]) -> str:
    if not selected:
        return ""
    evidence = sorted(selected, key=lambda item: item.get("selection_rank") or 999)[0]
    return (
        f"{evidence.get('modality')} at "
        f"{float(evidence.get('timestamp_seconds') or 0):.2f}s: {evidence.get('text')}"
    )


if __name__ == "__main__":
    main()
