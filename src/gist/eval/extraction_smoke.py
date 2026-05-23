import argparse
import json
from pathlib import Path

from gist.eval.extraction import (
    ExtractionEvalCase,
    ExpectedExtractionItem,
    render_extraction_eval_markdown,
    run_extraction_eval_cases,
)
from gist.eval.regression import TimeRange
from gist.gateway.structured import extract_from_compression_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run structured extraction and evaluate the output in one smoke pass."
    )
    parser.add_argument("--compression", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--expected-label", required=True)
    parser.add_argument("--support-term", action="append", default=[])
    parser.add_argument("--expected-start-seconds", type=float)
    parser.add_argument("--expected-end-seconds", type=float)
    parser.add_argument("--timestamp-tolerance-seconds", type=float, default=8.0)
    parser.add_argument("--max-items", type=int)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    extraction_path = args.output_dir / "extraction.json"
    eval_json_path = args.output_dir / "extraction-eval.json"
    eval_markdown_path = args.output_dir / "extraction-eval.md"
    dataset_path = args.output_dir / "extraction-eval.dataset.jsonl"

    extraction = extract_from_compression_file(
        compression_path=args.compression,
        schema_path=args.schema,
    )
    extraction.write_json(extraction_path)

    case = ExtractionEvalCase(
        id=args.case_id,
        extraction_path=extraction_path,
        expected_items=[
            ExpectedExtractionItem(
                label=args.expected_label,
                support_terms=args.support_term,
                time_range=_expected_time_range(args),
            )
        ],
        timestamp_tolerance_seconds=args.timestamp_tolerance_seconds,
        max_items=args.max_items,
    )
    dataset_path.write_text(case.model_dump_json(exclude_none=True) + "\n")
    report = run_extraction_eval_cases([case])
    report.write_json(eval_json_path)
    eval_markdown_path.write_text(render_extraction_eval_markdown(report))

    print(f"items={len(extraction.items)}")
    print(f"passed={'yes' if report.passed else 'no'}")
    print(f"extraction={extraction_path}")
    print(f"eval_report={eval_json_path}")
    print(f"eval_markdown={eval_markdown_path}")
    return 0 if report.passed else 1


def _expected_time_range(args: argparse.Namespace) -> TimeRange | None:
    if args.expected_start_seconds is None or args.expected_end_seconds is None:
        return None
    return TimeRange(
        start_seconds=args.expected_start_seconds,
        end_seconds=args.expected_end_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
