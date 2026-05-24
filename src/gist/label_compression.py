import argparse
import shlex
from pathlib import Path

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
    evaluate_label_quality,
    render_label_quality_html,
    render_label_quality_markdown,
)
from gist.reports.structured import (
    render_structured_extraction_csv,
    render_structured_extraction_html,
    render_structured_extraction_markdown,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Label an existing Gist compression.json without rerunning video processing."
    )
    parser.add_argument("--compression", required=True, type=Path)
    parser.add_argument("--task", required=True, help="Natural-language labeling task.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--preset", choices=sorted(EXTRACTION_PRESETS))
    parser.add_argument(
        "--extractor-command",
        help="Optional external extractor command. Receives the structured payload on stdin.",
    )
    parser.add_argument("--extractor-timeout", type=float, default=120.0)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suggestion = suggest_extraction_preset(args.task)
    extraction_preset = args.preset or suggestion.recommended_preset
    schema_name = schema_name_for_extraction_preset(extraction_preset)
    schema = resolve_extraction_schema(preset=extraction_preset)
    compression = load_compression_response(args.compression)
    extractor = (
        SubprocessStructuredExtractor(
            command=shlex.split(args.extractor_command),
            timeout_seconds=args.extractor_timeout,
        )
        if args.extractor_command
        else None
    )

    extraction = extract_from_compression_file(
        compression_path=args.compression,
        preset=extraction_preset,
        extractor=extractor,
    )

    extraction_json = args.output_dir / "extraction.json"
    extraction_csv = args.output_dir / "extraction.csv"
    extraction_markdown = args.output_dir / "extraction.md"
    extraction_html = args.output_dir / "extraction.html"
    quality_json = args.output_dir / "quality.json"
    quality_markdown = args.output_dir / "quality.md"
    quality_html = args.output_dir / "quality.html"
    report_html = args.output_dir / "report.html"
    quality = evaluate_label_quality(
        extraction=extraction,
        schema=schema,
        evidence_count=len(compression.selected),
    )
    extraction.write_json(extraction_json)
    extraction_csv.write_text(render_structured_extraction_csv(extraction))
    extraction_markdown.write_text(render_structured_extraction_markdown(extraction))
    extraction_html.write_text(render_structured_extraction_html(extraction))
    quality.write_json(quality_json)
    quality_markdown.write_text(render_label_quality_markdown(quality))
    quality_html.write_text(render_label_quality_html(quality))
    report_html.write_text(
        render_label_report(
            task=args.task,
            query=extraction.query,
            recommended_preset=suggestion.recommended_preset,
            extraction_preset=extraction_preset,
            schema_name=schema_name,
            reason=suggestion.reason,
            extraction_json=extraction_json,
            extraction_csv=extraction_csv,
            output_root=args.compression.parent,
            item_count=len(extraction.items),
        )
    )

    print(f"recommended_preset={suggestion.recommended_preset}")
    print(f"extraction_preset={extraction_preset}")
    print(f"schema_name={schema_name}")
    print(f"items={len(extraction.items)}")
    print(f"provider={extraction.provider}")
    print(f"quality_passed={'yes' if quality.passed else 'no'}")
    print(f"quality_warnings={len(quality.warnings)}")
    print(f"extraction={extraction_json}")
    print(f"extraction_csv={extraction_csv}")
    print(f"extraction_markdown={extraction_markdown}")
    print(f"extraction_html={extraction_html}")
    print(f"quality={quality_json}")
    print(f"quality_markdown={quality_markdown}")
    print(f"quality_html={quality_html}")
    print(f"report={report_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
