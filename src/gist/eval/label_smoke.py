import argparse
import csv
import json
from pathlib import Path

from gist import label


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a fast gist-label contract smoke test without model or video processing."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/label-smoke"))
    parser.add_argument(
        "--task",
        default="find every time prospects complain about pricing",
        help="Labeling task used to exercise preset routing.",
    )
    parser.add_argument("--video", type=Path, help="Optional fake video path to pass through the CLI.")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.video or args.output_dir / "fake-video.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.touch(exist_ok=True)

    captured_gist_args: list[str] = []

    def fake_runner(gist_args: list[str]) -> int:
        captured_gist_args[:] = gist_args
        extraction_path = Path(gist_args[gist_args.index("--extraction-output") + 1])
        csv_path = Path(gist_args[gist_args.index("--extraction-csv-output") + 1])
        extraction_path.write_text(
            json.dumps(
                {
                    "schema_name": "customer_objections",
                    "items": [
                        {
                            "label": "pricing objection",
                            "description": "The buyer says pricing is too expensive.",
                            "start_seconds": 12.0,
                            "end_seconds": 22.0,
                        }
                    ],
                },
                indent=2,
            )
        )
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["label", "description", "start_seconds", "end_seconds"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "label": "pricing objection",
                    "description": "The buyer says pricing is too expensive.",
                    "start_seconds": 12.0,
                    "end_seconds": 22.0,
                }
            )
        return 0

    exit_code = label.main(
        [
            str(video_path),
            "--task",
            args.task,
            "--output-dir",
            str(args.output_dir),
            "--no-clips",
            "--quiet",
        ],
        runner=fake_runner,
    )
    if exit_code != 0:
        return exit_code

    report_path = args.output_dir / "report.html"
    extraction_path = args.output_dir / "extraction.json"
    csv_path = args.output_dir / "extraction.csv"
    passed = _has_expected_outputs(report_path, extraction_path, csv_path, captured_gist_args)
    smoke_path = args.output_dir / "label-smoke.json"
    smoke_path.write_text(
        json.dumps(
            {
                "passed": passed,
                "task": args.task,
                "video": str(video_path),
                "gist_args": captured_gist_args,
                "report": str(report_path),
                "extraction": str(extraction_path),
                "extraction_csv": str(csv_path),
            },
            indent=2,
        )
    )

    print(f"passed={'yes' if passed else 'no'}")
    print(f"output_dir={args.output_dir}")
    print(f"smoke={smoke_path}")
    return 0 if passed else 1


def _has_expected_outputs(
    report_path: Path,
    extraction_path: Path,
    csv_path: Path,
    captured_gist_args: list[str],
) -> bool:
    if not report_path.exists() or not extraction_path.exists() or not csv_path.exists():
        return False
    report_text = report_path.read_text()
    return (
        "Gist Label Report" in report_text
        and "customer-objections" in report_text
        and "pricing objection" in extraction_path.read_text()
        and "--extraction-preset" in captured_gist_args
        and "customer-objections" in captured_gist_args
    )


if __name__ == "__main__":
    raise SystemExit(main())
