import argparse
from pathlib import Path

from gist.core.presets import CompressionPreset
from gist.core.token_estimation import TokenEstimatorProfile
from gist.eval.dataset import load_jsonl_dataset
from gist.eval.reporting import render_html_report, render_markdown_report
from gist.eval.runner import EvalRunner
from gist.eval.schemas import EvalSettings


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate Gist compression on a JSONL dataset.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=Path(".gist/eval"))
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--html-output", type=Path)
    parser.add_argument("--preset", choices=[preset.value for preset in CompressionPreset], default="balanced")
    parser.add_argument(
        "--token-estimator",
        choices=[profile.value for profile in TokenEstimatorProfile],
        default=TokenEstimatorProfile.GENERIC.value,
    )
    parser.add_argument("--decompose-query", action="store_true")
    parser.add_argument("--adaptive-budget", action="store_true")
    parser.add_argument(
        "--single-config",
        action="store_true",
        help="Run only the configured preset/options instead of the default variant sweep.",
    )
    args = parser.parse_args(argv)

    examples = load_jsonl_dataset(args.dataset)
    settings = EvalSettings(
            preset=CompressionPreset(args.preset),
            decompose_query=args.decompose_query,
            adaptive_budget=args.adaptive_budget,
            token_estimator=TokenEstimatorProfile(args.token_estimator),
        )
    report = EvalRunner(output_root=args.output_root).run(
        examples,
        settings=settings if args.single_config else None,
    )
    report.write_json(args.output)
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown_report(report))
    if args.html_output:
        args.html_output.parent.mkdir(parents=True, exist_ok=True)
        args.html_output.write_text(render_html_report(report))


def main() -> None:
    run()


if __name__ == "__main__":
    main()
