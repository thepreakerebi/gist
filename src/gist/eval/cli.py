import argparse
from pathlib import Path

from gist.core.modes import AudioScoringMode, VisualScoringMode
from gist.core.presets import CompressionPreset
from gist.core.token_estimation import TokenEstimatorProfile
from gist.eval.benchmarks import BenchmarkName, SOTA_BENCHMARK_VARIANTS, load_benchmark_jsonl
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
        "--visual-scorer",
        choices=[scorer.value for scorer in VisualScoringMode],
        default=VisualScoringMode.BASELINE.value,
    )
    parser.add_argument(
        "--audio-scorer",
        choices=[scorer.value for scorer in AudioScoringMode],
        default=AudioScoringMode.BASELINE.value,
    )
    parser.add_argument(
        "--token-estimator",
        choices=[profile.value for profile in TokenEstimatorProfile],
        default=TokenEstimatorProfile.GENERIC.value,
    )
    parser.add_argument("--decompose-query", action="store_true")
    parser.add_argument("--adaptive-budget", action="store_true")
    parser.add_argument("--spatial-pruning", action="store_true")
    parser.add_argument("--spatial-retention-ratio", type=float, default=0.35)
    parser.add_argument("--spatial-grid-size", type=int, default=14)
    parser.add_argument(
        "--benchmark",
        choices=[benchmark.value for benchmark in BenchmarkName],
        help="Parse dataset as a benchmark JSONL format instead of native EvalExample JSONL.",
    )
    parser.add_argument(
        "--sota-sweep",
        action="store_true",
        help="Run the benchmark SOTA variant sweep instead of the default sweep.",
    )
    parser.add_argument(
        "--single-config",
        action="store_true",
        help="Run only the configured preset/options instead of the default variant sweep.",
    )
    args = parser.parse_args(argv)

    examples = (
        [
            example.to_eval_example()
            for example in load_benchmark_jsonl(args.dataset, BenchmarkName(args.benchmark))
        ]
        if args.benchmark
        else load_jsonl_dataset(args.dataset)
    )
    settings = EvalSettings(
        preset=CompressionPreset(args.preset),
        visual_scorer=VisualScoringMode(args.visual_scorer),
        audio_scorer=AudioScoringMode(args.audio_scorer),
        decompose_query=args.decompose_query,
        adaptive_budget=args.adaptive_budget,
        token_estimator=TokenEstimatorProfile(args.token_estimator),
        spatial_pruning=args.spatial_pruning,
        spatial_retention_ratio=args.spatial_retention_ratio,
        spatial_grid_size=args.spatial_grid_size,
    )
    report = EvalRunner(output_root=args.output_root).run(
        examples,
        variants=SOTA_BENCHMARK_VARIANTS if args.sota_sweep else None,
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
