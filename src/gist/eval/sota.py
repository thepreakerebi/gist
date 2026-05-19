import argparse
from pathlib import Path

from gist.core.modes import AudioScoringMode, VisualScoringMode
from gist.core.presets import CompressionPreset
from gist.core.token_estimation import TokenEstimatorProfile
from gist.eval.benchmarks import (
    BenchmarkName,
    benchmark_readiness_issues,
    load_benchmark_jsonl,
    resolve_benchmark_video_paths,
    write_benchmark_jsonl,
)
from gist.eval.cli import run as run_eval


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run Gist SOTA benchmark variants with a local Video-LLM gateway."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, choices=[item.value for item in BenchmarkName])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--gateway-command",
        required=False,
        help="Command that implements the Gist subprocess Video-LLM gateway protocol.",
    )
    parser.add_argument(
        "--persistent-gateway-command",
        help="Long-running command that implements the JSONL Video-LLM gateway protocol.",
    )
    parser.add_argument("--gateway-timeout", type=float, default=600.0)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--audio-window-seconds", type=float)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--single-config",
        action="store_true",
        help="Run one configured Gist setting instead of the full SOTA variant sweep.",
    )
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
    parser.add_argument("--task-aware-selection", action="store_true")
    parser.add_argument("--spatial-pruning", action="store_true")
    parser.add_argument("--spatial-retention-ratio", type=float, default=0.35)
    parser.add_argument("--spatial-grid-size", type=int, default=14)
    args = parser.parse_args(argv)
    if args.gateway_command and args.persistent_gateway_command:
        raise SystemExit("Use either --gateway-command or --persistent-gateway-command, not both")
    if not args.gateway_command and not args.persistent_gateway_command and not args.dry_run:
        raise SystemExit("--gateway-command or --persistent-gateway-command is required")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_root = args.output_root or output_dir / ".gist-eval"
    benchmark = BenchmarkName(args.benchmark)
    examples = load_benchmark_jsonl(args.dataset, benchmark)
    if args.video_root is not None:
        examples = resolve_benchmark_video_paths(examples, args.video_root)
    if args.limit is not None:
        examples = examples[: args.limit]
    examples = [
        example.with_ingestion_settings(
            sample_count=args.sample_count,
            audio_window_seconds=args.audio_window_seconds,
        )
        for example in examples
    ]

    issues = benchmark_readiness_issues(examples)
    prepared_dataset = write_benchmark_jsonl(examples, output_dir / "prepared-benchmark.jsonl")
    if args.dry_run:
        print(f"examples={len(examples)}")
        print(f"prepared_dataset={prepared_dataset}")
        print(f"issues={len(issues)}")
        for issue in issues:
            print(f"- {issue}")
        return
    if issues:
        joined = "\n".join(f"- {issue}" for issue in issues)
        raise SystemExit(f"Benchmark dataset is not ready:\n{joined}")

    eval_args = [
        "--dataset",
        str(prepared_dataset),
        "--benchmark",
        args.benchmark,
        "--output",
        str(output_dir / "sota-report.json"),
        "--markdown-output",
        str(output_dir / "sota-report.md"),
        "--html-output",
        str(output_dir / "sota-report.html"),
        "--output-root",
        str(output_root),
        "--gateway-timeout",
        str(args.gateway_timeout),
    ]
    if args.persistent_gateway_command:
        eval_args.extend(["--persistent-gateway-command", args.persistent_gateway_command])
    elif args.gateway_command:
        eval_args.extend(["--gateway-command", args.gateway_command])
    if args.single_config:
        eval_args.extend(
            [
                "--single-config",
                "--preset",
                args.preset,
                "--visual-scorer",
                args.visual_scorer,
                "--audio-scorer",
                args.audio_scorer,
                "--token-estimator",
                args.token_estimator,
                "--spatial-retention-ratio",
                str(args.spatial_retention_ratio),
                "--spatial-grid-size",
                str(args.spatial_grid_size),
            ]
        )
        if args.decompose_query:
            eval_args.append("--decompose-query")
        if args.adaptive_budget:
            eval_args.append("--adaptive-budget")
        if args.task_aware_selection:
            eval_args.append("--task-aware-selection")
        if args.spatial_pruning:
            eval_args.append("--spatial-pruning")
    else:
        eval_args.append("--sota-sweep")

    run_eval(eval_args)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
