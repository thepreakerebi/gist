import argparse
from pathlib import Path

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
        required=True,
        help="Command that implements the Gist subprocess Video-LLM gateway protocol.",
    )
    parser.add_argument("--gateway-timeout", type=float, default=600.0)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--audio-window-seconds", type=float)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

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

    run_eval(
        [
            "--dataset",
            str(prepared_dataset),
            "--benchmark",
            args.benchmark,
            "--sota-sweep",
            "--output",
            str(output_dir / "sota-report.json"),
            "--markdown-output",
            str(output_dir / "sota-report.md"),
            "--html-output",
            str(output_dir / "sota-report.html"),
            "--output-root",
            str(output_root),
            "--gateway-command",
            args.gateway_command,
            "--gateway-timeout",
            str(args.gateway_timeout),
        ]
    )


def main() -> None:
    run()


if __name__ == "__main__":
    main()
