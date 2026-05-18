import argparse
from pathlib import Path

from gist.eval.benchmarks import BenchmarkName
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
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_root = args.output_root or output_dir / ".gist-eval"

    run_eval(
        [
            "--dataset",
            str(args.dataset),
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
