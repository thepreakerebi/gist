#!/usr/bin/env python3
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from gist.eval.frame_sweep import write_frame_sweep_summary


DEFAULT_MODEL = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a Video-MME subset sweep across different VLM frame budgets."
    )
    parser.add_argument("--work-dir", type=Path, default=Path("data/videomme-real-subset"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/videomme-frame-sweep"))
    parser.add_argument("--video-count", type=int, default=2)
    parser.add_argument("--questions-per-video", type=int, default=3)
    parser.add_argument("--duration", default="short")
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--audio-window-seconds", type=float, default=2.0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--frame-counts", default="1,4,8")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--gateway-timeout", type=float, default=900.0)
    parser.add_argument("--whisper-model-size", default="tiny")
    parser.add_argument("--whisper-device", default="cpu")
    parser.add_argument("--whisper-compute-type", default="int8")
    parser.add_argument("--frame-sampling", choices=["start", "anchor"], default="start")
    parser.add_argument("--prompt-strategy", choices=["default", "task_aware"], default="default")
    parser.add_argument("--skip-prepare", action="store_true")
    args = parser.parse_args()

    frame_counts = _parse_frame_counts(args.frame_counts)
    reports: dict[int, Path] = {}
    for frame_count in frame_counts:
        run_dir = args.output_dir / f"frames-{frame_count}"
        command = [
            sys.executable,
            "scripts/run_videomme_sota_subset.py",
            "--work-dir",
            str(args.work_dir),
            "--output-dir",
            str(run_dir),
            "--video-count",
            str(args.video_count),
            "--questions-per-video",
            str(args.questions_per_video),
            "--duration",
            args.duration,
            "--sample-count",
            str(args.sample_count),
            "--audio-window-seconds",
            str(args.audio_window_seconds),
            "--model",
            args.model,
            "--max-frames",
            str(frame_count),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--gateway-timeout",
            str(args.gateway_timeout),
            "--whisper-model-size",
            args.whisper_model_size,
            "--whisper-device",
            args.whisper_device,
            "--whisper-compute-type",
            args.whisper_compute_type,
            "--frame-sampling",
            args.frame_sampling,
            "--prompt-strategy",
            args.prompt_strategy,
        ]
        if args.skip_prepare or reports:
            command.append("--skip-prepare")
        _run(command)
        reports[frame_count] = run_dir / "sota-report.json"

    write_frame_sweep_summary(
        report_paths=reports,
        output_json=args.output_dir / "frame-sweep-summary.json",
        output_markdown=args.output_dir / "frame-sweep-summary.md",
    )
    print(f"summary={args.output_dir / 'frame-sweep-summary.md'}")
    return 0


def _parse_frame_counts(value: str) -> list[int]:
    counts: list[int] = []
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        count = int(stripped)
        if count <= 0:
            raise argparse.ArgumentTypeError("frame counts must be positive")
        counts.append(count)
    if not counts:
        raise argparse.ArgumentTypeError("at least one frame count is required")
    return sorted(set(counts))


def _run(command: list[str]) -> None:
    print("+ " + shlex.join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
