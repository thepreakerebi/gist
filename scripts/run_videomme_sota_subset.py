#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_MODEL = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare and run a real Video-MME subset through the Gist SOTA sweep."
    )
    parser.add_argument("--work-dir", type=Path, default=Path("data/videomme-real-subset"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/videomme-real-sota"))
    parser.add_argument("--video-count", type=int, default=2)
    parser.add_argument("--questions-per-video", type=int, default=3)
    parser.add_argument("--duration", default="short")
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--audio-window-seconds", type=float, default=2.0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-frames", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--gateway-timeout", type=float, default=900.0)
    parser.add_argument("--whisper-model-size", default="tiny")
    parser.add_argument("--whisper-device", default="cpu")
    parser.add_argument("--whisper-compute-type", default="int8")
    parser.add_argument("--skip-prepare", action="store_true")
    args = parser.parse_args()

    if not args.skip_prepare:
        _run(
            [
                sys.executable,
                "scripts/prepare_videomme_subset.py",
                "--output-dir",
                str(args.work_dir),
                "--duration",
                args.duration,
                "--video-count",
                str(args.video_count),
                "--questions-per-video",
                str(args.questions_per_video),
            ]
        )

    dataset = args.work_dir / "videomme-subset.jsonl"
    if not dataset.exists():
        raise SystemExit(f"dataset not found: {dataset}")

    gateway_command = (
        f"{sys.executable} scripts/run_hf_vlm_gateway_server.py "
        f"--model {args.model} "
        f"--max-frames {args.max_frames} "
        f"--max-new-tokens {args.max_new_tokens}"
    )
    env = os.environ.copy()
    env.update(
        {
            "GIST_WHISPER_MODEL_SIZE": args.whisper_model_size,
            "GIST_WHISPER_DEVICE": args.whisper_device,
            "GIST_WHISPER_COMPUTE_TYPE": args.whisper_compute_type,
        }
    )
    _run(
        [
            sys.executable,
            "-m",
            "gist.eval.sota",
            "--dataset",
            str(dataset),
            "--benchmark",
            "video_mme",
            "--output-dir",
            str(args.output_dir),
            "--limit",
            str(args.video_count * args.questions_per_video),
            "--sample-count",
            str(args.sample_count),
            "--audio-window-seconds",
            str(args.audio_window_seconds),
            "--persistent-gateway-command",
            gateway_command,
            "--gateway-timeout",
            str(args.gateway_timeout),
        ],
        env=env,
    )
    print(f"report={args.output_dir / 'sota-report.html'}")
    return 0


def _run(command: list[str], env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
