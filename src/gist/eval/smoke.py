import argparse
import subprocess
import sys
from pathlib import Path

from gist.core.modes import AudioScoringMode, VisualScoringMode
from gist.core.presets import CompressionPreset
from gist.eval.reporting import render_html_report, render_markdown_report
from gist.eval.runner import EvalRunner
from gist.eval.schemas import EvalExample, EvalSettings
from gist.gateway.subprocess import SubprocessVideoLlmGateway


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run an end-to-end Gist smoke test on a local video."
    )
    parser.add_argument("--video", type=Path, help="Video file to evaluate.")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/smoke"))
    parser.add_argument("--query", default="pricing")
    parser.add_argument("--expected-answer", default="pricing")
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--audio-window-seconds", type=float, default=1.0)
    parser.add_argument("--gateway-command")
    parser.add_argument("--gateway-timeout", type=float, default=120.0)
    parser.add_argument(
        "--visual-scorer",
        choices=[mode.value for mode in VisualScoringMode],
        default=VisualScoringMode.BASELINE.value,
    )
    parser.add_argument(
        "--audio-scorer",
        choices=[mode.value for mode in AudioScoringMode],
        default=AudioScoringMode.BASELINE.value,
    )
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.video or _generate_synthetic_video(output_dir / "synthetic-smoke.mp4")

    example = EvalExample(
        id="smoke",
        video_id=video_path.stem,
        video_path=video_path,
        query=args.query,
        expected_answer=args.expected_answer,
        duration_seconds=1.0,
        relevant_timestamps=[0.0],
        sample_count=args.sample_count,
        audio_window_seconds=args.audio_window_seconds,
    )
    gateway = SubprocessVideoLlmGateway(
        command=_gateway_command(args.gateway_command),
        timeout_seconds=args.gateway_timeout,
    )
    report = EvalRunner(
        output_root=output_dir / ".gist-eval",
        gateway=gateway,
    ).run(
        [example],
        settings=EvalSettings(
            preset=CompressionPreset.AGGRESSIVE,
            visual_scorer=VisualScoringMode(args.visual_scorer),
            audio_scorer=AudioScoringMode(args.audio_scorer),
            adaptive_budget=True,
            spatial_pruning=True,
        ),
    )

    json_path = output_dir / "smoke-report.json"
    markdown_path = output_dir / "smoke-report.md"
    html_path = output_dir / "smoke-report.html"
    report.write_json(json_path)
    markdown_path.write_text(render_markdown_report(report))
    html_path.write_text(render_html_report(report))

    variant = report.results[0].variants[0]
    print(f"video={video_path}")
    print(f"json={json_path}")
    print(f"markdown={markdown_path}")
    print(f"html={html_path}")
    print(f"answer_score={variant.answer_score}")
    print(f"selected={variant.response.metrics.selected_candidates}")
    print(
        "spatial_reduction="
        f"{variant.response.metrics.estimated_spatial_token_reduction_percent:.2f}%"
    )


def _gateway_command(command: str | None) -> list[str]:
    if command:
        import shlex

        return shlex.split(command)
    return [
        sys.executable,
        "-c",
        (
            "import json,sys;"
            "payload=json.load(sys.stdin);"
            "print(json.dumps({'answer': payload['query'], 'provider': 'smoke-fake'}))"
        ),
    ]


def _generate_synthetic_video(path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x180:rate=12:duration=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:duration=2",
            "-shortest",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return path


def main() -> None:
    run()


if __name__ == "__main__":
    main()
