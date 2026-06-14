import argparse
import json
from html import escape
from pathlib import Path

from pydantic import BaseModel, Field

from gist.cli import main as run_gist
from gist.core.modes import AudioScoringMode, VisualScoringMode
from gist.core.schemas import CompressionResponse
from gist.eval.quality import QualityCase, QualityResult, evaluate_quality_case
from gist.eval.regression import TimeRange
from gist.media.ffmpeg import FfmpegMediaProcessor
from gist.pipeline import resolve_audio_scorer


class LongVideoSmokeReport(BaseModel):
    passed: bool
    compression_path: Path
    duration_seconds: float
    minimum_duration_seconds: float
    requested_audio_scorer: AudioScoringMode | None = None
    expected_audio_scorer: AudioScoringMode | None = None
    resolved_audio_scorer: AudioScoringMode | None = None
    routing_passed: bool
    quality: QualityResult
    failures: list[str] = Field(default_factory=list)

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n")


def evaluate_long_video_smoke(
    compression_path: Path,
    quality_case: QualityCase,
    minimum_duration_seconds: float = 3600.0,
    requested_audio_scorer: AudioScoringMode | None = None,
    expected_audio_scorer: AudioScoringMode | None = None,
) -> LongVideoSmokeReport:
    duration_seconds, compression = _load_run_artifact(compression_path)
    quality = evaluate_quality_case(quality_case)
    failures = list(quality.failures)

    if duration_seconds < minimum_duration_seconds:
        failures.append(
            f"video duration {duration_seconds:.2f}s below required "
            f"{minimum_duration_seconds:.2f}s"
        )

    resolved_audio_scorer = compression.audio_scorer_used
    routing_passed = (
        expected_audio_scorer is None or resolved_audio_scorer == expected_audio_scorer
    )
    if not routing_passed:
        actual = resolved_audio_scorer.value if resolved_audio_scorer is not None else "unknown"
        failures.append(
            f"audio scorer {actual} did not match expected {expected_audio_scorer.value}"
        )

    return LongVideoSmokeReport(
        passed=not failures,
        compression_path=compression_path,
        duration_seconds=duration_seconds,
        minimum_duration_seconds=minimum_duration_seconds,
        requested_audio_scorer=requested_audio_scorer,
        expected_audio_scorer=expected_audio_scorer,
        resolved_audio_scorer=resolved_audio_scorer,
        routing_passed=routing_passed,
        quality=quality,
        failures=failures,
    )


def render_long_video_smoke_markdown(report: LongVideoSmokeReport) -> str:
    quality = report.quality
    lines = [
        "# Gist Long-Video Smoke Report",
        "",
        f"- Status: {'pass' if report.passed else 'fail'}",
        f"- Duration: {report.duration_seconds / 60:.2f} minutes",
        f"- Audio scorer: {report.resolved_audio_scorer or 'unknown'}",
        f"- Routing gate: {'pass' if report.routing_passed else 'fail'}",
        f"- Answer term recall: {quality.answer_term_recall:.2f}",
        f"- Evidence term coverage: {quality.evidence_term_coverage:.2f}",
        f"- Evidence relevance: {quality.evidence_relevance_rate:.2f}",
        f"- Timestamp hit rate: {quality.timestamp_hit_rate:.2f}",
        f"- Grounded evidence: {quality.grounded_evidence_rate:.2f}",
        f"- Token reduction: {quality.token_reduction_percent:.2f}%",
        f"- Selected evidence: {quality.selected_evidence}",
        "",
        "## Failures",
        "",
    ]
    lines.extend(f"- {failure}" for failure in report.failures)
    if not report.failures:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def render_long_video_smoke_html(report: LongVideoSmokeReport) -> str:
    quality = report.quality
    failures = "".join(f"<li>{escape(failure)}</li>" for failure in report.failures)
    if not failures:
        failures = "<li>None</li>"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Gist Long-Video Smoke Report</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 32px; color: #172026; }}
    table {{ border-collapse: collapse; width: min(760px, 100%); }}
    th, td {{ border: 1px solid #d7dfdf; padding: 8px 10px; text-align: left; }}
    th {{ background: #edf5f3; }}
  </style>
</head>
<body>
  <h1>Gist Long-Video Smoke Report</h1>
  <p><strong>Status:</strong> {'pass' if report.passed else 'fail'}</p>
  <table>
    <tr><th>Duration</th><td>{report.duration_seconds / 60:.2f} minutes</td></tr>
    <tr><th>Audio scorer</th><td>{escape(str(report.resolved_audio_scorer or 'unknown'))}</td></tr>
    <tr><th>Routing gate</th><td>{'pass' if report.routing_passed else 'fail'}</td></tr>
    <tr><th>Answer term recall</th><td>{quality.answer_term_recall:.2f}</td></tr>
    <tr><th>Evidence term coverage</th><td>{quality.evidence_term_coverage:.2f}</td></tr>
    <tr><th>Evidence relevance</th><td>{quality.evidence_relevance_rate:.2f}</td></tr>
    <tr><th>Timestamp hit rate</th><td>{quality.timestamp_hit_rate:.2f}</td></tr>
    <tr><th>Grounded evidence</th><td>{quality.grounded_evidence_rate:.2f}</td></tr>
    <tr><th>Token reduction</th><td>{quality.token_reduction_percent:.2f}%</td></tr>
    <tr><th>Selected evidence</th><td>{quality.selected_evidence}</td></tr>
  </table>
  <h2>Failures</h2>
  <ul>{failures}</ul>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run and gate a real long-video Gist compression using answer, evidence, "
            "timestamp, grounding, routing, and token-reduction checks."
        )
    )
    parser.add_argument("--video", type=Path)
    parser.add_argument("--compression", type=Path)
    parser.add_argument("--query")
    parser.add_argument("--expected-answer-term", action="append", default=[])
    parser.add_argument("--expected-evidence-term", action="append", default=[])
    parser.add_argument(
        "--relevant-range",
        action="append",
        type=_parse_time_range,
        default=[],
        metavar="START:END",
    )
    parser.add_argument("--output-root", type=Path, default=Path(".gist/runs"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/long-video-smoke"))
    parser.add_argument("--minimum-duration-seconds", type=float, default=3600.0)
    parser.add_argument("--min-answer-term-recall", type=float, default=0.75)
    parser.add_argument("--min-evidence-term-coverage", type=float, default=0.75)
    parser.add_argument("--min-evidence-relevance-rate", type=float, default=0.8)
    parser.add_argument("--min-timestamp-hit-rate", type=float, default=0.75)
    parser.add_argument("--min-grounded-evidence-rate", type=float, default=0.8)
    parser.add_argument("--min-token-reduction-percent", type=float, default=90.0)
    parser.add_argument("--max-selected-evidence", type=int, default=6)
    parser.add_argument(
        "--visual-scorer",
        choices=list(VisualScoringMode),
        default=VisualScoringMode.CLIP_SCENE,
    )
    parser.add_argument(
        "--audio-scorer",
        choices=list(AudioScoringMode),
        default=AudioScoringMode.AUTO,
    )
    parser.add_argument(
        "--expect-audio-scorer",
        choices=[
            AudioScoringMode.BASELINE,
            AudioScoringMode.WHISPER,
            AudioScoringMode.CLAP,
        ],
    )
    parser.add_argument(
        "--answer-with",
        choices=["extractive", "local-text", "ollama"],
        default="local-text",
    )
    args = parser.parse_args(argv)

    _validate_args(parser, args)
    requested_audio_scorer = AudioScoringMode(args.audio_scorer)
    compression_path = args.compression
    expected_audio_scorer = (
        AudioScoringMode(args.expect_audio_scorer)
        if args.expect_audio_scorer is not None
        else None
    )

    if compression_path is None:
        assert args.video is not None
        assert args.query is not None
        duration_seconds = FfmpegMediaProcessor().probe(args.video).duration_seconds
        if expected_audio_scorer is None:
            expected_audio_scorer = resolve_audio_scorer(
                requested=requested_audio_scorer,
                query=args.query,
                duration_seconds=duration_seconds,
            )
        run_gist(
            [
                str(args.video),
                "--query",
                args.query,
                "--output-root",
                str(args.output_root),
                "--processing-mode",
                "auto",
                "--visual-scorer",
                str(args.visual_scorer),
                "--audio-scorer",
                requested_audio_scorer.value,
                "--adaptive-budget",
                "--decompose-query",
                "--html-report",
                "--export-evidence-package",
                "--answer-with",
                args.answer_with,
            ]
        )
        compression_path = _compression_path(args.output_root, args.video, args.query)

    quality_case = QualityCase(
        id="long-video-smoke",
        compression_path=compression_path,
        expected_answer_terms=args.expected_answer_term,
        expected_evidence_terms=args.expected_evidence_term,
        relevant_ranges=args.relevant_range,
        timestamp_tolerance_seconds=5.0,
        min_answer_term_recall=args.min_answer_term_recall,
        min_evidence_term_coverage=args.min_evidence_term_coverage,
        min_evidence_relevance_rate=args.min_evidence_relevance_rate,
        min_timestamp_hit_rate=args.min_timestamp_hit_rate,
        min_grounded_evidence_rate=args.min_grounded_evidence_rate,
        min_token_reduction_percent=args.min_token_reduction_percent,
        max_selected_evidence=args.max_selected_evidence,
    )
    report = evaluate_long_video_smoke(
        compression_path=compression_path,
        quality_case=quality_case,
        minimum_duration_seconds=args.minimum_duration_seconds,
        requested_audio_scorer=requested_audio_scorer,
        expected_audio_scorer=expected_audio_scorer,
    )
    args.report_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.report_dir / "long-video-smoke.json"
    markdown_path = args.report_dir / "long-video-smoke.md"
    html_path = args.report_dir / "long-video-smoke.html"
    report.write_json(json_path)
    markdown_path.write_text(render_long_video_smoke_markdown(report))
    html_path.write_text(render_long_video_smoke_html(report))

    print(f"passed={'yes' if report.passed else 'no'}")
    print(f"duration_minutes={report.duration_seconds / 60:.2f}")
    print(f"audio_scorer={report.resolved_audio_scorer or 'unknown'}")
    print(f"grounded={report.quality.grounded_evidence_rate:.2f}")
    print(f"token_reduction={report.quality.token_reduction_percent:.2f}%")
    print(f"compression={compression_path}")
    print(f"report={html_path}")
    for failure in report.failures:
        print(f"  - {failure}")
    return 0 if report.passed else 1


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if (args.video is None) == (args.compression is None):
        parser.error("provide exactly one of --video or --compression")
    if args.video is not None and not args.query:
        parser.error("--query is required with --video")
    if not args.expected_answer_term:
        parser.error("at least one --expected-answer-term is required")
    if not args.expected_evidence_term:
        parser.error("at least one --expected-evidence-term is required")
    if not args.relevant_range:
        parser.error("at least one --relevant-range START:END is required")


def _load_run_artifact(path: Path) -> tuple[float, CompressionResponse]:
    payload = json.loads(path.read_text())
    compression_payload = payload.get("compression", payload)
    compression = CompressionResponse.model_validate(compression_payload)
    ingestion = payload.get("ingestion")
    if ingestion is None:
        raise ValueError("long-video smoke artifacts must include ingestion metadata")
    duration_seconds = float(ingestion["metadata"]["duration_seconds"])
    return duration_seconds, compression


def _parse_time_range(value: str) -> TimeRange:
    try:
        start_raw, end_raw = value.split(":", maxsplit=1)
        start_seconds = float(start_raw)
        end_seconds = float(end_raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("range must use START:END seconds") from exc
    if end_seconds < start_seconds:
        raise argparse.ArgumentTypeError("range end must be greater than or equal to start")
    return TimeRange(start_seconds=start_seconds, end_seconds=end_seconds)


def _compression_path(output_root: Path, video_path: Path, query: str) -> Path:
    return output_root / _safe_stem(video_path) / _safe_stem(query) / "compression.json"


def _safe_stem(value: str | Path) -> str:
    raw = Path(value).stem if isinstance(value, Path) else value
    normalized = "".join(char if char.isalnum() else "-" for char in raw.lower()).strip("-")
    return normalized[:80] or "gist"


if __name__ == "__main__":
    raise SystemExit(main())
