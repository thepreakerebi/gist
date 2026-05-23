import argparse
import json
from pathlib import Path

from gist.core.answering import answer_from_evidence, verify_answer_claims
from gist.core.modes import AudioScoringMode, VisualScoringMode
from gist.core.evidence_pruning import (
    annotate_evidence_support,
    consolidate_redundant_evidence,
    prune_evidence_to_answer,
    prune_evidence_to_answer_citations,
)
from gist.core.presets import CompressionPreset
from gist.core.progress import StepLogger
from gist.core.schemas import CompressionResponse, SelectedCandidate
from gist.gateway.evidence_package import build_evidence_package
from gist.gateway.local_text import LocalTextEvidenceGateway
from gist.gateway.ollama import DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_URL, OllamaTextGateway
from gist.gateway.schemas import GatewayRequest
from gist.media.clips import adaptive_clip_span
from gist.media.ffmpeg import FfmpegMediaProcessor
from gist.media.longform import ProcessingMode
from gist.pipeline import LocalCompressionPipeline
from gist.reports import render_local_compression_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compress a local video into query-relevant Gist evidence clips."
    )
    parser.add_argument("video_path", type=Path)
    parser.add_argument("--query", required=True)
    parser.add_argument("--output-root", type=Path, default=Path(".gist/runs"))
    parser.add_argument(
        "--preset",
        choices=list(CompressionPreset),
        default=CompressionPreset.BALANCED,
    )
    parser.add_argument(
        "--processing-mode",
        choices=list(ProcessingMode),
        default=ProcessingMode.AUTO,
    )
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--audio-window-seconds", type=float)
    parser.add_argument(
        "--visual-scorer",
        choices=list(VisualScoringMode),
        default=VisualScoringMode.BASELINE,
    )
    parser.add_argument(
        "--audio-scorer",
        choices=list(AudioScoringMode),
        default=AudioScoringMode.BASELINE,
    )
    parser.add_argument("--adaptive-budget", action="store_true")
    parser.add_argument("--decompose-query", action="store_true")
    parser.add_argument(
        "--no-visual-ocr",
        action="store_true",
        help="Disable OCR extraction from sampled frames.",
    )
    parser.add_argument("--no-clips", action="store_true")
    parser.add_argument(
        "--no-answer-prune",
        action="store_true",
        help="Keep all selected evidence instead of pruning final clips against the answer.",
    )
    parser.add_argument("--html-report", action="store_true")
    parser.add_argument("--export-evidence-package", action="store_true")
    parser.add_argument(
        "--answer-with",
        choices=["extractive", "local-text", "ollama"],
        default="extractive",
    )
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--quiet", action="store_true", help="Disable progress logging.")
    args = parser.parse_args()

    progress = StepLogger(enabled=not args.quiet)
    run_dir = args.output_root / _safe_stem(args.video_path) / _safe_stem(args.query)
    run_dir.mkdir(parents=True, exist_ok=True)

    progress(f"starting run: video={args.video_path}, query={args.query!r}")
    pipeline = LocalCompressionPipeline(output_root=args.output_root)
    ingestion, compression = pipeline.run(
        video_path=args.video_path,
        query=args.query,
        preset=CompressionPreset(args.preset),
        sample_count=args.sample_count,
        audio_window_seconds=args.audio_window_seconds,
        processing_mode=ProcessingMode(args.processing_mode),
        visual_scorer=VisualScoringMode(args.visual_scorer),
        audio_scorer=AudioScoringMode(args.audio_scorer),
        adaptive_budget=args.adaptive_budget,
        decompose_query=args.decompose_query,
        task_aware_selection=True,
        visual_ocr=not args.no_visual_ocr,
        progress=progress,
    )
    if not args.no_clips:
        progress("rendering evidence clips")
        compression = _attach_evidence_clips(
            compression=compression,
            video_path=args.video_path,
            output_dir=run_dir / "clips",
            progress=progress,
        )

    compression = _answer_compression(args, compression, progress)
    if not args.no_answer_prune:
        progress("pruning evidence against answer")
        before_prune_ids = [item.id for item in compression.selected]
        compression = prune_evidence_to_answer(compression)
        after_prune_ids = [item.id for item in compression.selected]
        if after_prune_ids != before_prune_ids:
            progress("re-answering from pruned evidence")
            compression = _answer_compression(args, compression, progress)
        progress("pruning uncited final evidence")
        before_citation_ids = [item.id for item in compression.selected]
        compression = prune_evidence_to_answer_citations(compression)
        after_citation_ids = [item.id for item in compression.selected]
        if after_citation_ids != before_citation_ids:
            progress("re-answering from cited evidence")
            compression = _answer_compression(args, compression, progress)
            progress("pruning uncited cited evidence")
            compression = prune_evidence_to_answer_citations(compression)
        before_consolidate_ids = [item.id for item in compression.selected]
        progress("consolidating redundant final evidence")
        compression = consolidate_redundant_evidence(compression)
        after_consolidate_ids = [item.id for item in compression.selected]
        if after_consolidate_ids != before_consolidate_ids:
            progress("re-answering from consolidated evidence")
            compression = _answer_compression(args, compression, progress)
            progress("pruning uncited consolidated evidence")
            compression = prune_evidence_to_answer_citations(compression)

    response_path = run_dir / "compression.json"
    progress(f"writing JSON output: {response_path}")
    response_path.write_text(
        json.dumps(
            {
                "ingestion": ingestion.model_dump(mode="json"),
                "compression": compression.model_dump(mode="json"),
            },
            indent=2,
        )
        + "\n"
    )
    html_path = None
    if args.html_report:
        html_path = run_dir / "report.html"
        progress(f"writing HTML report: {html_path}")
        html_path.write_text(render_local_compression_report(ingestion, compression))
    package_path = None
    if args.export_evidence_package:
        package_path = run_dir / "evidence_package.json"
        progress(f"writing evidence package: {package_path}")
        package_path.write_text(
            json.dumps(build_evidence_package(ingestion, compression), indent=2) + "\n"
        )

    print(f"video_id={compression.video_id}")
    if ingestion.settings is not None:
        print(f"processing_mode={ingestion.settings.processing_mode}")
        print(f"frames={ingestion.settings.sample_count}")
        print(f"audio_window_seconds={ingestion.settings.audio_window_seconds:g}")
        print(f"audio_windows={len(ingestion.audio_windows)}")
        print(f"plan={ingestion.settings.reason}")
    print(f"selected={compression.metrics.selected_candidates}")
    print(f"candidate_reduction={compression.metrics.estimated_candidate_reduction_percent:.2f}%")
    print(f"token_reduction={compression.metrics.estimated_token_reduction_percent:.2f}%")
    print(f"output={response_path}")
    if html_path is not None:
        print(f"html_report={html_path}")
    if package_path is not None:
        print(f"evidence_package={package_path}")
    return 0


def _attach_evidence_clips(
    compression: CompressionResponse,
    video_path: Path,
    output_dir: Path,
    progress: StepLogger | None = None,
) -> CompressionResponse:
    processor = FfmpegMediaProcessor()
    duration_seconds = processor.probe(video_path).duration_seconds
    _clear_previous_clips(output_dir)
    selected = []
    total = len(compression.selected)
    for index, item in enumerate(compression.selected, start=1):
        if progress is not None:
            progress(f"rendering evidence clip {index}/{total}: {item.id}")
        selected.append(
            _with_evidence_clip(
            item=item,
            compression=compression,
            video_path=video_path,
            output_dir=output_dir,
            video_duration_seconds=duration_seconds,
            processor=processor,
            )
        )
    return compression.model_copy(update={"selected": selected})


def _gateway_request(query: str, compression: CompressionResponse) -> GatewayRequest:
    return GatewayRequest(query=query, compression=compression)


def _answer_compression(
    args: argparse.Namespace,
    compression: CompressionResponse,
    progress: StepLogger,
) -> CompressionResponse:
    if args.answer_with == "extractive":
        answer = answer_from_evidence(compression)
        answered = compression.model_copy(
            update={"answer": verify_answer_claims(answer, compression)}
        )
        return annotate_evidence_support(answered)
    if args.answer_with == "local-text":
        progress("answering with local text evidence gateway")
        gateway_response = LocalTextEvidenceGateway().answer(
            _gateway_request(args.query, compression)
        )
    elif args.answer_with == "ollama":
        progress(f"answering with Ollama model: {args.ollama_model}")
        gateway_response = OllamaTextGateway(
            model=args.ollama_model,
            base_url=args.ollama_url,
        ).answer(_gateway_request(args.query, compression))
    else:
        raise ValueError(f"unsupported answer gateway: {args.answer_with}")

    answered = compression.model_copy(
        update={
            "answer": verify_answer_claims(gateway_response.answer, compression),
            "answer_provider": gateway_response.provider,
        }
    )
    return annotate_evidence_support(answered)


def _clear_previous_clips(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for path in output_dir.glob("*.mp4"):
        path.unlink(missing_ok=True)


def _with_evidence_clip(
    item: SelectedCandidate,
    compression: CompressionResponse,
    video_path: Path,
    output_dir: Path,
    video_duration_seconds: float,
    processor: FfmpegMediaProcessor,
) -> SelectedCandidate:
    span = adaptive_clip_span(
        item=item,
        query=compression.query,
        query_intent=compression.query_intent,
        video_duration_seconds=video_duration_seconds,
    )
    clip_name = f"{_safe_stem(item.id)}_{span.start_seconds:.2f}-{span.end_seconds:.2f}s.mp4"
    clip_path = output_dir / clip_name
    if not clip_path.exists():
        processor.extract_clip(
            video_path=video_path,
            output_path=clip_path,
            start_seconds=span.start_seconds,
            duration_seconds=span.duration_seconds,
        )
    return item.model_copy(
        update={
            "clip_path": clip_path,
            "clip_start_seconds": span.start_seconds,
            "clip_end_seconds": span.end_seconds,
            "reason": f"{item.reason}; {span.reason}",
        }
    )


def _safe_stem(value: str | Path) -> str:
    raw = Path(value).stem if isinstance(value, Path) else value
    normalized = "".join(char if char.isalnum() else "-" for char in raw.lower()).strip("-")
    return normalized[:80] or "gist"


if __name__ == "__main__":
    raise SystemExit(main())
