import argparse
import json
from pathlib import Path

from gist.audio.whisper import TranscriptQuality
from gist.core.answering import answer_from_evidence, verify_answer_claims
from gist.core.evidence_pruning import (
    annotate_evidence_support,
    consolidate_redundant_evidence,
    prune_evidence_to_answer,
    prune_evidence_to_answer_citations,
    prune_weakly_grounded_evidence,
)
from gist.core.modes import AudioScoringMode, VisualScoringMode
from gist.core.presets import CompressionPreset
from gist.core.progress import StepLogger
from gist.core.quality_gate import apply_quality_gate
from gist.core.schemas import CompressionResponse, Modality, SelectedCandidate
from gist.gateway.evidence_package import build_evidence_package
from gist.gateway.local_text import LocalTextEvidenceGateway
from gist.gateway.ollama import DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_URL, OllamaTextGateway
from gist.gateway.schemas import GatewayRequest
from gist.gateway.structured import LocalStructuredExtractor, resolve_extraction_schema
from gist.media.clips import adaptive_clip_span
from gist.media.ffmpeg import FfmpegMediaProcessor
from gist.media.longform import ProcessingMode
from gist.pipeline import LocalCompressionPipeline
from gist.reports import render_local_compression_report
from gist.reports.structured import render_structured_extraction_csv
from gist.vision.spatial import (
    build_query_spatial_mask,
    estimate_spatial_tokens,
    write_spatial_mask,
    write_spatial_mask_overlay,
    write_spatial_mask_preview,
)


def main(argv: list[str] | None = None) -> int:
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
        default=AudioScoringMode.AUTO,
    )
    parser.add_argument(
        "--transcript-quality",
        choices=list(TranscriptQuality),
        default=TranscriptQuality.BALANCED,
        help=(
            "Whisper preset for transcript-backed evidence. "
            "Use fast for cheap local runs, balanced by default, accurate for better "
            "long-video summaries."
        ),
    )
    parser.add_argument("--whisper-model-size")
    parser.add_argument("--whisper-device")
    parser.add_argument("--whisper-compute-type")
    parser.add_argument("--whisper-beam-size", type=int)
    parser.add_argument(
        "--auto-transcript-retry",
        action="store_true",
        help=(
            "If transcript-backed evidence is flagged as noisy, rerun candidate "
            "generation once with a stronger transcript-quality preset."
        ),
    )
    parser.add_argument(
        "--transcript-retry-quality",
        choices=list(TranscriptQuality),
        default=TranscriptQuality.ACCURATE,
        help="Transcript quality preset used by --auto-transcript-retry.",
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
        "--spatial-pruning",
        action="store_true",
        help="Attach query-conditioned spatial masks for selected visual evidence.",
    )
    parser.add_argument("--spatial-retention-ratio", type=float, default=0.35)
    parser.add_argument("--spatial-grid-size", type=int, default=14)
    parser.add_argument(
        "--no-answer-prune",
        action="store_true",
        help="Keep all selected evidence instead of pruning final clips against the answer.",
    )
    parser.add_argument("--html-report", action="store_true")
    parser.add_argument("--export-evidence-package", action="store_true")
    parser.add_argument(
        "--extraction-schema",
        type=Path,
        help="JSON schema for timestamped structured extraction from selected evidence.",
    )
    parser.add_argument(
        "--extraction-schema-name",
        help="Built-in extraction schema name from `gist-structured-schemas`.",
    )
    parser.add_argument(
        "--extraction-preset",
        help="Extraction preset alias from `gist-structured-schemas --presets`.",
    )
    parser.add_argument(
        "--extraction-output",
        type=Path,
        help="Optional path for structured extraction JSON output.",
    )
    parser.add_argument(
        "--extraction-csv-output",
        type=Path,
        help="Optional path for structured extraction CSV output.",
    )
    parser.add_argument(
        "--answer-with",
        choices=["extractive", "local-text", "ollama"],
        default="extractive",
    )
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--quiet", action="store_true", help="Disable progress logging.")
    args = parser.parse_args(argv)

    extraction_selectors = [
        args.extraction_schema is not None,
        args.extraction_schema_name is not None,
        args.extraction_preset is not None,
    ]
    if sum(extraction_selectors) > 1:
        raise SystemExit(
            "Use only one of --extraction-schema, --extraction-schema-name, "
            "or --extraction-preset"
        )

    progress = StepLogger(enabled=not args.quiet)
    run_dir = args.output_root / _safe_stem(args.video_path) / _safe_stem(args.query)
    run_dir.mkdir(parents=True, exist_ok=True)

    progress(f"starting run: video={args.video_path}, query={args.query!r}")
    pipeline = LocalCompressionPipeline(output_root=args.output_root)
    transcript_quality = TranscriptQuality(args.transcript_quality)
    ingestion, compression = _run_pipeline_pass(
        args=args,
        pipeline=pipeline,
        transcript_quality=transcript_quality,
        progress=progress,
    )
    compression = _finalize_compression(
        args=args,
        compression=compression,
        run_dir=run_dir,
        progress=progress,
    )
    compression = _with_retry_metadata(
        compression=compression,
        auto_retry_enabled=args.auto_transcript_retry,
        retry_attempted=False,
    )
    retry_quality = TranscriptQuality(args.transcript_retry_quality)
    if _should_retry_transcripts(args, compression, transcript_quality, retry_quality):
        progress(
            "retrying with stronger transcript quality: "
            f"{transcript_quality.value} -> {retry_quality.value}"
        )
        ingestion, compression = _run_pipeline_pass(
            args=args,
            pipeline=pipeline,
            transcript_quality=retry_quality,
            progress=progress,
        )
        compression = _finalize_compression(
            args=args,
            compression=compression,
            run_dir=run_dir,
            progress=progress,
        )
        compression = _with_retry_metadata(
            compression=compression,
            auto_retry_enabled=args.auto_transcript_retry,
            retry_attempted=True,
            retry_from_quality=transcript_quality,
            retry_to_quality=retry_quality,
        )

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
    extraction_path = None
    extraction_csv_path = None
    if any(extraction_selectors):
        extraction_path = args.extraction_output or run_dir / "extraction.json"
        progress(f"writing structured extraction: {extraction_path}")
        extraction = LocalStructuredExtractor().extract(
            schema=resolve_extraction_schema(
                schema_path=args.extraction_schema,
                schema_name=args.extraction_schema_name,
                preset=args.extraction_preset,
            ),
            compression=compression,
        )
        extraction.write_json(extraction_path)
        if args.extraction_csv_output is not None:
            extraction_csv_path = args.extraction_csv_output
            progress(f"writing structured extraction CSV: {extraction_csv_path}")
            extraction_csv_path.parent.mkdir(parents=True, exist_ok=True)
            extraction_csv_path.write_text(render_structured_extraction_csv(extraction))

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
    if extraction_path is not None:
        print(f"extraction={extraction_path}")
    if extraction_csv_path is not None:
        print(f"extraction_csv={extraction_csv_path}")
    return 0


def _run_pipeline_pass(
    args: argparse.Namespace,
    pipeline: LocalCompressionPipeline,
    transcript_quality: TranscriptQuality,
    progress: StepLogger,
):
    return pipeline.run(
        video_path=args.video_path,
        query=args.query,
        preset=CompressionPreset(args.preset),
        sample_count=args.sample_count,
        audio_window_seconds=args.audio_window_seconds,
        processing_mode=ProcessingMode(args.processing_mode),
        visual_scorer=VisualScoringMode(args.visual_scorer),
        audio_scorer=AudioScoringMode(args.audio_scorer),
        transcript_quality=transcript_quality,
        whisper_model_size=args.whisper_model_size,
        whisper_device=args.whisper_device,
        whisper_compute_type=args.whisper_compute_type,
        whisper_beam_size=args.whisper_beam_size,
        adaptive_budget=args.adaptive_budget,
        decompose_query=args.decompose_query,
        task_aware_selection=True,
        visual_ocr=not args.no_visual_ocr,
        progress=progress,
    )


def _finalize_compression(
    args: argparse.Namespace,
    compression: CompressionResponse,
    run_dir: Path,
    progress: StepLogger,
) -> CompressionResponse:
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
        before_grounding_ids = [item.id for item in compression.selected]
        progress("dropping weakly grounded final evidence")
        compression = prune_weakly_grounded_evidence(compression)
        after_grounding_ids = [item.id for item in compression.selected]
        if after_grounding_ids != before_grounding_ids:
            progress("re-answering from grounded evidence")
            compression = _answer_compression(args, compression, progress)
    if args.spatial_pruning:
        progress("attaching spatial masks")
        compression = _attach_spatial_masks(
            compression=compression,
            output_dir=run_dir / "spatial",
            grid_size=args.spatial_grid_size,
            retention_ratio=args.spatial_retention_ratio,
        )

    return apply_quality_gate(compression)


def _should_retry_transcripts(
    args: argparse.Namespace,
    compression: CompressionResponse,
    current_quality: TranscriptQuality,
    retry_quality: TranscriptQuality,
) -> bool:
    if not args.auto_transcript_retry:
        return False
    if current_quality == retry_quality:
        return False
    if not _is_stronger_transcript_quality(retry_quality, current_quality):
        return False
    if any(
        override is not None
        for override in [
            args.whisper_model_size,
            args.whisper_device,
            args.whisper_compute_type,
            args.whisper_beam_size,
        ]
    ):
        return False
    if compression.audio_scorer_used != AudioScoringMode.WHISPER:
        return False
    return any(
        warning.code == "noisy_transcript_evidence"
        for warning in compression.quality_warnings
    )


def _with_retry_metadata(
    compression: CompressionResponse,
    auto_retry_enabled: bool,
    retry_attempted: bool,
    retry_from_quality: TranscriptQuality | None = None,
    retry_to_quality: TranscriptQuality | None = None,
) -> CompressionResponse:
    metadata = compression.transcript_metadata
    if metadata is None:
        return compression
    return compression.model_copy(
        update={
            "transcript_metadata": metadata.model_copy(
                update={
                    "auto_retry_enabled": auto_retry_enabled,
                    "retry_attempted": retry_attempted,
                    "retry_from_quality": (
                        retry_from_quality.value if retry_from_quality is not None else None
                    ),
                    "retry_to_quality": (
                        retry_to_quality.value if retry_to_quality is not None else None
                    ),
                }
            )
        }
    )


def _is_stronger_transcript_quality(
    candidate: TranscriptQuality,
    baseline: TranscriptQuality,
) -> bool:
    order = {
        TranscriptQuality.FAST: 0,
        TranscriptQuality.BALANCED: 1,
        TranscriptQuality.ACCURATE: 2,
    }
    return order[candidate] > order[baseline]


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


def _attach_spatial_masks(
    compression: CompressionResponse,
    output_dir: Path,
    grid_size: int,
    retention_ratio: float,
) -> CompressionResponse:
    selected: list[SelectedCandidate] = []
    visual_count = 0
    for item in compression.selected:
        if item.modality != Modality.VISUAL:
            selected.append(item)
            continue

        visual_count += 1
        mask = build_query_spatial_mask(
            evidence_id=item.id,
            query=compression.query,
            evidence_text=item.text,
            grid_size=grid_size,
            retention_ratio=retention_ratio,
        )
        mask_path = output_dir / f"{_safe_stem(item.id)}.spatial-mask.json"
        preview_path = output_dir / f"{_safe_stem(item.id)}.spatial-mask.svg"
        overlay_path = output_dir / f"{_safe_stem(item.id)}.spatial-overlay.svg"
        write_spatial_mask(mask, mask_path)
        write_spatial_mask_preview(mask, preview_path)
        overlay = None
        if item.asset_path is not None and item.asset_path.exists():
            overlay = write_spatial_mask_overlay(mask, item.asset_path, overlay_path)
        selected.append(
            item.model_copy(
                update={
                    "spatial_mask_path": mask_path,
                    "spatial_mask_preview_path": preview_path,
                    "spatial_mask_overlay_path": overlay,
                }
            )
        )

    baseline_tokens, retained_tokens, reduction_percent = estimate_spatial_tokens(
        selected_visual_count=visual_count,
        grid_size=grid_size,
        retention_ratio=retention_ratio,
    )
    metrics = compression.metrics.model_copy(
        update={
            "estimated_spatial_visual_tokens": baseline_tokens,
            "estimated_retained_spatial_visual_tokens": retained_tokens,
            "estimated_spatial_token_reduction_percent": reduction_percent,
        }
    )
    return compression.model_copy(update={"selected": selected, "metrics": metrics})


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
