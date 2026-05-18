from pathlib import Path
import re
import time

from gist.core.compressor import GistCompressor
from gist.core.presets import CompressionPreset
from gist.core.schemas import (
    CompressionMetrics,
    CompressionRequest,
    CompressionResponse,
    Modality,
    SelectedCandidate,
)
from gist.eval.answers import answer_score
from gist.eval.baselines import score_topk_baseline, uniform_baseline
from gist.eval.metrics import timestamp_hit_rate
from gist.eval.schemas import (
    EvalExample,
    EvalExampleResult,
    EvalReport,
    EvalSettings,
    EvalSummary,
    EvalVariant,
    EvalVariantSummary,
    BaselineResult,
    GistVariantResult,
)
from gist.gateway.base import LlmGateway
from gist.gateway.schemas import GatewayRequest
from gist.media.clips import adaptive_clip_span
from gist.media.ffmpeg import FfmpegMediaProcessor
from gist.pipeline import LocalCompressionPipeline
from gist.vision.spatial import (
    build_query_spatial_mask,
    estimate_spatial_tokens,
    write_spatial_mask,
)


DEFAULT_VARIANTS = [
    EvalVariant(name="gist_fixed_balanced", preset=CompressionPreset.BALANCED),
    EvalVariant(name="gist_fixed_aggressive", preset=CompressionPreset.AGGRESSIVE),
    EvalVariant(name="gist_decomposed", preset=CompressionPreset.BALANCED, decompose_query=True),
    EvalVariant(name="gist_adaptive", preset=CompressionPreset.BALANCED, adaptive_budget=True),
    EvalVariant(
        name="gist_decomposed_adaptive",
        preset=CompressionPreset.BALANCED,
        decompose_query=True,
        adaptive_budget=True,
    ),
]


class EvalRunner:
    def __init__(
        self,
        compressor: GistCompressor | None = None,
        output_root: Path = Path(".gist/eval"),
        media_processor: FfmpegMediaProcessor | None = None,
        gateway: LlmGateway | None = None,
    ) -> None:
        self.compressor = compressor or GistCompressor()
        self.output_root = output_root
        self.media_processor = media_processor or FfmpegMediaProcessor()
        self.gateway = gateway

    def run(
        self,
        examples: list[EvalExample],
        settings: EvalSettings | None = None,
        variants: list[EvalVariant] | None = None,
    ) -> EvalReport:
        resolved_variants = variants or _variants_from_settings(settings) or DEFAULT_VARIANTS
        results = [self._run_example(example, resolved_variants) for example in examples]
        return EvalReport(
            settings=settings,
            variants=resolved_variants,
            summary=_summarize(results),
            results=results,
        )

    def _run_example(
        self,
        example: EvalExample,
        variants: list[EvalVariant],
    ) -> EvalExampleResult:
        variant_results = [self._run_variant(example, variant) for variant in variants]
        baseline_preset = variants[0].preset if variants else CompressionPreset.BALANCED
        baselines = [
            uniform_baseline(example, baseline_preset),
            score_topk_baseline(example, baseline_preset),
        ]
        return EvalExampleResult(
            id=example.id,
            query=example.query,
            variants=variant_results,
            baselines=[
                self._score_baseline_answer(example, baseline, baseline_preset)
                for baseline in baselines
            ],
        )

    def _run_variant(self, example: EvalExample, variant: EvalVariant) -> GistVariantResult:
        started = time.perf_counter()
        if example.video_path is not None:
            _ingestion, gist = LocalCompressionPipeline(
                output_root=self.output_root / example.id
            ).run(
                video_path=example.video_path,
                query=example.query,
                preset=variant.preset,
                sample_count=example.sample_count,
                audio_window_seconds=example.audio_window_seconds,
                visual_scorer=variant.visual_scorer,
                audio_scorer=variant.audio_scorer,
                adaptive_budget=variant.adaptive_budget,
                decompose_query=variant.decompose_query,
                token_estimator=variant.token_estimator,
            )
            gist = self._attach_evidence_clips(
                compression=gist,
                video_path=example.video_path,
                output_dir=self.output_root / example.id / "clips" / variant.name,
            )
            if variant.spatial_pruning:
                gist = self._attach_spatial_masks(
                    compression=gist,
                    output_dir=self.output_root / example.id / "spatial" / variant.name,
                    grid_size=variant.spatial_grid_size,
                    retention_ratio=variant.spatial_retention_ratio,
                )
        else:
            gist = self.compressor.compress(
                CompressionRequest(
                    video_id=example.video_id,
                    query=example.query,
                    duration_seconds=example.duration_seconds,
                    preset=variant.preset,
                    adaptive_budget=variant.adaptive_budget,
                    decompose_query=variant.decompose_query,
                    token_estimator=variant.token_estimator,
                    visual_candidates=example.visual_candidates,
                    audio_candidates=example.audio_candidates,
                )
            )
        latency_ms = (time.perf_counter() - started) * 1000
        gateway_response = (
            self.gateway.answer(GatewayRequest(query=example.query, compression=gist))
            if self.gateway is not None
            else None
        )

        return GistVariantResult(
            name=variant.name,
            settings=variant,
            response=gist,
            timestamp_hit_rate=timestamp_hit_rate(
                gist.selected,
                example.relevant_timestamps,
                example.timestamp_tolerance_seconds,
            ),
            latency_ms=latency_ms,
            predicted_answer=gateway_response.answer if gateway_response else None,
            answer_score=answer_score(
                predicted=gateway_response.answer if gateway_response else None,
                expected=example.expected_answer,
                choices=example.choices,
            ),
            answer_provider=gateway_response.provider if gateway_response else None,
        )

    def _attach_evidence_clips(
        self,
        compression: CompressionResponse,
        video_path: Path,
        output_dir: Path,
    ) -> CompressionResponse:
        video_duration_seconds = self.media_processor.probe(video_path).duration_seconds
        selected = [
            self._with_evidence_clip(
                item=item,
                compression=compression,
                video_path=video_path,
                output_dir=output_dir,
                video_duration_seconds=video_duration_seconds,
            )
            for item in compression.selected
        ]
        return compression.model_copy(update={"selected": selected})

    def _with_evidence_clip(
        self,
        item: SelectedCandidate,
        compression: CompressionResponse,
        video_path: Path,
        output_dir: Path,
        video_duration_seconds: float,
    ) -> SelectedCandidate:
        span = adaptive_clip_span(
            item=item,
            query=compression.query,
            query_intent=compression.query_intent,
            video_duration_seconds=video_duration_seconds,
        )
        clip_path = (
            output_dir
            / f"{_safe_file_stem(item.id)}_{span.start_seconds:.2f}-{span.end_seconds:.2f}s.mp4"
        )
        if not clip_path.exists():
            self.media_processor.extract_clip(
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
            }
        )

    def _attach_spatial_masks(
        self,
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
                grid_size=grid_size,
                retention_ratio=retention_ratio,
            )
            mask_path = output_dir / f"{_safe_file_stem(item.id)}.spatial-mask.json"
            write_spatial_mask(mask, mask_path)
            selected.append(item.model_copy(update={"spatial_mask_path": mask_path}))

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

    def _score_baseline_answer(
        self,
        example: EvalExample,
        baseline: BaselineResult,
        preset: CompressionPreset,
    ) -> BaselineResult:
        if self.gateway is None:
            return baseline

        compression = _baseline_compression_response(
            example=example,
            baseline=baseline,
            preset=preset,
        )
        gateway_response = self.gateway.answer(
            GatewayRequest(query=example.query, compression=compression)
        )
        return baseline.model_copy(
            update={
                "predicted_answer": gateway_response.answer,
                "answer_score": answer_score(
                    predicted=gateway_response.answer,
                    expected=example.expected_answer,
                    choices=example.choices,
                ),
                "answer_provider": gateway_response.provider,
            }
        )


def _summarize(results: list[EvalExampleResult]) -> EvalSummary:
    if not results:
        return EvalSummary(examples=0, variants={})

    variant_names = [variant.name for variant in results[0].variants]
    summaries: dict[str, EvalVariantSummary] = {}
    for variant_name in variant_names:
        variant_results = [
            variant
            for result in results
            for variant in result.variants
            if variant.name == variant_name
        ]
        summaries[variant_name] = EvalVariantSummary(
            avg_reduction_percent=sum(
                result.response.metrics.estimated_candidate_reduction_percent
                for result in variant_results
            )
            / len(variant_results),
            avg_token_reduction_percent=sum(
                result.response.metrics.estimated_token_reduction_percent
                for result in variant_results
            )
            / len(variant_results),
            avg_timestamp_hit_rate=sum(
                result.timestamp_hit_rate for result in variant_results
            )
            / len(variant_results),
            avg_latency_ms=sum(result.latency_ms for result in variant_results)
            / len(variant_results),
            avg_answer_score=_average_answer_score(variant_results),
        )

    return EvalSummary(examples=len(results), variants=summaries)


def _baseline_compression_response(
    example: EvalExample,
    baseline: BaselineResult,
    preset: CompressionPreset,
) -> CompressionResponse:
    input_count = len(example.visual_candidates) + len(example.audio_candidates)
    selected_count = len(baseline.selected)
    reduction_ratio = 1.0 if input_count == 0 else selected_count / input_count
    return CompressionResponse(
        video_id=example.video_id,
        query=example.query,
        preset=preset,
        selected=baseline.selected,
        metrics=CompressionMetrics(
            input_candidates=input_count,
            selected_candidates=selected_count,
            visual_selected=sum(item.modality == Modality.VISUAL for item in baseline.selected),
            audio_selected=sum(item.modality == Modality.AUDIO for item in baseline.selected),
            estimated_candidate_reduction_ratio=reduction_ratio,
            estimated_candidate_reduction_percent=baseline.reduction_percent,
            dropped_candidates=max(input_count - selected_count, 0),
            budget_preset_used=preset,
        ),
    )


def _average_answer_score(variant_results: list[GistVariantResult]) -> float | None:
    scores = [result.answer_score for result in variant_results if result.answer_score is not None]
    if not scores:
        return None
    return sum(scores) / len(scores)


def _variants_from_settings(settings: EvalSettings | None) -> list[EvalVariant] | None:
    if settings is None:
        return None
    return [
        EvalVariant(
            name="gist_configured",
            preset=settings.preset,
            visual_scorer=settings.visual_scorer,
            audio_scorer=settings.audio_scorer,
            decompose_query=settings.decompose_query,
            adaptive_budget=settings.adaptive_budget,
            token_estimator=settings.token_estimator,
            spatial_pruning=settings.spatial_pruning,
            spatial_retention_ratio=settings.spatial_retention_ratio,
            spatial_grid_size=settings.spatial_grid_size,
        )
    ]


def _safe_file_stem(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "evidence"
