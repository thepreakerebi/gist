from pathlib import Path
import time

from gist.core.compressor import GistCompressor
from gist.core.presets import CompressionPreset
from gist.core.schemas import CompressionRequest
from gist.eval.baselines import uniform_baseline
from gist.eval.metrics import timestamp_hit_rate
from gist.eval.schemas import (
    EvalExample,
    EvalExampleResult,
    EvalReport,
    EvalSettings,
    EvalSummary,
    EvalVariant,
    EvalVariantSummary,
    GistVariantResult,
)
from gist.pipeline import LocalCompressionPipeline


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
    ) -> None:
        self.compressor = compressor or GistCompressor()
        self.output_root = output_root

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
        return EvalExampleResult(
            id=example.id,
            query=example.query,
            variants=variant_results,
            baselines=[uniform_baseline(example, baseline_preset)],
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
                    visual_candidates=example.visual_candidates,
                    audio_candidates=example.audio_candidates,
                )
            )
        latency_ms = (time.perf_counter() - started) * 1000

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
        )

    return EvalSummary(examples=len(results), variants=summaries)


def _variants_from_settings(settings: EvalSettings | None) -> list[EvalVariant] | None:
    if settings is None:
        return None
    return [
        EvalVariant(
            name="gist_configured",
            preset=settings.preset,
            decompose_query=settings.decompose_query,
            adaptive_budget=settings.adaptive_budget,
        )
    ]
