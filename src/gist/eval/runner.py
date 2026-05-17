import time

from gist.core.compressor import GistCompressor
from gist.core.schemas import CompressionRequest
from gist.eval.baselines import uniform_baseline
from gist.eval.metrics import timestamp_hit_rate
from gist.eval.schemas import EvalExample, EvalExampleResult, EvalReport, EvalSettings, EvalSummary


class EvalRunner:
    def __init__(self, compressor: GistCompressor | None = None) -> None:
        self.compressor = compressor or GistCompressor()

    def run(self, examples: list[EvalExample], settings: EvalSettings) -> EvalReport:
        results = [self._run_example(example, settings) for example in examples]
        return EvalReport(
            settings=settings,
            summary=_summarize(results),
            results=results,
        )

    def _run_example(self, example: EvalExample, settings: EvalSettings) -> EvalExampleResult:
        started = time.perf_counter()
        gist = self.compressor.compress(
            CompressionRequest(
                video_id=example.video_id,
                query=example.query,
                duration_seconds=example.duration_seconds,
                preset=settings.preset,
                adaptive_budget=settings.adaptive_budget,
                decompose_query=settings.decompose_query,
                visual_candidates=example.visual_candidates,
                audio_candidates=example.audio_candidates,
            )
        )
        latency_ms = (time.perf_counter() - started) * 1000

        return EvalExampleResult(
            id=example.id,
            query=example.query,
            gist=gist,
            gist_timestamp_hit_rate=timestamp_hit_rate(
                gist.selected,
                example.relevant_timestamps,
                example.timestamp_tolerance_seconds,
            ),
            baselines=[uniform_baseline(example, settings.preset)],
            latency_ms=latency_ms,
        )


def _summarize(results: list[EvalExampleResult]) -> EvalSummary:
    if not results:
        return EvalSummary(
            examples=0,
            avg_gist_reduction_percent=0.0,
            avg_gist_timestamp_hit_rate=0.0,
            avg_latency_ms=0.0,
        )

    return EvalSummary(
        examples=len(results),
        avg_gist_reduction_percent=sum(
            result.gist.metrics.estimated_candidate_reduction_percent for result in results
        )
        / len(results),
        avg_gist_timestamp_hit_rate=sum(
            result.gist_timestamp_hit_rate
            for result in results
        )
        / len(results),
        avg_latency_ms=sum(result.latency_ms for result in results) / len(results),
    )
