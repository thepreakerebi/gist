from pathlib import Path
import os
import inspect

from gist.audio.clap import HuggingFaceClapAudioScorer
from gist.audio.whisper import FasterWhisperTranscriber
from gist.candidates.baseline import BaselineCandidateGenerator, CandidateSet
from gist.candidates.hierarchical import shortlist_relevant_segments
from gist.candidates.moments import fuse_transcript_moments
from gist.core.answering import answer_from_evidence
from gist.core.cache import (
    DiskCache,
    candidate_cache_key,
    ingestion_cache_key,
)
from gist.core.compressor import GistCompressor
from gist.core.modes import AudioScoringMode, VisualScoringMode
from gist.core.presets import CompressionPreset
from gist.core.progress import ProgressCallback
from gist.core.schemas import CompressionRequest, CompressionResponse
from gist.core.token_estimation import TokenEstimatorProfile, estimate_tokens
from gist.media.ingestion import MediaIngestor
from gist.media.longform import ProcessingMode, plan_ingestion
from gist.media.models import IngestedVideo
from gist.vision.clip import HuggingFaceClipFrameScorer


class LocalCompressionPipeline:
    def __init__(
        self,
        output_root: Path,
        ingestor: MediaIngestor | None = None,
        candidate_generator: BaselineCandidateGenerator | None = None,
        compressor: GistCompressor | None = None,
        cache: DiskCache | None = None,
    ) -> None:
        self.output_root = output_root
        self.ingestor = ingestor or MediaIngestor(output_root=output_root)
        self._uses_default_candidate_generator = candidate_generator is None
        self.candidate_generator = candidate_generator or BaselineCandidateGenerator()
        self.compressor = compressor or GistCompressor()
        self.cache = cache or DiskCache(output_root / "cache")

    def run(
        self,
        video_path: Path,
        query: str,
        preset: CompressionPreset = CompressionPreset.BALANCED,
        sample_count: int | None = 128,
        audio_window_seconds: float | None = 1.0,
        processing_mode: ProcessingMode = ProcessingMode.SHORT,
        visual_scorer: VisualScoringMode = VisualScoringMode.BASELINE,
        audio_scorer: AudioScoringMode = AudioScoringMode.BASELINE,
        adaptive_budget: bool = False,
        decompose_query: bool = False,
        token_estimator: TokenEstimatorProfile = TokenEstimatorProfile.GENERIC,
        task_aware_selection: bool = False,
        progress: ProgressCallback | None = None,
    ) -> tuple[IngestedVideo, CompressionResponse]:
        if progress is not None:
            progress("preparing candidates")
        ingested, candidates, raw_candidate_count = self.prepare_candidates(
            video_path=video_path,
            query=query,
            sample_count=sample_count,
            audio_window_seconds=audio_window_seconds,
            processing_mode=processing_mode,
            visual_scorer=visual_scorer,
            audio_scorer=audio_scorer,
            progress=progress,
        )

        if progress is not None:
            progress("compressing candidate set")
        compression = self.compressor.compress(
            CompressionRequest(
                video_id=ingested.video_id,
                query=query,
                duration_seconds=ingested.metadata.duration_seconds,
                preset=preset,
                adaptive_budget=adaptive_budget,
                decompose_query=decompose_query,
                token_estimator=token_estimator,
                task_aware_selection=task_aware_selection,
                visual_candidates=candidates.visual,
                audio_candidates=candidates.audio,
            )
        )
        compression = _with_raw_reduction_metrics(
            compression=compression,
            raw_candidate_count=raw_candidate_count,
            raw_visual_count=len(ingested.frames),
            raw_audio_count=len(ingested.audio_windows),
        )
        compression = compression.model_copy(update={"answer": answer_from_evidence(compression)})
        if progress is not None:
            progress(f"compression complete: selected={compression.metrics.selected_candidates}")
        return ingested, compression

    def prepare_candidates(
        self,
        video_path: Path,
        query: str,
        sample_count: int | None = 128,
        audio_window_seconds: float | None = 1.0,
        processing_mode: ProcessingMode = ProcessingMode.SHORT,
        visual_scorer: VisualScoringMode = VisualScoringMode.BASELINE,
        audio_scorer: AudioScoringMode = AudioScoringMode.BASELINE,
        progress: ProgressCallback | None = None,
    ) -> tuple[IngestedVideo, CandidateSet, int]:
        ingestion_key = ingestion_cache_key(
            video_path=video_path,
            sample_count=sample_count,
            audio_window_seconds=audio_window_seconds,
            processing_mode=processing_mode.value,
        )
        ingested = self.cache.get_ingestion(ingestion_key)
        if ingested is None:
            if progress is not None:
                progress("ingestion cache miss")
            ingested = _call_with_optional_progress(
                self.ingestor.ingest,
                progress=progress,
                video_path=video_path,
                sample_count=sample_count,
                audio_window_seconds=audio_window_seconds,
                processing_mode=processing_mode,
            )
            self.cache.set_ingestion(ingestion_key, ingested)
            if progress is not None:
                progress("ingestion cached")
        elif progress is not None:
            progress("ingestion cache hit")

        audio_context_window_count = _audio_context_window_count(ingested)
        candidate_generator = self._candidate_generator_for(
            visual_scorer=visual_scorer,
            audio_scorer=audio_scorer,
            audio_context_window_count=audio_context_window_count,
        )
        candidates_key = candidate_cache_key(
            ingestion=ingested,
            query=query,
            visual_scorer=visual_scorer,
            audio_scorer=audio_scorer,
            audio_context_window_count=audio_context_window_count,
        )
        raw_candidate_count = len(ingested.frames) + len(ingested.audio_windows)
        candidates = self.cache.get_candidates(candidates_key)
        if candidates is None:
            if progress is not None:
                progress("candidate cache miss")
            candidates = _call_with_optional_progress(
                candidate_generator.generate,
                progress=progress,
                ingested_video=ingested,
                query=query,
            )
            candidates = _maybe_shortlist_longform_segments(
                candidates=candidates,
                ingested=ingested,
                query=query,
                progress=progress,
            )
            candidates = _maybe_fuse_longform_moments(
                candidates=candidates,
                ingested=ingested,
                query=query,
                progress=progress,
            )
            self.cache.set_candidates(candidates_key, candidates)
            if progress is not None:
                progress("candidates cached")
        elif progress is not None:
            progress("candidate cache hit")

        return ingested, candidates, raw_candidate_count

    def _candidate_generator_for(
        self,
        visual_scorer: VisualScoringMode,
        audio_scorer: AudioScoringMode,
        audio_context_window_count: int = 1,
    ) -> BaselineCandidateGenerator:
        visual_adapter = None
        audio_transcriber = None
        audio_score_adapter = None

        scene_aware_visuals = False
        if visual_scorer in {VisualScoringMode.CLIP, VisualScoringMode.CLIP_SCENE}:
            visual_adapter = HuggingFaceClipFrameScorer()
            scene_aware_visuals = visual_scorer == VisualScoringMode.CLIP_SCENE
        elif visual_scorer != VisualScoringMode.BASELINE:
            raise ValueError(f"unsupported visual scorer: {visual_scorer}")

        if audio_scorer == AudioScoringMode.WHISPER:
            audio_transcriber = FasterWhisperTranscriber(
                model_size=os.getenv("GIST_WHISPER_MODEL_SIZE", "base"),
                device=os.getenv("GIST_WHISPER_DEVICE", "cpu"),
                compute_type=os.getenv("GIST_WHISPER_COMPUTE_TYPE", "int8"),
            )
        elif audio_scorer == AudioScoringMode.CLAP:
            audio_score_adapter = HuggingFaceClapAudioScorer()
        elif audio_scorer != AudioScoringMode.BASELINE:
            raise ValueError(f"unsupported audio scorer: {audio_scorer}")

        if visual_adapter is None and audio_transcriber is None and audio_score_adapter is None:
            if self._uses_default_candidate_generator:
                return BaselineCandidateGenerator(
                    audio_context_window_count=audio_context_window_count
                )
            return self.candidate_generator

        return BaselineCandidateGenerator(
            visual_scorer=visual_adapter,
            audio_transcriber=audio_transcriber,
            audio_scorer=audio_score_adapter,
            audio_context_window_count=audio_context_window_count,
            scene_aware_visuals=scene_aware_visuals,
        )


def recommended_processing_mode(video_path: Path, ingestor: MediaIngestor) -> ProcessingMode:
    metadata = ingestor.processor.probe(video_path)
    return plan_ingestion(metadata.duration_seconds, mode=ProcessingMode.AUTO).mode


def _audio_context_window_count(ingested: IngestedVideo) -> int:
    if ingested.settings is None:
        return 1
    return ingested.settings.audio_context_window_count


def _maybe_shortlist_longform_segments(
    candidates: CandidateSet,
    ingested: IngestedVideo,
    query: str,
    progress: ProgressCallback | None = None,
) -> CandidateSet:
    if ingested.settings is None or ingested.settings.processing_mode != ProcessingMode.LONG:
        return candidates

    if progress is not None:
        progress("shortlisting relevant long-form segments")
    return shortlist_relevant_segments(
        candidates=candidates,
        query=query,
        duration_seconds=ingested.metadata.duration_seconds,
    )


def _maybe_fuse_longform_moments(
    candidates: CandidateSet,
    ingested: IngestedVideo,
    query: str,
    progress: ProgressCallback | None = None,
) -> CandidateSet:
    if ingested.settings is None or ingested.settings.processing_mode != ProcessingMode.LONG:
        return candidates
    if not candidates.audio:
        return candidates

    if progress is not None:
        progress("fusing transcript-centered evidence moments")
    return fuse_transcript_moments(candidates=candidates, query=query)


def _with_raw_reduction_metrics(
    compression: CompressionResponse,
    raw_candidate_count: int,
    raw_visual_count: int,
    raw_audio_count: int,
) -> CompressionResponse:
    selected_count = compression.metrics.selected_candidates
    if raw_candidate_count <= 0:
        return compression

    reduction_ratio = selected_count / raw_candidate_count
    reduction_percent = (1.0 - reduction_ratio) * 100
    raw_token_estimate = estimate_tokens(
        input_visual_candidates=raw_visual_count,
        input_audio_candidates=raw_audio_count,
        selected_modalities=[item.modality for item in compression.selected],
        profile=compression.metrics.token_estimator,
    )
    metrics = compression.metrics.model_copy(
        update={
            "raw_input_candidates": raw_candidate_count,
            "fused_input_candidates": compression.metrics.input_candidates,
            "estimated_candidate_reduction_ratio": reduction_ratio,
            "estimated_candidate_reduction_percent": reduction_percent,
            "dropped_candidates": max(raw_candidate_count - selected_count, 0),
            "estimated_baseline_tokens": raw_token_estimate.baseline_tokens,
            "estimated_raw_baseline_tokens": raw_token_estimate.baseline_tokens,
            "estimated_compressed_tokens": raw_token_estimate.compressed_tokens,
            "estimated_saved_tokens": raw_token_estimate.saved_tokens,
            "estimated_token_reduction_ratio": raw_token_estimate.reduction_ratio,
            "estimated_token_reduction_percent": raw_token_estimate.reduction_percent,
        }
    )
    return compression.model_copy(update={"metrics": metrics})


def _call_with_optional_progress(function, progress: ProgressCallback | None, **kwargs):
    if "progress" in inspect.signature(function).parameters:
        return function(**kwargs, progress=progress)
    return function(**kwargs)
