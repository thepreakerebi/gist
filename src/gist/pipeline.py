from pathlib import Path
import os

from gist.audio.clap import HuggingFaceClapAudioScorer
from gist.audio.whisper import FasterWhisperTranscriber
from gist.candidates.baseline import BaselineCandidateGenerator, CandidateSet
from gist.candidates.hierarchical import shortlist_relevant_segments
from gist.core.cache import (
    DiskCache,
    candidate_cache_key,
    ingestion_cache_key,
)
from gist.core.compressor import GistCompressor
from gist.core.modes import AudioScoringMode, VisualScoringMode
from gist.core.presets import CompressionPreset
from gist.core.schemas import CompressionRequest, CompressionResponse
from gist.core.token_estimation import TokenEstimatorProfile
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
    ) -> tuple[IngestedVideo, CompressionResponse]:
        ingested, candidates = self.prepare_candidates(
            video_path=video_path,
            query=query,
            sample_count=sample_count,
            audio_window_seconds=audio_window_seconds,
            processing_mode=processing_mode,
            visual_scorer=visual_scorer,
            audio_scorer=audio_scorer,
        )

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
    ) -> tuple[IngestedVideo, CandidateSet]:
        ingestion_key = ingestion_cache_key(
            video_path=video_path,
            sample_count=sample_count,
            audio_window_seconds=audio_window_seconds,
            processing_mode=processing_mode.value,
        )
        ingested = self.cache.get_ingestion(ingestion_key)
        if ingested is None:
            ingested = self.ingestor.ingest(
                video_path=video_path,
                sample_count=sample_count,
                audio_window_seconds=audio_window_seconds,
                processing_mode=processing_mode,
            )
            self.cache.set_ingestion(ingestion_key, ingested)

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
        candidates = self.cache.get_candidates(candidates_key)
        if candidates is None:
            candidates = candidate_generator.generate(ingested, query=query)
            candidates = _maybe_shortlist_longform_segments(
                candidates=candidates,
                ingested=ingested,
                query=query,
            )
            self.cache.set_candidates(candidates_key, candidates)

        return ingested, candidates

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
) -> CandidateSet:
    if ingested.settings is None or ingested.settings.processing_mode != ProcessingMode.LONG:
        return candidates

    return shortlist_relevant_segments(
        candidates=candidates,
        query=query,
        duration_seconds=ingested.metadata.duration_seconds,
    )
