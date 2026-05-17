from pathlib import Path

from gist.audio.clap import HuggingFaceClapAudioScorer
from gist.audio.whisper import FasterWhisperTranscriber
from gist.candidates.baseline import BaselineCandidateGenerator
from gist.core.cache import (
    DiskCache,
    candidate_cache_key,
    ingestion_cache_key,
)
from gist.core.compressor import GistCompressor
from gist.core.modes import AudioScoringMode, VisualScoringMode
from gist.core.presets import CompressionPreset
from gist.core.schemas import CompressionRequest, CompressionResponse
from gist.media.ingestion import MediaIngestor
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
        self.candidate_generator = candidate_generator or BaselineCandidateGenerator()
        self.compressor = compressor or GistCompressor()
        self.cache = cache or DiskCache(output_root / "cache")

    def run(
        self,
        video_path: Path,
        query: str,
        preset: CompressionPreset = CompressionPreset.BALANCED,
        sample_count: int = 128,
        audio_window_seconds: float = 1.0,
        visual_scorer: VisualScoringMode = VisualScoringMode.BASELINE,
        audio_scorer: AudioScoringMode = AudioScoringMode.BASELINE,
        decompose_query: bool = False,
    ) -> tuple[IngestedVideo, CompressionResponse]:
        ingestion_key = ingestion_cache_key(
            video_path=video_path,
            sample_count=sample_count,
            audio_window_seconds=audio_window_seconds,
        )
        ingested = self.cache.get_ingestion(ingestion_key)
        if ingested is None:
            ingested = self.ingestor.ingest(
                video_path=video_path,
                sample_count=sample_count,
                audio_window_seconds=audio_window_seconds,
            )
            self.cache.set_ingestion(ingestion_key, ingested)

        candidate_generator = self._candidate_generator_for(
            visual_scorer=visual_scorer,
            audio_scorer=audio_scorer,
        )
        candidates_key = candidate_cache_key(
            ingestion=ingested,
            query=query,
            visual_scorer=visual_scorer,
            audio_scorer=audio_scorer,
        )
        candidates = self.cache.get_candidates(candidates_key)
        if candidates is None:
            candidates = candidate_generator.generate(ingested, query=query)
            self.cache.set_candidates(candidates_key, candidates)

        compression = self.compressor.compress(
            CompressionRequest(
                video_id=ingested.video_id,
                query=query,
                duration_seconds=ingested.metadata.duration_seconds,
                preset=preset,
                decompose_query=decompose_query,
                visual_candidates=candidates.visual,
                audio_candidates=candidates.audio,
            )
        )
        return ingested, compression

    def _candidate_generator_for(
        self,
        visual_scorer: VisualScoringMode,
        audio_scorer: AudioScoringMode,
    ) -> BaselineCandidateGenerator:
        visual_adapter = None
        audio_transcriber = None
        audio_score_adapter = None

        if visual_scorer == VisualScoringMode.CLIP:
            visual_adapter = HuggingFaceClipFrameScorer()
        elif visual_scorer != VisualScoringMode.BASELINE:
            raise ValueError(f"unsupported visual scorer: {visual_scorer}")

        if audio_scorer == AudioScoringMode.WHISPER:
            audio_transcriber = FasterWhisperTranscriber()
        elif audio_scorer == AudioScoringMode.CLAP:
            audio_score_adapter = HuggingFaceClapAudioScorer()
        elif audio_scorer != AudioScoringMode.BASELINE:
            raise ValueError(f"unsupported audio scorer: {audio_scorer}")

        if visual_adapter is None and audio_transcriber is None and audio_score_adapter is None:
            return self.candidate_generator

        return BaselineCandidateGenerator(
            visual_scorer=visual_adapter,
            audio_transcriber=audio_transcriber,
            audio_scorer=audio_score_adapter,
        )
