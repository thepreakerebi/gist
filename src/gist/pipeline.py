from pathlib import Path

from gist.candidates.baseline import BaselineCandidateGenerator
from gist.core.compressor import GistCompressor
from gist.core.modes import VisualScoringMode
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
    ) -> None:
        self.output_root = output_root
        self.ingestor = ingestor or MediaIngestor(output_root=output_root)
        self.candidate_generator = candidate_generator or BaselineCandidateGenerator()
        self.compressor = compressor or GistCompressor()

    def run(
        self,
        video_path: Path,
        query: str,
        preset: CompressionPreset = CompressionPreset.BALANCED,
        sample_count: int = 128,
        audio_window_seconds: float = 1.0,
        visual_scorer: VisualScoringMode = VisualScoringMode.BASELINE,
    ) -> tuple[IngestedVideo, CompressionResponse]:
        ingested = self.ingestor.ingest(
            video_path=video_path,
            sample_count=sample_count,
            audio_window_seconds=audio_window_seconds,
        )
        candidate_generator = self._candidate_generator_for(visual_scorer)
        candidates = candidate_generator.generate(ingested, query=query)
        compression = self.compressor.compress(
            CompressionRequest(
                video_id=ingested.video_id,
                query=query,
                duration_seconds=ingested.metadata.duration_seconds,
                preset=preset,
                visual_candidates=candidates.visual,
                audio_candidates=candidates.audio,
            )
        )
        return ingested, compression

    def _candidate_generator_for(
        self,
        visual_scorer: VisualScoringMode,
    ) -> BaselineCandidateGenerator:
        if visual_scorer == VisualScoringMode.BASELINE:
            return self.candidate_generator
        if visual_scorer == VisualScoringMode.CLIP:
            return BaselineCandidateGenerator(visual_scorer=HuggingFaceClipFrameScorer())
        raise ValueError(f"unsupported visual scorer: {visual_scorer}")
