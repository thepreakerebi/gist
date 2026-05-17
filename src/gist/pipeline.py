from pathlib import Path

from gist.candidates.baseline import BaselineCandidateGenerator
from gist.core.compressor import GistCompressor
from gist.core.presets import CompressionPreset
from gist.core.schemas import CompressionRequest, CompressionResponse
from gist.media.ingestion import MediaIngestor
from gist.media.models import IngestedVideo


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
    ) -> tuple[IngestedVideo, CompressionResponse]:
        ingested = self.ingestor.ingest(
            video_path=video_path,
            sample_count=sample_count,
            audio_window_seconds=audio_window_seconds,
        )
        candidates = self.candidate_generator.generate(ingested, query=query)
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

