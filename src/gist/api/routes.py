from fastapi import APIRouter

from gist.api.schemas import MediaIngestionRequest
from gist.core.compressor import GistCompressor
from gist.core.schemas import CompressionRequest, CompressionResponse
from gist.media.ingestion import MediaIngestor
from gist.media.models import IngestedVideo

router = APIRouter(prefix="/v1", tags=["compressions"])
compressor = GistCompressor()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/compressions", response_model=CompressionResponse)
def create_compression(request: CompressionRequest) -> CompressionResponse:
    return compressor.compress(request)


@router.post("/ingestions", response_model=IngestedVideo, tags=["ingestions"])
def create_ingestion(request: MediaIngestionRequest) -> IngestedVideo:
    ingestor = MediaIngestor(output_root=request.output_root)
    return ingestor.ingest(
        video_path=request.video_path,
        sample_count=request.sample_count,
        audio_window_seconds=request.audio_window_seconds,
    )
