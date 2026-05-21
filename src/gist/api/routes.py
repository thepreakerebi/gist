from fastapi import APIRouter, HTTPException

from gist.audio.errors import AudioTranscriptionError
from gist.api.schemas import (
    LocalVideoCompressionRequest,
    LocalVideoCompressionResponse,
    MediaIngestionRequest,
)
from gist.core.compressor import GistCompressor
from gist.core.schemas import CompressionRequest, CompressionResponse
from gist.media.ingestion import MediaIngestor
from gist.media.models import IngestedVideo
from gist.pipeline import LocalCompressionPipeline
from gist.vision.errors import VisualScoringError

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
        processing_mode=request.processing_mode,
    )


@router.post(
    "/local-video-compressions",
    response_model=LocalVideoCompressionResponse,
    tags=["compressions"],
)
def create_local_video_compression(
    request: LocalVideoCompressionRequest,
) -> LocalVideoCompressionResponse:
    try:
        ingestion, compression = LocalCompressionPipeline(output_root=request.output_root).run(
            video_path=request.video_path,
            query=request.query,
            preset=request.preset,
            sample_count=request.sample_count,
            audio_window_seconds=request.audio_window_seconds,
            processing_mode=request.processing_mode,
            visual_scorer=request.visual_scorer,
            audio_scorer=request.audio_scorer,
            adaptive_budget=request.adaptive_budget,
            decompose_query=request.decompose_query,
            token_estimator=request.token_estimator,
        )
        return LocalVideoCompressionResponse(ingestion=ingestion, compression=compression)
    except AudioTranscriptionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except VisualScoringError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
