from fastapi import APIRouter

from gist.core.compressor import GistCompressor
from gist.core.schemas import CompressionRequest, CompressionResponse

router = APIRouter(prefix="/v1", tags=["compressions"])
compressor = GistCompressor()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/compressions", response_model=CompressionResponse)
def create_compression(request: CompressionRequest) -> CompressionResponse:
    return compressor.compress(request)

