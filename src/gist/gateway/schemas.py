from pydantic import BaseModel

from gist.core.schemas import CompressionResponse


class GatewayRequest(BaseModel):
    query: str
    compression: CompressionResponse


class GatewayResponse(BaseModel):
    answer: str
    context: str
    provider: str

