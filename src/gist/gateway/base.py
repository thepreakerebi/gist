from typing import Protocol

from gist.gateway.schemas import GatewayRequest, GatewayResponse


class LlmGateway(Protocol):
    def answer(self, request: GatewayRequest) -> GatewayResponse:
        """Answer a query using compressed evidence context."""

