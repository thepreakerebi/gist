from gist.gateway.context import render_evidence_context
from gist.gateway.schemas import GatewayRequest, GatewayResponse


class EchoGateway:
    provider = "echo"

    def answer(self, request: GatewayRequest) -> GatewayResponse:
        context = render_evidence_context(request.compression)
        return GatewayResponse(
            answer="Echo gateway generated compressed evidence context only.",
            context=context,
            provider=self.provider,
        )

