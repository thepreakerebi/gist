from gist.core.answering import answer_from_evidence
from gist.gateway.context import render_evidence_context
from gist.gateway.schemas import GatewayRequest, GatewayResponse


class LocalTextEvidenceGateway:
    provider = "local-text-evidence"

    def answer(self, request: GatewayRequest) -> GatewayResponse:
        answer = answer_from_evidence(request.compression)
        if not answer:
            answer = "I could not derive a reliable answer from the selected evidence."
        return GatewayResponse(
            answer=answer,
            context=render_evidence_context(request.compression),
            provider=self.provider,
        )
