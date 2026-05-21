import io
import json
from urllib import error

import pytest

from gist.core.compressor import GistCompressor
from gist.core.schemas import Candidate, CompressionRequest
from gist.gateway.ollama import OllamaGatewayError, OllamaTextGateway
from gist.gateway.schemas import GatewayRequest


class _FakeHttpResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_ollama_gateway_posts_evidence_prompt(monkeypatch) -> None:
    seen = {}

    def fake_urlopen(http_request, timeout):
        seen["url"] = http_request.full_url
        seen["timeout"] = timeout
        seen["payload"] = json.loads(http_request.data.decode("utf-8"))
        return _FakeHttpResponse({"response": "She says they should follow their passions."})

    monkeypatch.setattr("gist.gateway.ollama.request.urlopen", fake_urlopen)
    compression = GistCompressor().compress(
        CompressionRequest(
            video_id="demo",
            query="What does she say?",
            duration_seconds=60,
            audio_candidates=[
                Candidate(
                    id="a1",
                    timestamp_seconds=10,
                    text="You have your robotics and I want to be awesome in space.",
                )
            ],
        )
    )

    response = OllamaTextGateway(model="llama3.2:3b").answer(
        GatewayRequest(query=compression.query, compression=compression)
    )

    assert response.answer == "She says they should follow their passions."
    assert response.provider == "ollama:llama3.2:3b"
    assert seen["url"] == "http://localhost:11434/api/generate"
    assert seen["payload"]["stream"] is False
    assert "Evidence:" in seen["payload"]["prompt"]


def test_ollama_gateway_reports_connection_failure(monkeypatch) -> None:
    def fake_urlopen(_http_request, timeout):
        raise error.URLError("connection refused")

    monkeypatch.setattr("gist.gateway.ollama.request.urlopen", fake_urlopen)
    compression = GistCompressor().compress(
        CompressionRequest(
            video_id="demo",
            query="pricing",
            duration_seconds=60,
            audio_candidates=[Candidate(id="a1", timestamp_seconds=10, text="pricing starts")],
        )
    )

    with pytest.raises(OllamaGatewayError, match="Could not reach Ollama"):
        OllamaTextGateway().answer(GatewayRequest(query="pricing", compression=compression))
