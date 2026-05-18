import sys

from gist.core.compressor import GistCompressor
from gist.core.schemas import Candidate, CompressionRequest
from gist.gateway.context import render_evidence_context
from gist.gateway.echo import EchoGateway
from gist.gateway.schemas import GatewayRequest
from gist.gateway.subprocess import SubprocessVideoLlmGateway


def test_render_evidence_context_includes_timestamps_and_reasons() -> None:
    compression = GistCompressor().compress(
        CompressionRequest(
            video_id="demo",
            query="pricing",
            duration_seconds=60,
            visual_candidates=[
                Candidate(id="v1", timestamp_seconds=10, text="pricing slide")
            ],
        )
    )

    context = render_evidence_context(compression)

    assert "pricing slide" in context
    assert "10.00s" in context
    assert "reason:" in context


def test_echo_gateway_returns_context() -> None:
    compression = GistCompressor().compress(
        CompressionRequest(
            video_id="demo",
            query="pricing",
            duration_seconds=60,
            audio_candidates=[
                Candidate(id="a1", timestamp_seconds=10, text="pricing starts")
            ],
        )
    )

    response = EchoGateway().answer(GatewayRequest(query="pricing", compression=compression))

    assert response.provider == "echo"
    assert "pricing starts" in response.context


def test_subprocess_gateway_reads_stdin_and_parses_json_stdout() -> None:
    compression = GistCompressor().compress(
        CompressionRequest(
            video_id="demo",
            query="pricing",
            duration_seconds=60,
            audio_candidates=[
                Candidate(id="a1", timestamp_seconds=10, text="pricing starts")
            ],
        )
    )
    gateway = SubprocessVideoLlmGateway(
        command=[
            sys.executable,
            "-c",
            (
                "import json,sys;"
                "payload=json.load(sys.stdin);"
                "print(json.dumps({'answer': payload['query'], 'provider': 'fake-video-llm'}))"
            ),
        ]
    )

    response = gateway.answer(GatewayRequest(query="pricing", compression=compression))

    assert response.answer == "pricing"
    assert response.provider == "fake-video-llm"
    assert "pricing starts" in response.context
