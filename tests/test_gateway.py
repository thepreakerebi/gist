import sys
from pathlib import Path

from gist.core.compressor import GistCompressor
from gist.core.schemas import Candidate, CompressionRequest, Modality
from gist.gateway.context import render_evidence_context
from gist.gateway.echo import EchoGateway
from gist.gateway.local_text import LocalTextEvidenceGateway
from gist.gateway.schemas import GatewayRequest
from gist.gateway.subprocess import (
    PersistentSubprocessVideoLlmGateway,
    SubprocessVideoLlmGateway,
    build_gateway_payload,
)


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


def test_local_text_gateway_answers_from_selected_evidence() -> None:
    compression = GistCompressor().compress(
        CompressionRequest(
            video_id="demo",
            query="Why is he afraid?",
            duration_seconds=60,
            audio_candidates=[
                Candidate(
                    id="a1",
                    timestamp_seconds=10,
                    text="He is freaked out because he has nightmares.",
                )
            ],
        )
    )

    response = LocalTextEvidenceGateway().answer(
        GatewayRequest(query="Why is he afraid?", compression=compression)
    )

    assert response.provider == "local-text-evidence"
    assert "nightmares" in response.answer
    assert "freaked out" in response.context


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


def test_persistent_subprocess_gateway_reuses_process() -> None:
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
    gateway = PersistentSubprocessVideoLlmGateway(
        command=[
            sys.executable,
            "-c",
            (
                "import json,sys;"
                "count=0\n"
                "for line in sys.stdin:\n"
                " payload=json.loads(line)\n"
                " if payload.get('type') == 'shutdown': break\n"
                " count += 1\n"
                " print(json.dumps({'answer': f\"{payload['query']}:{count}\", "
                "'provider': 'fake-persistent'}), flush=True)\n"
            ),
        ]
    )

    first = gateway.answer(GatewayRequest(query="pricing", compression=compression))
    second = gateway.answer(GatewayRequest(query="pricing", compression=compression))
    gateway.close()

    assert first.answer == "pricing:1"
    assert second.answer == "pricing:2"
    assert second.provider == "fake-persistent"


def test_subprocess_gateway_payload_includes_spatial_debug_paths(tmp_path: Path) -> None:
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
    selected = [
        compression.selected[0].model_copy(
            update={
                "modality": Modality.VISUAL,
                "spatial_mask_path": tmp_path / "mask.json",
                "spatial_mask_preview_path": tmp_path / "mask.svg",
                "spatial_mask_overlay_path": tmp_path / "overlay.svg",
            }
        )
    ]
    compression = compression.model_copy(update={"selected": selected})

    _context, payload = build_gateway_payload(
        GatewayRequest(query="pricing", compression=compression)
    )
    evidence = payload["evidence"][0]

    assert evidence["spatial_mask_path"] == str(tmp_path / "mask.json")
    assert evidence["spatial_mask_preview_path"] == str(tmp_path / "mask.svg")
    assert evidence["spatial_mask_overlay_path"] == str(tmp_path / "overlay.svg")
