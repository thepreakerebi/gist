import json
from pathlib import Path

import pytest

from gist.gateway.openai_vision import (
    OpenAIVisionGatewayError,
    create_responses_payload,
    extract_output_text,
    sample_evidence_frames,
)


def test_create_responses_payload_includes_context_and_images(tmp_path: Path) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"fake-jpeg")
    payload = create_responses_payload(
        gateway_payload={
            "query": "What is shown?",
            "context": "Selected evidence:\n- frame",
        },
        frame_paths=[frame],
        model="gpt-test",
        detail="low",
    )

    content = payload["input"][0]["content"]
    assert payload["model"] == "gpt-test"
    assert content[0]["type"] == "input_text"
    assert "What is shown?" in content[0]["text"]
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/jpeg;base64,")


def test_create_responses_payload_includes_task_guidance(tmp_path: Path) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"fake-jpeg")
    payload = create_responses_payload(
        gateway_payload={
            "query": "How many apples are shown?",
            "context": "Selected evidence:\n- visual frame",
            "compression": {"query_intent": "counting_comparison"},
        },
        frame_paths=[frame],
        model="gpt-test",
        detail="low",
    )

    prompt = payload["input"][0]["content"][0]["text"]
    assert "Task guidance:" in prompt
    assert "Count or compare visible entities" in prompt
    assert "answer with only the single best choice letter" in prompt


def test_sample_evidence_frames_uses_anchor_offsets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clip_a = tmp_path / "a.mp4"
    clip_b = tmp_path / "b.mp4"
    clip_a.write_bytes(b"fake-video")
    clip_b.write_bytes(b"fake-video")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> object:
        calls.append(command)
        Path(command[-1]).write_bytes(b"fake-jpeg")
        return object()

    monkeypatch.setattr("gist.gateway.openai_vision.subprocess.run", fake_run)

    frames = sample_evidence_frames(
        evidence=[
            {
                "clip_path": str(clip_a),
                "timestamp_seconds": 14.0,
                "clip_start_seconds": 10.0,
                "clip_end_seconds": 18.0,
            },
            {
                "clip_path": str(clip_b),
                "timestamp_seconds": 31.0,
                "clip_start_seconds": 30.0,
                "clip_end_seconds": 36.0,
            },
        ],
        output_dir=tmp_path / "frames",
        max_frames=2,
    )

    assert len(frames) == 2
    assert calls[0][calls[0].index("-ss") + 1] == "4.000"
    assert calls[1][calls[1].index("-ss") + 1] == "1.000"


def test_sample_evidence_frames_spreads_budget_over_top_clips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clips = [tmp_path / f"{index}.mp4" for index in range(3)]
    for clip in clips:
        clip.write_bytes(b"fake-video")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> object:
        calls.append(command)
        Path(command[-1]).write_bytes(b"fake-jpeg")
        return object()

    monkeypatch.setattr("gist.gateway.openai_vision.subprocess.run", fake_run)

    frames = sample_evidence_frames(
        evidence=[{"clip_path": str(clip)} for clip in clips],
        output_dir=tmp_path / "frames",
        max_frames=2,
    )

    assert len(frames) == 2
    assert [Path(call[call.index("-i") + 1]) for call in calls] == clips[:2]


def test_extract_output_text_supports_output_text_shortcut() -> None:
    assert extract_output_text({"output_text": " Mars "}) == "Mars"


def test_extract_output_text_supports_responses_output_items() -> None:
    response = {
        "output": [
            {
                "content": [
                    {
                        "type": "output_text",
                        "text": "Mars",
                    }
                ]
            }
        ]
    }

    assert extract_output_text(response) == "Mars"


def test_extract_output_text_rejects_empty_response() -> None:
    with pytest.raises(OpenAIVisionGatewayError, match="output text"):
        extract_output_text({})


def test_gateway_script_reports_missing_key(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    completed = subprocess.run(
        [sys.executable, "scripts/run_openai_video_gateway.py"],
        input=json.dumps({"query": "hello", "evidence": []}),
        capture_output=True,
        text=True,
        env=env,
        cwd=Path(__file__).parents[1],
    )

    assert completed.returncode == 2
    assert "OPENAI_API_KEY" in completed.stderr
