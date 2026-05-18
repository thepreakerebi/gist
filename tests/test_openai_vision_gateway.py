import json
from pathlib import Path

import pytest

from gist.gateway.openai_vision import (
    OpenAIVisionGatewayError,
    create_responses_payload,
    extract_output_text,
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
