from pathlib import Path
import subprocess
import sys

from gist.gateway.hf_vision import build_messages, extract_pipeline_text


def test_build_messages_includes_images_and_prompt() -> None:
    messages = build_messages(
        {
            "query": "What happens?",
            "context": "Selected evidence:\n- rocket launch",
        },
        images=["image-a", "image-b"],
    )

    assert messages[0]["role"] == "user"
    assert messages[0]["content"][0] == {"type": "image", "image": "image-a"}
    assert messages[0]["content"][1] == {"type": "image", "image": "image-b"}
    assert messages[0]["content"][2]["type"] == "text"
    assert "What happens?" in messages[0]["content"][2]["text"]


def test_extract_pipeline_text_supports_string_output() -> None:
    assert extract_pipeline_text("Mars") == "Mars"


def test_extract_pipeline_text_supports_generated_text_string() -> None:
    assert extract_pipeline_text([{"generated_text": "Mars"}]) == "Mars"


def test_extract_pipeline_text_supports_chat_message_output() -> None:
    output = [
        {
            "generated_text": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": [{"type": "text", "text": "Mars"}]},
            ]
        }
    ]

    assert extract_pipeline_text(output) == "Mars"


def test_hf_gateway_script_reports_invalid_json() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/run_hf_vlm_gateway.py", "--model", "fake/model"],
        input="{bad json}",
        capture_output=True,
        text=True,
        cwd=Path(__file__).parents[1],
    )

    assert completed.returncode == 2
    assert "run_hf_vlm_gateway.py:" in completed.stderr
