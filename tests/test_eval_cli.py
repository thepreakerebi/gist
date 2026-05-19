import json
import sys

from gist.eval.cli import run


def test_eval_cli_writes_json_and_markdown_reports(tmp_path) -> None:
    dataset = tmp_path / "eval.jsonl"
    output = tmp_path / "report.json"
    markdown = tmp_path / "report.md"
    html = tmp_path / "report.html"
    dataset.write_text(
        json.dumps(
            {
                "id": "case-1",
                "video_id": "v1",
                "query": "pricing",
                "duration_seconds": 60,
                "relevant_timestamps": [10],
                "visual_candidates": [
                    {"id": "v-1", "timestamp_seconds": 10, "text": "pricing slide"}
                ],
            }
        )
        + "\n"
    )

    run(
        [
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--output-root",
            str(tmp_path / "eval-cache"),
            "--markdown-output",
            str(markdown),
            "--html-output",
            str(html),
            "--preset",
            "aggressive",
        ]
    )

    assert output.exists()
    assert markdown.exists()
    assert html.exists()
    payload = json.loads(output.read_text())
    assert payload["summary"]["examples"] == 1
    assert len(payload["variants"]) == 5
    assert "avg_token_reduction_percent" in next(iter(payload["summary"]["variants"].values()))


def test_eval_cli_supports_single_config_mode(tmp_path) -> None:
    dataset = tmp_path / "eval.jsonl"
    output = tmp_path / "report.json"
    dataset.write_text(
        json.dumps(
            {
                "id": "case-1",
                "video_id": "v1",
                "query": "pricing",
                "duration_seconds": 60,
                "visual_candidates": [
                    {"id": "v-1", "timestamp_seconds": 10, "text": "pricing slide"}
                ],
            }
        )
        + "\n"
    )

    run(
        [
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--single-config",
            "--preset",
            "aggressive",
            "--visual-scorer",
            "baseline",
            "--audio-scorer",
            "baseline",
        ]
    )

    payload = json.loads(output.read_text())
    assert len(payload["variants"]) == 1
    assert payload["variants"][0]["name"] == "gist_configured"
    assert payload["variants"][0]["visual_scorer"] == "baseline"
    assert payload["variants"][0]["audio_scorer"] == "baseline"


def test_eval_cli_supports_benchmark_sota_sweep(tmp_path) -> None:
    dataset = tmp_path / "video_mme.jsonl"
    output = tmp_path / "report.json"
    dataset.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "video_id": "v1",
                "question": "What happens?",
                "duration": 60,
                "answer": "A",
                "options": ["A", "B"],
            }
        )
        + "\n"
    )

    run(
        [
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--benchmark",
            "video_mme",
            "--sota-sweep",
        ]
    )

    payload = json.loads(output.read_text())
    assert [variant["name"] for variant in payload["variants"]] == [
        "gist_core",
        "gist_scene_clip",
        "gist_scene_router_adaptive",
        "gist_scene_router_adaptive_whisper",
        "gist_task_router_adaptive_whisper",
        "gist_scene_spatial",
    ]
    assert payload["results"][0]["variants"][0]["answer_score"] == 0


def test_eval_cli_supports_subprocess_gateway(tmp_path) -> None:
    dataset = tmp_path / "eval.jsonl"
    output = tmp_path / "report.json"
    dataset.write_text(
        json.dumps(
            {
                "id": "case-1",
                "video_id": "v1",
                "query": "pricing",
                "duration_seconds": 60,
                "expected_answer": "pricing",
                "visual_candidates": [
                    {"id": "v-1", "timestamp_seconds": 10, "text": "pricing slide"}
                ],
            }
        )
        + "\n"
    )

    run(
        [
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--single-config",
            "--gateway-command",
            (
                f"{sys.executable} -c "
                "\"import json,sys; payload=json.load(sys.stdin); "
                "print(json.dumps({'answer': payload['query']}))\""
            ),
        ]
    )

    payload = json.loads(output.read_text())
    variant = payload["results"][0]["variants"][0]
    assert variant["predicted_answer"] == "pricing"
    assert variant["answer_score"] == 1
