import json

from gist.eval.cli import run


def test_eval_cli_writes_json_and_markdown_reports(tmp_path) -> None:
    dataset = tmp_path / "eval.jsonl"
    output = tmp_path / "report.json"
    markdown = tmp_path / "report.md"
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
            "--preset",
            "aggressive",
        ]
    )

    assert output.exists()
    assert markdown.exists()
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
        ]
    )

    payload = json.loads(output.read_text())
    assert len(payload["variants"]) == 1
    assert payload["variants"][0]["name"] == "gist_configured"
