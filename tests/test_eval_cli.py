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
            "--markdown-output",
            str(markdown),
            "--preset",
            "aggressive",
        ]
    )

    assert output.exists()
    assert markdown.exists()
    assert json.loads(output.read_text())["summary"]["examples"] == 1
