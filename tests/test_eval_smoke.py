import json

from gist.eval.smoke import run


def test_smoke_runner_generates_reports(tmp_path, capsys) -> None:
    output_dir = tmp_path / "smoke"

    run(
        [
            "--output-dir",
            str(output_dir),
            "--query",
            "pricing",
            "--expected-answer",
            "pricing",
            "--sample-count",
            "4",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads((output_dir / "smoke-report.json").read_text())
    assert "answer_score=1.0" in captured.out
    assert (output_dir / "smoke-report.md").exists()
    assert (output_dir / "smoke-report.html").exists()
    assert payload["results"][0]["variants"][0]["answer_score"] == 1
