from pathlib import Path

import gist.label as label


def test_label_cli_suggests_preset_and_calls_gist(tmp_path: Path, monkeypatch, capsys) -> None:
    calls = []

    def fake_gist_main(argv):
        calls.append(argv)
        return 0

    monkeypatch.setattr(label, "gist_main", fake_gist_main)

    exit_code = label.main(
        [
            str(tmp_path / "video.mp4"),
            "--task",
            "find every time prospects complain about pricing",
            "--output-dir",
            str(tmp_path / "labels"),
            "--no-clips",
            "--quiet",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "--extraction-preset" in calls[0]
    assert "customer-objections" in calls[0]
    assert "--extraction-output" in calls[0]
    assert str(tmp_path / "labels" / "extraction.json") in calls[0]
    assert "--extraction-csv-output" in calls[0]
    assert str(tmp_path / "labels" / "extraction.csv") in calls[0]
    assert "recommended_preset=customer-objections" in captured.out
    assert "schema_name=customer_objections" in captured.out


def test_label_cli_allows_preset_override(tmp_path: Path, monkeypatch, capsys) -> None:
    calls = []

    def fake_gist_main(argv):
        calls.append(argv)
        return 0

    monkeypatch.setattr(label, "gist_main", fake_gist_main)

    exit_code = label.main(
        [
            str(tmp_path / "video.mp4"),
            "--task",
            "find every time prospects complain about pricing",
            "--preset",
            "sales-feedback",
            "--output-dir",
            str(tmp_path / "labels"),
            "--query",
            "custom query",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "sales-feedback" in calls[0]
    assert "custom query" in calls[0]
    assert "recommended_preset=customer-objections" in captured.out
    assert "extraction_preset=sales-feedback" in captured.out
    assert "schema_name=sales_feedback" in captured.out
