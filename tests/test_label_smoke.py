from pathlib import Path

from gist.eval import label_smoke


def test_label_smoke_writes_contract_outputs(tmp_path: Path, capsys) -> None:
    exit_code = label_smoke.main(["--output-dir", str(tmp_path / "label-smoke")])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert (tmp_path / "label-smoke" / "extraction.json").exists()
    assert (tmp_path / "label-smoke" / "extraction.csv").exists()
    assert (tmp_path / "label-smoke" / "report.html").exists()
    assert (tmp_path / "label-smoke" / "label-smoke.json").exists()
    assert "passed=yes" in captured.out
    assert "customer-objections" in (tmp_path / "label-smoke" / "report.html").read_text()
