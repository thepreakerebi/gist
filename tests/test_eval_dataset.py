import pytest

from gist.eval.dataset import load_jsonl_dataset


def test_load_jsonl_dataset(tmp_path) -> None:
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text(
        '{"id":"case-1","video_id":"v1","query":"pricing","duration_seconds":60}\n'
    )

    examples = load_jsonl_dataset(dataset)

    assert len(examples) == 1
    assert examples[0].id == "case-1"


def test_load_jsonl_dataset_reports_invalid_json_line(tmp_path) -> None:
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text("{bad json}\n")

    with pytest.raises(ValueError, match="invalid JSON on line 1"):
        load_jsonl_dataset(dataset)

