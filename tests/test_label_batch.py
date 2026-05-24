import json
from pathlib import Path

from gist.core.presets import CompressionPreset
from gist.core.schemas import CompressionMetrics, CompressionResponse, Modality, SelectedCandidate
from gist.label_batch import main


def test_label_batch_labels_multiple_compressions(tmp_path: Path, capsys) -> None:
    root = tmp_path / "runs"
    _write_compression(
        root / "video-a" / "pricing" / "compression.json",
        "The customer says pricing is too expensive.",
    )
    _write_compression(
        root / "video-b" / "security" / "compression.json",
        "The buyer has a security concern about data privacy.",
    )
    output_dir = tmp_path / "batch"

    exit_code = main(
        [
            "--input-root",
            str(root),
            "--task",
            "find every time prospects complain about pricing or security",
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    report = json.loads((output_dir / "batch-report.json").read_text())
    assert report["case_count"] == 2
    assert report["total_items"] == 2
    assert report["pass_rate"] == 1.0
    assert (output_dir / "batch-report.md").exists()
    assert (output_dir / "batch-report.html").exists()
    assert len(list(output_dir.glob("*/extraction.json"))) == 2
    assert "cases=2" in captured.out
    assert "report=" in captured.out


def test_label_batch_accepts_explicit_compressions(tmp_path: Path) -> None:
    first = tmp_path / "first" / "compression.json"
    second = tmp_path / "second" / "compression.json"
    _write_compression(first, "The customer says pricing is too expensive.")
    _write_compression(second, "The customer says pricing is too expensive.")
    output_dir = tmp_path / "batch"

    exit_code = main(
        [
            "--compression",
            str(second),
            "--compression",
            str(first),
            "--task",
            "find pricing objections",
            "--output-dir",
            str(output_dir),
            "--max-cases",
            "1",
        ]
    )

    assert exit_code == 0
    report = json.loads((output_dir / "batch-report.json").read_text())
    assert report["case_count"] == 1


def _write_compression(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = CompressionResponse(
        video_id=path.parent.parent.name,
        query="Find customer objections.",
        preset=CompressionPreset.BALANCED,
        selected=[
            SelectedCandidate(
                id=f"{path.parent.parent.name}:audio:1",
                modality=Modality.AUDIO,
                timestamp_seconds=10,
                text=text,
                clip_start_seconds=10,
                clip_end_seconds=20,
                selection_rank=1,
                relevance_score=0.8,
                normalized_score=0.8,
                mmr_score=0.7,
                source_score_type="test",
                reason="test",
            )
        ],
        metrics=CompressionMetrics(
            input_candidates=1,
            selected_candidates=1,
            visual_selected=0,
            audio_selected=1,
            estimated_candidate_reduction_ratio=1,
            estimated_candidate_reduction_percent=0,
            dropped_candidates=0,
            budget_preset_used=CompressionPreset.BALANCED,
        ),
    )
    path.write_text(json.dumps({"compression": compression.model_dump(mode="json")}))
