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


def test_label_batch_filters_and_writes_manifest(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    keep = root / "video-a" / "pricing" / "compression.json"
    drop = root / "video-b" / "security" / "compression.json"
    _write_compression(
        keep,
        "The customer says pricing is too expensive.",
        query="Find pricing objections.",
        evidence_count=2,
    )
    _write_compression(
        drop,
        "The buyer has a security concern about data privacy.",
        query="Find security objections.",
    )
    output_dir = tmp_path / "batch"
    manifest = tmp_path / "selected.jsonl"

    exit_code = main(
        [
            "--input-root",
            str(root),
            "--task",
            "find pricing objections",
            "--output-dir",
            str(output_dir),
            "--include",
            "*pricing*",
            "--exclude",
            "*security*",
            "--query-contains",
            "pricing",
            "--min-evidence",
            "2",
            "--write-manifest",
            str(manifest),
        ]
    )

    assert exit_code == 0
    report = json.loads((output_dir / "batch-report.json").read_text())
    assert report["case_count"] == 1
    assert report["manifest_path"] == str(manifest)
    assert str(keep) in manifest.read_text()
    assert str(drop) not in manifest.read_text()


def test_label_batch_reads_manifest(tmp_path: Path) -> None:
    compression = tmp_path / "runs" / "video-a" / "pricing" / "compression.json"
    _write_compression(compression, "The customer says pricing is too expensive.")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"compression": str(compression)}) + "\n")
    output_dir = tmp_path / "batch"

    exit_code = main(
        [
            "--manifest",
            str(manifest),
            "--task",
            "find pricing objections",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    report = json.loads((output_dir / "batch-report.json").read_text())
    assert report["case_count"] == 1


def _write_compression(
    path: Path,
    text: str,
    query: str = "Find customer objections.",
    evidence_count: int = 1,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = [
        SelectedCandidate(
            id=f"{path.parent.parent.name}:audio:{index}",
            modality=Modality.AUDIO,
            timestamp_seconds=10 + index,
            text=text,
            clip_start_seconds=10 + index,
            clip_end_seconds=20 + index,
            selection_rank=index,
            relevance_score=0.8,
            normalized_score=0.8,
            mmr_score=0.7,
            source_score_type="test",
            reason="test",
        )
        for index in range(1, evidence_count + 1)
    ]
    compression = CompressionResponse(
        video_id=path.parent.parent.name,
        query=query,
        preset=CompressionPreset.BALANCED,
        selected=selected,
        metrics=CompressionMetrics(
            input_candidates=evidence_count,
            selected_candidates=evidence_count,
            visual_selected=0,
            audio_selected=evidence_count,
            estimated_candidate_reduction_ratio=1,
            estimated_candidate_reduction_percent=0,
            dropped_candidates=0,
            budget_preset_used=CompressionPreset.BALANCED,
        ),
    )
    path.write_text(json.dumps({"compression": compression.model_dump(mode="json")}))
