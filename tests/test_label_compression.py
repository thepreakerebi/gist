import json
from pathlib import Path

from gist.core.presets import CompressionPreset
from gist.core.schemas import CompressionMetrics, CompressionResponse, Modality, SelectedCandidate
from gist.label_compression import main


def test_label_compression_labels_existing_compression(tmp_path: Path, capsys) -> None:
    compression_path = tmp_path / "compression.json"
    output_dir = tmp_path / "labels"
    compression = _compression(
        selected=[
            _item(
                "a-1",
                text="The customer has a pricing objection because the plan is too expensive.",
                clip_start=30,
                clip_end=45,
            )
        ]
    )
    compression_path.write_text(json.dumps({"compression": compression.model_dump(mode="json")}))

    exit_code = main(
        [
            "--compression",
            str(compression_path),
            "--task",
            "find every time prospects complain about pricing",
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    extraction = json.loads((output_dir / "extraction.json").read_text())
    assert extraction["schema_name"] == "customer_objections"
    assert extraction["items"][0]["label"] == "pricing objection"
    assert (output_dir / "extraction.csv").exists()
    assert (output_dir / "extraction.md").exists()
    assert (output_dir / "extraction.html").exists()
    quality = json.loads((output_dir / "quality.json").read_text())
    assert quality["item_count"] == 1
    assert quality["evidence_count"] == 1
    assert quality["warnings"] == []
    assert (output_dir / "quality.md").exists()
    assert (output_dir / "quality.html").exists()
    assert "Gist Label Report" in (output_dir / "report.html").read_text()
    assert "items=1" in captured.out
    assert "quality_passed=yes" in captured.out
    assert "quality=" in captured.out
    assert "report=" in captured.out


def test_label_compression_accepts_preset_override(tmp_path: Path, capsys) -> None:
    compression_path = tmp_path / "compression.json"
    output_dir = tmp_path / "labels"
    compression = _compression(
        selected=[
            _item(
                "a-1",
                text="The buyer asked for an automation workflow improvement.",
                clip_start=20,
                clip_end=34,
            )
        ],
    )
    compression_path.write_text(json.dumps({"compression": compression.model_dump(mode="json")}))

    exit_code = main(
        [
            "--compression",
            str(compression_path),
            "--task",
            "find every time prospects complain about pricing",
            "--preset",
            "feature-requests",
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    extraction = json.loads((output_dir / "extraction.json").read_text())
    assert extraction["schema_name"] == "feature_requests"
    assert "extraction_preset=feature-requests" in captured.out


def _compression(selected: list[SelectedCandidate]) -> CompressionResponse:
    return CompressionResponse(
        video_id="video",
        query="Find pricing objections.",
        preset=CompressionPreset.BALANCED,
        selected=selected,
        metrics=CompressionMetrics(
            input_candidates=len(selected),
            selected_candidates=len(selected),
            visual_selected=sum(item.modality == Modality.VISUAL for item in selected),
            audio_selected=sum(item.modality == Modality.AUDIO for item in selected),
            estimated_candidate_reduction_ratio=1,
            estimated_candidate_reduction_percent=0,
            dropped_candidates=0,
            budget_preset_used=CompressionPreset.BALANCED,
        ),
    )


def _item(
    id_: str,
    text: str,
    clip_start: float,
    clip_end: float,
) -> SelectedCandidate:
    return SelectedCandidate(
        id=id_,
        modality=Modality.AUDIO,
        timestamp_seconds=clip_start,
        text=text,
        clip_start_seconds=clip_start,
        clip_end_seconds=clip_end,
        selection_rank=1,
        relevance_score=0.8,
        normalized_score=0.8,
        mmr_score=0.7,
        source_score_type="test",
        reason="test",
    )
