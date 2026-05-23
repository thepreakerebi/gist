import json
from pathlib import Path

from gist.core.presets import CompressionPreset
from gist.core.schemas import CompressionMetrics, CompressionResponse, Modality, SelectedCandidate
from gist.eval.extraction_smoke import main


def test_extraction_smoke_runs_extract_and_eval(tmp_path: Path) -> None:
    compression_path = tmp_path / "compression.json"
    schema_path = tmp_path / "schema.json"
    output_dir = tmp_path / "smoke"
    schema_path.write_text(
        json.dumps(
            {
                "name": "sales_feedback",
                "labels": ["pricing objection"],
                "fields": [{"name": "summary", "required": True}],
            }
        )
    )
    compression_path.write_text(
        json.dumps(
            {
                "compression": _compression(
                    selected=[
                        _item(
                            "a-1",
                            text="The buyer says pricing is too expensive.",
                            clip_start=30,
                            clip_end=45,
                        )
                    ]
                ).model_dump(mode="json")
            }
        )
    )

    exit_code = main(
        [
            "--compression",
            str(compression_path),
            "--schema",
            str(schema_path),
            "--output-dir",
            str(output_dir),
            "--case-id",
            "sales-smoke",
            "--expected-label",
            "pricing objection",
            "--support-term",
            "pricing",
            "--support-term",
            "expensive",
            "--expected-start-seconds",
            "28",
            "--expected-end-seconds",
            "46",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "extraction.json").exists()
    assert (output_dir / "extraction-eval.json").exists()
    assert (output_dir / "extraction-eval.md").exists()
    assert (output_dir / "extraction-eval.dataset.jsonl").exists()


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
