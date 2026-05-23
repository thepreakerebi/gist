import json
from pathlib import Path

from gist.core.presets import CompressionPreset
from gist.core.schemas import CompressionMetrics, CompressionResponse, Modality, SelectedCandidate
from gist.gateway.structured import (
    ExtractionField,
    ExtractionSchema,
    LocalStructuredExtractor,
    build_structured_extraction_prompt,
    extract_from_compression_file,
    main,
)


def test_structured_extractor_returns_timestamped_items(tmp_path: Path) -> None:
    clip_path = tmp_path / "clip.mp4"
    compression = _compression(
        selected=[
            _item(
                "a-1",
                text="The buyer says pricing is too expensive but the workflow is useful.",
                clip_start=30,
                clip_end=45,
                clip_path=clip_path,
            )
        ]
    )
    schema = ExtractionSchema(
        name="sales_feedback",
        item_type="feedback",
        labels=["pricing objection", "feature request"],
        fields=[
            ExtractionField(name="label", required=True),
            ExtractionField(name="summary", required=True),
            ExtractionField(name="sentiment"),
        ],
    )

    response = LocalStructuredExtractor().extract(schema, compression)

    assert response.provider == "local-structured-extractor"
    assert response.items[0].label == "pricing objection"
    assert response.items[0].timestamp_start_seconds == 30
    assert response.items[0].timestamp_end_seconds == 45
    assert response.items[0].clip_path == str(clip_path)
    assert response.items[0].values["sentiment"] == "negative"


def test_structured_extractor_skips_evidence_without_matching_label() -> None:
    compression = _compression(
        selected=[
            _item(
                "a-1",
                text="The speaker introduces the agenda.",
                clip_start=0,
                clip_end=10,
            )
        ]
    )
    schema = ExtractionSchema(
        name="sales_feedback",
        labels=["pricing objection"],
    )

    response = LocalStructuredExtractor().extract(schema, compression)

    assert response.items == []


def test_structured_prompt_includes_schema_and_evidence() -> None:
    compression = _compression(
        selected=[_item("a-1", text="Pricing is too expensive.", clip_start=1, clip_end=2)]
    )
    schema = ExtractionSchema(
        name="sales_feedback",
        description="Find sales moments.",
        labels=["pricing objection"],
        fields=[ExtractionField(name="summary", description="Moment summary")],
    )

    prompt = build_structured_extraction_prompt(schema, compression)

    assert "Return strict JSON" in prompt
    assert "sales_feedback" in prompt
    assert "pricing objection" in prompt
    assert "Pricing is too expensive" in prompt


def test_extraction_schema_loads_from_file(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "name": "product_mentions",
                "labels": ["product mention"],
                "fields": [{"name": "summary", "required": True}],
            }
        )
    )

    schema = ExtractionSchema.from_file(schema_path)

    assert schema.name == "product_mentions"
    assert schema.labels == ["product mention"]
    assert schema.fields[0].name == "summary"


def test_extract_from_compression_file_writes_model_contract(tmp_path: Path) -> None:
    compression_path = tmp_path / "compression.json"
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "name": "sales_feedback",
                "labels": ["pricing objection"],
                "fields": [{"name": "summary", "required": True}],
            }
        )
    )
    compression = _compression(
        selected=[
            _item(
                "a-1",
                text="The buyer says pricing is too expensive.",
                clip_start=30,
                clip_end=45,
            )
        ]
    )
    compression_path.write_text(
        json.dumps({"compression": compression.model_dump(mode="json")})
    )

    response = extract_from_compression_file(compression_path, schema_path)

    assert response.schema_name == "sales_feedback"
    assert response.items[0].label == "pricing objection"


def test_structured_extract_cli_writes_output(tmp_path: Path) -> None:
    compression_path = tmp_path / "compression.json"
    schema_path = tmp_path / "schema.json"
    output_path = tmp_path / "extraction.json"
    schema_path.write_text(
        json.dumps(
            {
                "name": "sales_feedback",
                "labels": ["pricing objection"],
                "fields": [{"name": "summary", "required": True}],
            }
        )
    )
    compression = _compression(
        selected=[
            _item(
                "a-1",
                text="The buyer says pricing is too expensive.",
                clip_start=30,
                clip_end=45,
            )
        ]
    )
    compression_path.write_text(
        json.dumps({"compression": compression.model_dump(mode="json")})
    )

    exit_code = main(
        [
            "--compression",
            str(compression_path),
            "--schema",
            str(schema_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(output_path.read_text())["items"][0]["label"] == "pricing objection"


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
    clip_path: Path | None = None,
) -> SelectedCandidate:
    return SelectedCandidate(
        id=id_,
        modality=Modality.AUDIO,
        timestamp_seconds=clip_start,
        text=text,
        clip_path=clip_path,
        clip_start_seconds=clip_start,
        clip_end_seconds=clip_end,
        selection_rank=1,
        relevance_score=0.8,
        normalized_score=0.8,
        mmr_score=0.7,
        source_score_type="test",
        reason="test",
    )
