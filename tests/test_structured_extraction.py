import csv
import json
import sys
from io import StringIO
from pathlib import Path

from gist.core.presets import CompressionPreset
from gist.core.schemas import CompressionMetrics, CompressionResponse, Modality, SelectedCandidate
from gist.gateway.structured import (
    ExtractionField,
    ExtractionSchema,
    LocalStructuredExtractor,
    StructuredExtractionError,
    SubprocessStructuredExtractor,
    build_structured_extraction_payload,
    build_structured_extraction_prompt,
    extract_from_compression_file,
    list_builtin_extraction_schemas,
    main,
    resolve_extraction_schema,
    schemas_main,
    suggest_extraction_preset,
)
from gist.reports.structured import (
    render_structured_extraction_csv,
    render_structured_extraction_html,
    render_structured_extraction_markdown,
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


def test_structured_payload_includes_schema_and_evidence() -> None:
    compression = _compression(
        selected=[_item("a-1", text="Pricing is too expensive.", clip_start=1, clip_end=2)]
    )
    schema = ExtractionSchema(
        name="sales_feedback",
        labels=["pricing objection"],
    )

    payload = build_structured_extraction_payload(schema, compression)

    assert payload["type"] == "gist.structured_extraction"
    assert payload["schema"]["name"] == "sales_feedback"
    assert payload["evidence"][0]["id"] == "a-1"
    assert "Pricing is too expensive" in payload["context"]


def test_subprocess_structured_extractor_parses_items() -> None:
    compression = _compression(
        selected=[_item("a-1", text="Pricing is too expensive.", clip_start=1, clip_end=2)]
    )
    schema = ExtractionSchema(
        name="sales_feedback",
        labels=["pricing objection"],
    )
    extractor = SubprocessStructuredExtractor(
        command=[
            sys.executable,
            "-c",
            (
                "import json,sys;"
                "payload=json.load(sys.stdin);"
                "item={"
                "'label':'pricing objection',"
                "'description':'Pricing is too expensive.',"
                "'timestamp_start_seconds':1,"
                "'timestamp_end_seconds':2,"
                "'evidence_id':'a-1',"
                "'evidence_rank':1,"
                "'confidence':0.9,"
                "'support_text':'Pricing is too expensive.'"
                "};"
                "print(json.dumps({'provider':'fake-extractor','items':[item]}))"
            ),
        ]
    )

    response = extractor.extract(schema, compression)

    assert response.provider == "fake-extractor"
    assert response.items[0].label == "pricing objection"


def test_subprocess_structured_extractor_rejects_invalid_stdout() -> None:
    compression = _compression(
        selected=[_item("a-1", text="Pricing is too expensive.", clip_start=1, clip_end=2)]
    )
    schema = ExtractionSchema(name="sales_feedback")
    extractor = SubprocessStructuredExtractor(
        command=[sys.executable, "-c", "print('not-json')"]
    )

    try:
        extractor.extract(schema, compression)
    except StructuredExtractionError as exc:
        assert "valid JSON" in str(exc)
    else:
        raise AssertionError("expected StructuredExtractionError")


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


def test_builtin_extraction_schemas_are_valid() -> None:
    schema_paths = sorted(Path("data/extraction").glob("*.schema.json"))

    schemas = [ExtractionSchema.from_file(path) for path in schema_paths]

    assert {schema.name for schema in schemas} >= {
        "customer_objections",
        "feature_requests",
        "meeting_decisions",
        "product_announcements",
        "sales_feedback",
    }
    assert all(schema.item_type for schema in schemas)
    assert all(schema.labels for schema in schemas)
    assert all(schema.fields for schema in schemas)


def test_packaged_extraction_schemas_match_repo_templates() -> None:
    repo_schemas = {
        path.name: json.loads(path.read_text())
        for path in sorted(Path("data/extraction").glob("*.schema.json"))
    }
    packaged_schemas = {
        path.name: json.loads(path.read_text())
        for path in sorted(Path("src/gist/data/extraction").glob("*.schema.json"))
    }

    assert packaged_schemas == repo_schemas


def test_lists_builtin_extraction_schemas() -> None:
    schemas = list_builtin_extraction_schemas()

    assert {schema.name for schema in schemas} >= {
        "customer_objections",
        "feature_requests",
        "meeting_decisions",
        "product_announcements",
        "sales_feedback",
    }
    assert all("gist/data/extraction" in schema.path for schema in schemas)


def test_structured_schemas_cli_outputs_text(capsys) -> None:
    exit_code = schemas_main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "product_announcements" in captured.out
    assert "product-announcements.schema.json" in captured.out


def test_structured_schemas_cli_outputs_json(capsys) -> None:
    exit_code = schemas_main(["--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert any(schema["name"] == "feature_requests" for schema in payload)


def test_structured_schemas_cli_outputs_presets(capsys) -> None:
    exit_code = schemas_main(["--presets"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "customer-objections: customer_objections" in captured.out


def test_suggests_extraction_preset_from_task() -> None:
    suggestion = suggest_extraction_preset(
        "find every time prospects complain about pricing"
    )

    assert suggestion.recommended_preset == "customer-objections"
    assert suggestion.schema_name == "customer_objections"
    assert "pricing" in suggestion.matched_terms


def test_structured_schemas_cli_suggests_preset(capsys) -> None:
    exit_code = schemas_main(
        ["--suggest", "find every time prospects complain about pricing"]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "recommended_preset=customer-objections" in captured.out
    assert "schema_name=customer_objections" in captured.out


def test_structured_schemas_cli_suggests_preset_json(capsys) -> None:
    exit_code = schemas_main(["--suggest", "capture new product launches", "--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["recommended_preset"] == "product-announcements"
    assert payload["schema_name"] == "product_announcements"


def test_resolves_builtin_extraction_schema_by_name() -> None:
    schema = resolve_extraction_schema(schema_name="product-announcements")

    assert schema.name == "product_announcements"
    assert "new feature" in schema.labels


def test_resolves_extraction_schema_by_preset() -> None:
    schema = resolve_extraction_schema(preset="customer-objections")

    assert schema.name == "customer_objections"
    assert "pricing objection" in schema.labels


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


def test_extract_from_compression_file_accepts_schema_name(tmp_path: Path) -> None:
    compression_path = tmp_path / "compression.json"
    compression = _compression(
        selected=[
            _item(
                "a-1",
                text="The customer has a pricing objection.",
                clip_start=30,
                clip_end=45,
            )
        ]
    )
    compression_path.write_text(
        json.dumps({"compression": compression.model_dump(mode="json")})
    )

    response = extract_from_compression_file(
        compression_path=compression_path,
        schema_name="customer_objections",
    )

    assert response.schema_name == "customer_objections"
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


def test_structured_extract_cli_accepts_schema_name(tmp_path: Path) -> None:
    compression_path = tmp_path / "compression.json"
    output_path = tmp_path / "extraction.json"
    compression = _compression(
        selected=[
            _item(
                "a-1",
                text="The customer has a pricing objection.",
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
            "--schema-name",
            "customer_objections",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text())
    assert payload["schema_name"] == "customer_objections"
    assert payload["items"][0]["label"] == "pricing objection"


def test_structured_extract_cli_accepts_preset(tmp_path: Path) -> None:
    compression_path = tmp_path / "compression.json"
    output_path = tmp_path / "extraction.json"
    compression = _compression(
        selected=[
            _item(
                "a-1",
                text="The customer has a pricing objection.",
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
            "--preset",
            "customer-objections",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text())
    assert payload["schema_name"] == "customer_objections"
    assert payload["items"][0]["label"] == "pricing objection"


def test_structured_extract_cli_writes_reports(tmp_path: Path) -> None:
    compression_path = tmp_path / "compression.json"
    schema_path = tmp_path / "schema.json"
    output_path = tmp_path / "extraction.json"
    markdown_path = tmp_path / "extraction.md"
    html_path = tmp_path / "extraction.html"
    csv_path = tmp_path / "extraction.csv"
    schema_path.write_text(
        json.dumps(
            {
                "name": "sales_feedback",
                "labels": ["pricing objection"],
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
            "--markdown-output",
            str(markdown_path),
            "--html-output",
            str(html_path),
            "--csv-output",
            str(csv_path),
        ]
    )

    assert exit_code == 0
    assert "pricing objection" in markdown_path.read_text()
    assert "<html" in html_path.read_text()
    rows = list(csv.DictReader(StringIO(csv_path.read_text())))
    assert rows[0]["label"] == "pricing objection"


def test_structured_report_renderers_include_items() -> None:
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
    schema = ExtractionSchema(
        name="sales_feedback",
        labels=["pricing objection"],
        fields=[ExtractionField(name="label")],
    )
    extraction = LocalStructuredExtractor().extract(schema, compression)

    markdown = render_structured_extraction_markdown(extraction)
    html = render_structured_extraction_html(extraction)
    csv_text = render_structured_extraction_csv(extraction)

    assert "pricing objection" in markdown
    assert "Gist Structured Extraction Report" in html
    assert "The buyer says pricing is too expensive." in html
    rows = list(csv.DictReader(StringIO(csv_text)))
    assert rows[0]["schema_name"] == "sales_feedback"
    assert rows[0]["label"] == "pricing objection"
    assert rows[0]["value_label"] == "pricing objection"


def test_structured_extract_cli_uses_subprocess_extractor(tmp_path: Path) -> None:
    compression_path = tmp_path / "compression.json"
    schema_path = tmp_path / "schema.json"
    output_path = tmp_path / "extraction.json"
    schema_path.write_text(json.dumps({"name": "sales_feedback"}))
    compression = _compression(
        selected=[_item("a-1", text="Pricing is too expensive.", clip_start=30, clip_end=45)]
    )
    compression_path.write_text(
        json.dumps({"compression": compression.model_dump(mode="json")})
    )
    command = (
        f"{sys.executable} -c "
        "\"import json,sys;"
        "json.load(sys.stdin);"
        "item={"
        "'label':'pricing objection',"
        "'description':'Pricing is too expensive.',"
        "'timestamp_start_seconds':30,"
        "'timestamp_end_seconds':45,"
        "'evidence_id':'a-1',"
        "'evidence_rank':1,"
        "'confidence':0.9,"
        "'support_text':'Pricing is too expensive.'"
        "};"
        "print(json.dumps({'items':[item]}))\""
    )

    exit_code = main(
        [
            "--compression",
            str(compression_path),
            "--schema",
            str(schema_path),
            "--output",
            str(output_path),
            "--extractor-command",
            command,
        ]
    )

    assert exit_code == 0
    assert json.loads(output_path.read_text())["items"][0]["label"] == "pricing objection"


def test_reference_structured_extractor_script(tmp_path: Path) -> None:
    compression_path = tmp_path / "compression.json"
    schema_path = tmp_path / "schema.json"
    output_path = tmp_path / "extraction.json"
    schema_path.write_text(
        json.dumps(
            {
                "name": "sales_feedback",
                "labels": ["pricing objection"],
            }
        )
    )
    compression = _compression(
        selected=[_item("a-1", text="Pricing is too expensive.", clip_start=30, clip_end=45)]
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
            "--extractor-command",
            f"{sys.executable} scripts/run_local_structured_extractor.py",
        ]
    )

    payload = json.loads(output_path.read_text())
    assert exit_code == 0
    assert payload["provider"] == "local-reference-structured-extractor"
    assert payload["items"][0]["label"] == "pricing objection"
    assert payload["items"][0]["values"]["sentiment"] == "negative"


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
