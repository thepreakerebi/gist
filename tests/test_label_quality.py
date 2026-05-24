from gist.gateway.structured import ExtractionSchema, StructuredExtractionResponse
from gist.label_quality import (
    evaluate_label_quality,
    render_label_quality_html,
    render_label_quality_markdown,
)


def test_label_quality_reports_no_items_as_error() -> None:
    extraction = StructuredExtractionResponse(
        schema_name="customer_objections",
        query="find pricing objections",
        item_type="customer_objection",
        items=[],
        prompt="prompt",
        provider="test",
    )
    schema = ExtractionSchema(name="customer_objections")

    report = evaluate_label_quality(extraction, schema, evidence_count=3)

    assert report.passed is False
    assert report.item_count == 0
    assert report.extraction_rate == 0
    assert report.warnings[0].code == "no_items"
    assert "No structured items" in render_label_quality_markdown(report)
    assert "Gist Label Quality Report" in render_label_quality_html(report)
