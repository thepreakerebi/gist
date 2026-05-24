import argparse
import json
import re
import shlex
import subprocess
from importlib import resources
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from gist.core.schemas import CompressionResponse, SelectedCandidate
from gist.gateway.context import render_evidence_context
from gist.reports.structured import (
    render_structured_extraction_csv,
    render_structured_extraction_html,
    render_structured_extraction_markdown,
)


STRUCTURED_EXTRACTION_VERSION = "gist.structured-extraction.v1"
SCHEMA_FILE_PATTERN = "*.schema.json"
EXTRACTION_PRESETS = {
    "customer-objections": "customer_objections",
    "feature-requests": "feature_requests",
    "meeting-decisions": "meeting_decisions",
    "product-announcements": "product_announcements",
    "sales-feedback": "sales_feedback",
}
PRESET_SUGGESTION_RULES = {
    "customer-objections": [
        "objection",
        "complain",
        "complaint",
        "concern",
        "blocker",
        "expensive",
        "pricing",
        "price",
        "too much",
        "security",
        "trust",
        "implementation",
    ],
    "feature-requests": [
        "feature request",
        "requested feature",
        "ask for",
        "asked for",
        "need",
        "wish",
        "missing feature",
        "integration request",
        "workflow",
        "improvement",
        "automation",
        "reporting",
    ],
    "meeting-decisions": [
        "decision",
        "decide",
        "action item",
        "follow up",
        "follow-up",
        "owner",
        "deadline",
        "due date",
        "open question",
        "risk",
        "assigned",
    ],
    "product-announcements": [
        "announcement",
        "announce",
        "launch",
        "released",
        "new product",
        "new feature",
        "availability",
        "roadmap",
        "integration",
        "keynote",
        "demo",
    ],
    "sales-feedback": [
        "sales",
        "feedback",
        "demo feedback",
        "product mention",
        "positive reaction",
        "negative reaction",
        "buyer",
        "prospect",
        "customer reaction",
    ],
}


class ExtractionField(BaseModel):
    name: str
    description: str = ""
    required: bool = False


class ExtractionSchema(BaseModel):
    name: str
    description: str = ""
    item_type: str = "event"
    labels: list[str] = Field(default_factory=list)
    fields: list[ExtractionField] = Field(default_factory=list)

    @classmethod
    def from_file(cls, path: Path) -> "ExtractionSchema":
        return cls.model_validate(json.loads(path.read_text()))

    @classmethod
    def from_json_text(cls, text: str) -> "ExtractionSchema":
        return cls.model_validate(json.loads(text))


class ExtractedItem(BaseModel):
    label: str
    description: str
    timestamp_start_seconds: float
    timestamp_end_seconds: float
    evidence_id: str
    evidence_rank: int
    confidence: float
    support_text: str
    clip_path: str | None = None
    values: dict[str, Any] = Field(default_factory=dict)


class StructuredExtractionResponse(BaseModel):
    schema_version: str = STRUCTURED_EXTRACTION_VERSION
    schema_name: str
    query: str
    item_type: str
    items: list[ExtractedItem]
    prompt: str
    provider: str

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n")


class StructuredExtractionError(RuntimeError):
    """Raised when a structured extraction adapter returns invalid output."""


class BuiltinExtractionSchema(BaseModel):
    name: str
    description: str
    item_type: str
    labels: list[str]
    path: str


class ExtractionPresetSuggestion(BaseModel):
    recommended_preset: str
    schema_name: str
    score: int
    matched_terms: list[str]
    reason: str


class LocalStructuredExtractor:
    provider = "local-structured-extractor"

    def extract(
        self,
        schema: ExtractionSchema,
        compression: CompressionResponse,
    ) -> StructuredExtractionResponse:
        prompt = build_structured_extraction_prompt(schema, compression)
        extracted_items = [
            item
            for evidence in compression.selected
            if (item := _extract_item(schema, evidence)) is not None
        ]
        return StructuredExtractionResponse(
            schema_name=schema.name,
            query=compression.query,
            item_type=schema.item_type,
            items=_dedupe_extracted_items(extracted_items),
            prompt=prompt,
            provider=self.provider,
        )


class SubprocessStructuredExtractor:
    provider = "subprocess-structured-extractor"

    def __init__(
        self,
        command: list[str],
        timeout_seconds: float = 120.0,
        provider: str | None = None,
    ) -> None:
        if not command:
            raise ValueError("command must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        self.command = command
        self.timeout_seconds = timeout_seconds
        self.provider = provider or self.provider

    def extract(
        self,
        schema: ExtractionSchema,
        compression: CompressionResponse,
    ) -> StructuredExtractionResponse:
        prompt = build_structured_extraction_prompt(schema, compression)
        payload = build_structured_extraction_payload(schema, compression, prompt)
        completed = self._run(payload)
        return _parse_subprocess_extraction_stdout(
            stdout=completed.stdout,
            schema=schema,
            compression=compression,
            prompt=prompt,
            fallback_provider=self.provider,
        )

    def _run(self, payload: dict[str, Any]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                self.command,
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise StructuredExtractionError(
                f"structured extraction command timed out after {self.timeout_seconds:.1f}s"
            ) from exc
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else ""
            raise StructuredExtractionError(
                "structured extraction command failed with exit code "
                f"{exc.returncode}: {stderr}"
            ) from exc


def build_structured_extraction_payload(
    schema: ExtractionSchema,
    compression: CompressionResponse,
    prompt: str | None = None,
) -> dict[str, Any]:
    return {
        "type": "gist.structured_extraction",
        "schema": schema.model_dump(mode="json"),
        "query": compression.query,
        "prompt": prompt or build_structured_extraction_prompt(schema, compression),
        "context": render_evidence_context(compression),
        "evidence": [
            {
                "id": item.id,
                "modality": item.modality.value,
                "timestamp_seconds": item.timestamp_seconds,
                "clip_start_seconds": item.clip_start_seconds,
                "clip_end_seconds": item.clip_end_seconds,
                "text": item.text,
                "clip_path": str(item.clip_path) if item.clip_path is not None else None,
                "asset_path": str(item.asset_path) if item.asset_path is not None else None,
                "selection_rank": item.selection_rank,
                "relevance_score": item.relevance_score,
                "support_label": item.support_label,
            }
            for item in compression.selected
        ],
        "compression": compression.model_dump(mode="json"),
    }


def load_compression_response(path: Path) -> CompressionResponse:
    payload = json.loads(path.read_text())
    compression_payload = payload.get("compression", payload)
    return CompressionResponse.model_validate(compression_payload)


def extract_from_compression_file(
    compression_path: Path,
    schema_path: Path | None = None,
    schema_name: str | None = None,
    preset: str | None = None,
    extractor: LocalStructuredExtractor | SubprocessStructuredExtractor | None = None,
) -> StructuredExtractionResponse:
    resolved_extractor = extractor or LocalStructuredExtractor()
    return resolved_extractor.extract(
        schema=resolve_extraction_schema(
            schema_path=schema_path,
            schema_name=schema_name,
            preset=preset,
        ),
        compression=load_compression_response(compression_path),
    )


def resolve_extraction_schema(
    schema_path: Path | None = None,
    schema_name: str | None = None,
    preset: str | None = None,
) -> ExtractionSchema:
    selectors = [
        value is not None
        for value in (schema_path, schema_name, preset)
    ]
    if sum(selectors) > 1:
        raise ValueError("Use only one of schema_path, schema_name, or preset")
    if schema_path is not None:
        return ExtractionSchema.from_file(schema_path)
    if preset is not None:
        schema_name = schema_name_for_extraction_preset(preset)
    if schema_name is None:
        raise ValueError("schema_path, schema_name, or preset is required")

    normalized_name = _normalize_schema_name(schema_name)
    schema = _resolve_packaged_schema_by_name(normalized_name)
    if schema is not None:
        return schema
    schema = _resolve_schema_from_dir(_default_schema_dir(), normalized_name)
    if schema is not None:
        return schema
    available = ", ".join(schema.name for schema in list_builtin_extraction_schemas())
    raise ValueError(
        f"unknown extraction schema name `{schema_name}`. "
        f"Available schemas: {available or 'none'}"
    )


def schema_name_for_extraction_preset(preset: str) -> str:
    normalized_preset = _normalize_preset_name(preset)
    if normalized_preset in EXTRACTION_PRESETS:
        return EXTRACTION_PRESETS[normalized_preset]
    available = ", ".join(EXTRACTION_PRESETS)
    raise ValueError(
        f"unknown extraction preset `{preset}`. Available presets: {available}"
    )


def suggest_extraction_preset(task: str) -> ExtractionPresetSuggestion:
    normalized_task = _normalize_task_text(task)
    scores = [
        (
            _preset_match_terms(normalized_task, keywords),
            preset,
        )
        for preset, keywords in PRESET_SUGGESTION_RULES.items()
    ]
    matched_terms, preset = max(
        scores,
        key=lambda item: (len(item[0]), _preset_priority(item[1])),
    )
    if not matched_terms:
        preset = "sales-feedback"
    schema_name = schema_name_for_extraction_preset(preset)
    reason = _preset_suggestion_reason(preset, matched_terms)
    return ExtractionPresetSuggestion(
        recommended_preset=preset,
        schema_name=schema_name,
        score=len(matched_terms),
        matched_terms=matched_terms,
        reason=reason,
    )


def list_builtin_extraction_schemas(
    schema_dir: Path | None = None,
) -> list[BuiltinExtractionSchema]:
    if schema_dir is not None:
        return _list_builtin_schemas_from_dir(schema_dir)
    packaged = _list_packaged_builtin_schemas()
    if packaged:
        return packaged
    return _list_builtin_schemas_from_dir(_default_schema_dir())


def _list_builtin_schemas_from_dir(schema_dir: Path) -> list[BuiltinExtractionSchema]:
    if not schema_dir.exists():
        return []
    schemas = []
    for path in sorted(schema_dir.glob(SCHEMA_FILE_PATTERN)):
        schema = ExtractionSchema.from_file(path)
        schemas.append(
            BuiltinExtractionSchema(
                name=schema.name,
                description=schema.description,
                item_type=schema.item_type,
                labels=schema.labels,
                path=str(path),
            )
        )
    return schemas


def _list_packaged_builtin_schemas() -> list[BuiltinExtractionSchema]:
    try:
        schema_root = resources.files("gist").joinpath("data", "extraction")
    except (ModuleNotFoundError, AttributeError):
        return []
    if not schema_root.is_dir():
        return []

    schemas = []
    for resource in sorted(schema_root.iterdir(), key=lambda item: item.name):
        if not resource.name.endswith(".schema.json"):
            continue
        schema = ExtractionSchema.from_json_text(resource.read_text())
        schemas.append(
            BuiltinExtractionSchema(
                name=schema.name,
                description=schema.description,
                item_type=schema.item_type,
                labels=schema.labels,
                path=str(resource),
            )
        )
    return schemas


def _resolve_packaged_schema_by_name(normalized_name: str) -> ExtractionSchema | None:
    try:
        schema_root = resources.files("gist").joinpath("data", "extraction")
    except (ModuleNotFoundError, AttributeError):
        return None
    if not schema_root.is_dir():
        return None
    for resource in sorted(schema_root.iterdir(), key=lambda item: item.name):
        if not resource.name.endswith(".schema.json"):
            continue
        schema = ExtractionSchema.from_json_text(resource.read_text())
        if _normalize_schema_name(schema.name) == normalized_name:
            return schema
    return None


def _resolve_schema_from_dir(
    schema_dir: Path,
    normalized_name: str,
) -> ExtractionSchema | None:
    if not schema_dir.exists():
        return None
    for path in sorted(schema_dir.glob(SCHEMA_FILE_PATTERN)):
        schema = ExtractionSchema.from_file(path)
        if _normalize_schema_name(schema.name) == normalized_name:
            return schema
    return None


def schemas_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List built-in Gist extraction schemas.")
    parser.add_argument("--json", action="store_true", help="Print schemas as JSON.")
    parser.add_argument(
        "--presets",
        action="store_true",
        help="List extraction presets instead of schema files.",
    )
    parser.add_argument(
        "--schema-dir",
        type=Path,
        help="Override the schema directory. Mainly useful for tests and custom builds.",
    )
    parser.add_argument(
        "--suggest",
        help="Suggest an extraction preset for a natural-language labeling task.",
    )
    args = parser.parse_args(argv)

    if args.suggest:
        suggestion = suggest_extraction_preset(args.suggest)
        if args.json:
            print(suggestion.model_dump_json(indent=2))
            return 0
        print(f"recommended_preset={suggestion.recommended_preset}")
        print(f"schema_name={suggestion.schema_name}")
        print(f"reason={suggestion.reason}")
        if suggestion.matched_terms:
            print(f"matched_terms={', '.join(suggestion.matched_terms)}")
        return 0

    if args.presets:
        if args.json:
            payload = [
                {"preset": preset, "schema_name": schema_name}
                for preset, schema_name in EXTRACTION_PRESETS.items()
            ]
            print(json.dumps(payload, indent=2))
            return 0
        for preset, schema_name in EXTRACTION_PRESETS.items():
            print(f"{preset}: {schema_name}")
        return 0

    schemas = list_builtin_extraction_schemas(args.schema_dir)
    if args.json:
        payload = [schema.model_dump(mode="json") for schema in schemas]
        print(json.dumps(payload, indent=2))
        return 0

    if not schemas:
        print("No built-in extraction schemas found.")
        return 0

    for schema in schemas:
        labels = ", ".join(schema.labels)
        print(f"{schema.name}")
        print(f"  item_type: {schema.item_type}")
        print(f"  labels: {labels}")
        print(f"  path: {schema.path}")
        if schema.description:
            print(f"  description: {schema.description}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract timestamped structured records from a Gist compression file."
    )
    parser.add_argument("--compression", required=True, type=Path)
    schema_group = parser.add_mutually_exclusive_group(required=True)
    schema_group.add_argument("--schema", type=Path)
    schema_group.add_argument(
        "--schema-name",
        help="Built-in schema name from `gist-structured-schemas`.",
    )
    schema_group.add_argument(
        "--preset",
        choices=sorted(EXTRACTION_PRESETS),
        help="Extraction preset alias for a built-in schema.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--html-output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument(
        "--extractor-command",
        help=(
            "External structured extractor command. Gist sends JSON to stdin and "
            "expects JSON stdout with an `items` array."
        ),
    )
    parser.add_argument("--extractor-timeout", type=float, default=120.0)
    args = parser.parse_args(argv)

    extractor = (
        SubprocessStructuredExtractor(
            command=shlex.split(args.extractor_command),
            timeout_seconds=args.extractor_timeout,
        )
        if args.extractor_command
        else None
    )
    extraction = extract_from_compression_file(
        compression_path=args.compression,
        schema_path=args.schema,
        schema_name=args.schema_name,
        preset=args.preset,
        extractor=extractor,
    )
    extraction.write_json(args.output)
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_structured_extraction_markdown(extraction))
    if args.html_output is not None:
        args.html_output.parent.mkdir(parents=True, exist_ok=True)
        args.html_output.write_text(render_structured_extraction_html(extraction))
    if args.csv_output is not None:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        args.csv_output.write_text(render_structured_extraction_csv(extraction))
    print(f"items={len(extraction.items)}")
    print(f"provider={extraction.provider}")
    print(f"output={args.output}")
    if args.markdown_output is not None:
        print(f"markdown={args.markdown_output}")
    if args.html_output is not None:
        print(f"html={args.html_output}")
    if args.csv_output is not None:
        print(f"csv={args.csv_output}")
    return 0


def _default_schema_dir() -> Path:
    current = Path(__file__).resolve()
    for parent in [Path.cwd(), *current.parents]:
        candidate = parent / "data" / "extraction"
        if candidate.exists():
            return candidate
    return Path.cwd() / "data" / "extraction"


def _normalize_schema_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _normalize_preset_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _normalize_task_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def _preset_match_terms(task: str, keywords: list[str]) -> list[str]:
    return sorted(
        {
            keyword
            for keyword in keywords
            if re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", task)
        }
    )


def _preset_priority(preset: str) -> int:
    priority = {
        "customer-objections": 5,
        "feature-requests": 4,
        "meeting-decisions": 3,
        "product-announcements": 2,
        "sales-feedback": 1,
    }
    return priority.get(preset, 0)


def _preset_suggestion_reason(preset: str, matched_terms: list[str]) -> str:
    if matched_terms:
        return (
            f"matched task terms ({', '.join(matched_terms)}) to "
            f"the {preset} extraction preset"
        )
    return "no strong preset-specific terms found; sales-feedback is the broadest default"


def build_structured_extraction_prompt(
    schema: ExtractionSchema,
    compression: CompressionResponse,
) -> str:
    fields = "\n".join(
        f"- {field.name}: {field.description or 'No description'}"
        f"{' (required)' if field.required else ''}"
        for field in schema.fields
    )
    labels = ", ".join(schema.labels) if schema.labels else "infer from evidence"
    return (
        "Extract structured items from the compressed Gist evidence only. "
        "Return strict JSON with an `items` array. Each item must include label, "
        "description, timestamp_start_seconds, timestamp_end_seconds, evidence_id, "
        "confidence, support_text, clip_path, and values.\n\n"
        f"Schema: {schema.name}\n"
        f"Description: {schema.description}\n"
        f"Item type: {schema.item_type}\n"
        f"Allowed labels: {labels}\n"
        f"Fields:\n{fields or '- none'}\n\n"
        f"{render_evidence_context(compression)}"
    )


def _parse_subprocess_extraction_stdout(
    stdout: str,
    schema: ExtractionSchema,
    compression: CompressionResponse,
    prompt: str,
    fallback_provider: str,
) -> StructuredExtractionResponse:
    stripped = stdout.strip()
    if not stripped:
        raise StructuredExtractionError("structured extraction command returned empty stdout")
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise StructuredExtractionError(
            "structured extraction command stdout must be valid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise StructuredExtractionError(
            "structured extraction command JSON stdout must be an object"
        )
    items = payload.get("items")
    if not isinstance(items, list):
        raise StructuredExtractionError(
            "structured extraction command JSON stdout must include an items array"
        )
    provider = payload.get("provider")
    return StructuredExtractionResponse(
        schema_name=schema.name,
        query=compression.query,
        item_type=schema.item_type,
        items=[ExtractedItem.model_validate(item) for item in items],
        prompt=payload.get("prompt") if isinstance(payload.get("prompt"), str) else prompt,
        provider=provider if isinstance(provider, str) and provider.strip() else fallback_provider,
    )


def _extract_item(
    schema: ExtractionSchema,
    evidence: SelectedCandidate,
) -> ExtractedItem | None:
    text = " ".join(evidence.text.split())
    if not text:
        return None
    label, label_score = _best_label(schema, text)
    if schema.labels and label_score < _minimum_label_score(schema):
        return None
    if not _passes_schema_trigger(schema, text, label, label_score):
        return None

    start_seconds = evidence.clip_start_seconds
    end_seconds = evidence.clip_end_seconds
    if start_seconds is None or end_seconds is None:
        start_seconds = evidence.scene_start_seconds or evidence.timestamp_seconds
        end_seconds = evidence.scene_end_seconds or evidence.timestamp_seconds

    return ExtractedItem(
        label=label,
        description=_summarize_text(text),
        timestamp_start_seconds=min(start_seconds, end_seconds),
        timestamp_end_seconds=max(start_seconds, end_seconds),
        evidence_id=evidence.id,
        evidence_rank=evidence.selection_rank,
        confidence=_confidence(evidence, label_score),
        support_text=text,
        clip_path=str(evidence.clip_path) if evidence.clip_path is not None else None,
        values=_field_values(schema, text, label),
    )


def _best_label(schema: ExtractionSchema, text: str) -> tuple[str, float]:
    if not schema.labels:
        return schema.item_type, 1.0
    scored = [(_label_match_score(schema, label, text), label) for label in schema.labels]
    score, label = max(scored, key=lambda item: (item[0], item[1]))
    return label, score


def _field_values(schema: ExtractionSchema, text: str, label: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for field in schema.fields:
        normalized = field.name.lower()
        if normalized == "label":
            values[field.name] = label
        elif normalized in {"description", "summary"}:
            values[field.name] = _summarize_text(text)
        elif normalized in {"sentiment", "reaction"}:
            values[field.name] = _sentiment(text)
        elif normalized == "severity":
            values[field.name] = _severity(text)
        elif normalized in {"suggested_response", "next_step"}:
            values[field.name] = _suggested_response(text, label)
        elif normalized in {"requester", "speaker", "owner"}:
            values[field.name] = _speaker_or_requester(text)
        elif normalized in {"priority_signal", "impact"}:
            values[field.name] = _priority_signal(text)
        elif normalized == "pain_point":
            values[field.name] = _pain_point(text)
        else:
            values[field.name] = _best_phrase_for_field(field, text)
    return values


def _best_phrase_for_field(field: ExtractionField, text: str) -> str | None:
    field_terms = _terms(f"{field.name} {field.description}")
    sentences = [sentence.strip() for sentence in re.split(r"[.!?]+", text) if sentence.strip()]
    if not sentences:
        return text if _term_overlap(" ".join(field_terms), text) > 0 else None
    scored = [(_term_overlap(" ".join(field_terms), sentence), sentence) for sentence in sentences]
    score, sentence = max(scored, key=lambda item: item[0])
    return sentence if score > 0 else None


def _dedupe_extracted_items(items: list[ExtractedItem]) -> list[ExtractedItem]:
    deduped: list[ExtractedItem] = []
    for item in sorted(
        items,
        key=lambda candidate: (
            candidate.timestamp_start_seconds,
            candidate.evidence_rank,
            -candidate.confidence,
        ),
    ):
        duplicate_index = _find_duplicate_item(deduped, item)
        if duplicate_index is None:
            deduped.append(item)
            continue
        if item.confidence > deduped[duplicate_index].confidence:
            deduped[duplicate_index] = item
    return sorted(deduped, key=lambda item: item.evidence_rank)


def _find_duplicate_item(items: list[ExtractedItem], candidate: ExtractedItem) -> int | None:
    for index, existing in enumerate(items):
        if existing.label != candidate.label:
            continue
        if _time_overlap_ratio(existing, candidate) >= 0.5:
            return index
        if _jaccard(_terms(existing.support_text), _terms(candidate.support_text)) >= 0.72:
            return index
    return None


def _time_overlap_ratio(left: ExtractedItem, right: ExtractedItem) -> float:
    start = max(left.timestamp_start_seconds, right.timestamp_start_seconds)
    end = min(left.timestamp_end_seconds, right.timestamp_end_seconds)
    overlap = max(0.0, end - start)
    shortest = min(
        left.timestamp_end_seconds - left.timestamp_start_seconds,
        right.timestamp_end_seconds - right.timestamp_start_seconds,
    )
    return overlap / shortest if shortest > 0 else 0.0


def _label_match_score(schema: ExtractionSchema, label: str, text: str) -> float:
    base_score = _term_overlap(label, text)
    keyword_score = _keyword_overlap(_label_keywords(schema.name, label), text)
    return max(base_score, keyword_score)


def _label_keywords(schema_name: str, label: str) -> set[str]:
    normalized_schema = _normalize_schema_name(schema_name)
    normalized_label = label.lower()
    keywords: dict[tuple[str, str], set[str]] = {
        ("customer_objections", "pricing objection"): {
            "budget",
            "cost",
            "expensive",
            "price",
            "pricing",
            "too much",
        },
        ("customer_objections", "security concern"): {
            "compliance",
            "privacy",
            "risk",
            "secure",
            "security",
        },
        ("customer_objections", "implementation concern"): {
            "adoption",
            "deploy",
            "implementation",
            "migration",
            "onboarding",
            "setup",
        },
        ("customer_objections", "missing feature"): {
            "can't",
            "cannot",
            "doesn't support",
            "lack",
            "missing",
            "need",
        },
        ("customer_objections", "integration concern"): {
            "api",
            "integrate",
            "integration",
            "sync",
            "workflow",
        },
        ("customer_objections", "trust concern"): {
            "accuracy",
            "confidence",
            "reliability",
            "trust",
            "wrong",
        },
        ("customer_objections", "timing objection"): {
            "deadline",
            "later",
            "not now",
            "quarter",
            "timing",
        },
        ("feature_requests", "new capability"): {
            "add",
            "capability",
            "feature",
            "need",
            "support",
            "want",
        },
        ("feature_requests", "workflow improvement"): {
            "better",
            "improve",
            "improvement",
            "process",
            "workflow",
        },
        ("feature_requests", "integration request"): {
            "api",
            "connect",
            "integration",
            "sync",
        },
        ("feature_requests", "automation request"): {
            "agent",
            "automate",
            "automation",
            "automatically",
            "workflow",
        },
        ("feature_requests", "reporting request"): {
            "analytics",
            "dashboard",
            "metrics",
            "report",
            "reporting",
        },
        ("feature_requests", "usability request"): {
            "confusing",
            "easy",
            "simpler",
            "ui",
            "usability",
            "ux",
        },
        ("feature_requests", "performance request"): {
            "fast",
            "faster",
            "latency",
            "performance",
            "scale",
            "slow",
            "speed",
        },
    }
    return keywords.get((normalized_schema, normalized_label), set()) | _terms(label)


def _minimum_label_score(schema: ExtractionSchema) -> float:
    normalized_schema = _normalize_schema_name(schema.name)
    if normalized_schema in {"customer_objections", "feature_requests"}:
        return 0.2
    return 0.01


def _passes_schema_trigger(
    schema: ExtractionSchema,
    text: str,
    label: str,
    label_score: float,
) -> bool:
    normalized_schema = _normalize_schema_name(schema.name)
    text_terms = _terms(text)
    if normalized_schema == "feature_requests":
        request_terms = {
            "add",
            "ask",
            "asked",
            "better",
            "build",
            "could",
            "improve",
            "need",
            "request",
            "should",
            "want",
            "wish",
        }
        return bool(text_terms & request_terms) or label_score >= 0.5
    if normalized_schema == "customer_objections":
        objection_terms = {
            "block",
            "blocked",
            "can't",
            "cannot",
            "concern",
            "expensive",
            "hard",
            "issue",
            "objection",
            "problem",
            "risk",
            "worry",
            "wrong",
        }
        return bool(text_terms & objection_terms) or label_score >= 0.5
    return bool(label) or label_score > 0


def _summarize_text(text: str, max_chars: int = 260) -> str:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]
    summary = sentences[0] if sentences else text
    if len(summary) <= max_chars:
        return summary
    return summary[: max_chars - 3].rstrip() + "..."


def _severity(text: str) -> str | None:
    terms = _terms(text)
    if terms & {"blocked", "blocker", "critical", "dealbreaker", "must", "risk"}:
        return "high"
    if terms & {"concern", "expensive", "hard", "issue", "problem", "slow"}:
        return "medium"
    if terms & {"maybe", "minor", "later"}:
        return "low"
    return None


def _suggested_response(text: str, label: str) -> str | None:
    normalized_label = label.lower()
    if "pricing" in normalized_label:
        return "Clarify budget impact, ROI, and pricing flexibility."
    if "security" in normalized_label or "trust" in normalized_label:
        return "Provide security, privacy, reliability, and accuracy evidence."
    if "implementation" in normalized_label or "integration" in normalized_label:
        return "Follow up with implementation scope, integration plan, and owner."
    if "timing" in normalized_label:
        return "Clarify timeline, urgency, and next decision point."
    if _terms(text) & {"need", "want", "wish", "should"}:
        return "Capture the requested workflow and confirm priority with the requester."
    return None


def _speaker_or_requester(text: str) -> str | None:
    lowered = text.lower()
    patterns = [
        r"\b(?:customer|buyer|prospect|user|founder|builder|team)\b",
        r"\b(?:i|we)\s+(?:need|want|wish|asked|care|would)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            return match.group(0)
    return None


def _priority_signal(text: str) -> str | None:
    priority_terms = {
        "blocker",
        "critical",
        "deadline",
        "expensive",
        "must",
        "need",
        "pain",
        "risk",
        "want",
    }
    return _best_sentence_with_terms(text, priority_terms)


def _pain_point(text: str) -> str | None:
    pain_terms = {
        "can't",
        "cannot",
        "expensive",
        "friction",
        "hard",
        "issue",
        "pain",
        "problem",
        "slow",
        "too much",
    }
    return _best_sentence_with_terms(text, pain_terms)


def _best_sentence_with_terms(text: str, terms: set[str]) -> str | None:
    sentences = [sentence.strip() for sentence in re.split(r"[.!?]+", text) if sentence.strip()]
    if not sentences:
        return text if _keyword_overlap(terms, text) > 0 else None
    scored = [(_keyword_overlap(terms, sentence), sentence) for sentence in sentences]
    score, sentence = max(scored, key=lambda item: item[0])
    return _summarize_text(sentence, max_chars=180) if score > 0 else None


def _sentiment(text: str) -> str:
    terms = _terms(text)
    positive = {"good", "great", "love", "strong", "positive", "works", "useful"}
    negative = {"bad", "expensive", "hate", "negative", "problem", "issue", "hard"}
    if terms & negative:
        return "negative"
    if terms & positive:
        return "positive"
    return "neutral"


def _confidence(evidence: SelectedCandidate, label_score: float) -> float:
    support = evidence.evidence_support_score or evidence.query_support_score or 0.0
    score = max(evidence.normalized_score, evidence.relevance_score, support, label_score)
    return max(0.0, min(score, 1.0))


def _term_overlap(left: str, right: str) -> float:
    left_terms = _terms(left)
    right_terms = _terms(right)
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms)


def _keyword_overlap(keywords: set[str], text: str) -> float:
    if not keywords:
        return 0.0
    text_terms = _terms(text)
    normalized_text = text.lower()
    matches = 0
    for keyword in keywords:
        if " " in keyword:
            if keyword in normalized_text:
                matches += 1
        elif keyword in text_terms:
            matches += 1
    return matches / len(keywords)


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _terms(value: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "for",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if token not in stopwords
    }


if __name__ == "__main__":
    raise SystemExit(main())
