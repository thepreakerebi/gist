import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from gist.core.schemas import CompressionResponse, SelectedCandidate
from gist.gateway.context import render_evidence_context


STRUCTURED_EXTRACTION_VERSION = "gist.structured-extraction.v1"


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


class LocalStructuredExtractor:
    provider = "local-structured-extractor"

    def extract(
        self,
        schema: ExtractionSchema,
        compression: CompressionResponse,
    ) -> StructuredExtractionResponse:
        prompt = build_structured_extraction_prompt(schema, compression)
        items = [
            item
            for evidence in compression.selected
            if (item := _extract_item(schema, evidence)) is not None
        ]
        return StructuredExtractionResponse(
            schema_name=schema.name,
            query=compression.query,
            item_type=schema.item_type,
            items=items,
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
    schema_path: Path,
    extractor: LocalStructuredExtractor | SubprocessStructuredExtractor | None = None,
) -> StructuredExtractionResponse:
    resolved_extractor = extractor or LocalStructuredExtractor()
    return resolved_extractor.extract(
        schema=ExtractionSchema.from_file(schema_path),
        compression=load_compression_response(compression_path),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract timestamped structured records from a Gist compression file."
    )
    parser.add_argument("--compression", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
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
        extractor=extractor,
    )
    extraction.write_json(args.output)
    print(f"items={len(extraction.items)}")
    print(f"provider={extraction.provider}")
    print(f"output={args.output}")
    return 0


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
    if schema.labels and label_score == 0:
        return None

    start_seconds = evidence.clip_start_seconds
    end_seconds = evidence.clip_end_seconds
    if start_seconds is None or end_seconds is None:
        start_seconds = evidence.scene_start_seconds or evidence.timestamp_seconds
        end_seconds = evidence.scene_end_seconds or evidence.timestamp_seconds

    return ExtractedItem(
        label=label,
        description=text,
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
    scored = [(_term_overlap(label, text), label) for label in schema.labels]
    score, label = max(scored, key=lambda item: (item[0], item[1]))
    return label, score


def _field_values(schema: ExtractionSchema, text: str, label: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for field in schema.fields:
        normalized = field.name.lower()
        if normalized == "label":
            values[field.name] = label
        elif normalized in {"description", "summary"}:
            values[field.name] = text
        elif normalized in {"sentiment", "reaction"}:
            values[field.name] = _sentiment(text)
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
