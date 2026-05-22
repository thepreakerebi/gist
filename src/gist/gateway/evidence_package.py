from pathlib import Path
from typing import Any

from gist.core.schemas import CompressionResponse, SelectedCandidate
from gist.media.models import IngestedVideo


EVIDENCE_PACKAGE_VERSION = "gist.evidence-package.v1"


def build_evidence_package(
    ingestion: IngestedVideo,
    compression: CompressionResponse,
) -> dict[str, Any]:
    return {
        "schema": EVIDENCE_PACKAGE_VERSION,
        "video": {
            "id": ingestion.video_id,
            "source_path": str(ingestion.source_path),
            "duration_seconds": ingestion.metadata.duration_seconds,
            "width": ingestion.metadata.width,
            "height": ingestion.metadata.height,
            "frame_rate": ingestion.metadata.frame_rate,
            "has_audio": ingestion.metadata.has_audio,
        },
        "query": compression.query,
        "answer_hint": compression.answer,
        "answer_provider": compression.answer_provider,
        "query_intent": str(compression.query_intent) if compression.query_intent else None,
        "routing_reason": compression.routing_reason,
        "compression": compression.metrics.model_dump(mode="json"),
        "evidence": [_evidence_item(item) for item in compression.selected],
        "prompt": build_evidence_prompt(compression),
    }


def build_evidence_prompt(compression: CompressionResponse) -> str:
    lines = [
        "Answer the user query using only the provided video evidence clips and transcripts.",
        f"Query: {compression.query}",
    ]
    if compression.answer:
        lines.append(f"Initial answer hint: {compression.answer}")
    lines.append("Evidence:")
    for index, item in enumerate(compression.selected, start=1):
        time_range = _time_range(item)
        lines.append(
            f"{index}. {time_range} clip={_path_to_string(item.clip_path) or 'n/a'} "
            f"transcript={item.text!r}"
        )
    lines.append(
        "Return a concise answer and cite the evidence numbers that support it."
    )
    return "\n".join(lines)


def _evidence_item(item: SelectedCandidate) -> dict[str, Any]:
    return {
        "id": item.id,
        "modality": item.modality.value,
        "timestamp_seconds": item.timestamp_seconds,
        "clip_start_seconds": item.clip_start_seconds,
        "clip_end_seconds": item.clip_end_seconds,
        "clip_path": _path_to_string(item.clip_path),
        "asset_path": _path_to_string(item.asset_path),
        "transcript": item.text,
        "segment_id": item.segment_id,
        "scene_start_seconds": item.scene_start_seconds,
        "scene_end_seconds": item.scene_end_seconds,
        "relevance_score": item.relevance_score,
        "mmr_score": item.mmr_score,
        "reason": item.reason,
    }


def _time_range(item: SelectedCandidate) -> str:
    if item.clip_start_seconds is not None and item.clip_end_seconds is not None:
        return f"{item.clip_start_seconds:.2f}s-{item.clip_end_seconds:.2f}s"
    return f"{item.timestamp_seconds:.2f}s"


def _path_to_string(path: Path | None) -> str | None:
    return str(path) if path is not None else None
