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
        "Answer the user query using only the transcript-backed evidence below.",
        "The clips are supporting video moments, but this text-only gateway cannot inspect pixels.",
        "Do not infer visual details from visual-only clips unless transcript text is provided.",
        f"Query: {compression.query}",
    ]
    lines.append("Evidence:")
    prompt_items = _transcript_backed_items(compression.selected)
    if not prompt_items:
        prompt_items = list(enumerate(compression.selected, start=1))
    for index, item in prompt_items:
        time_range = _time_range(item)
        transcript = _prompt_transcript(item)
        lines.append(
            f"{index}. {time_range} clip={_path_to_string(item.clip_path) or 'n/a'} "
            f"transcript={transcript!r}"
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
        "spatial_mask_path": _path_to_string(item.spatial_mask_path),
        "spatial_mask_preview_path": _path_to_string(item.spatial_mask_preview_path),
        "transcript": item.text,
        "segment_id": item.segment_id,
        "scene_start_seconds": item.scene_start_seconds,
        "scene_end_seconds": item.scene_end_seconds,
        "relevance_score": item.relevance_score,
        "mmr_score": item.mmr_score,
        "answer_support_score": item.answer_support_score,
        "query_support_score": item.query_support_score,
        "evidence_support_score": item.evidence_support_score,
        "audio_support_score": item.audio_support_score,
        "ocr_support_score": item.ocr_support_score,
        "visual_support_score": item.visual_support_score,
        "cross_modal_support_score": item.cross_modal_support_score,
        "support_label": item.support_label,
        "reason": item.reason,
    }


def _prompt_transcript(item: SelectedCandidate) -> str:
    text = " ".join(item.text.split())
    if not text:
        return "[no transcript text available for this clip]"
    if _is_visual_only_text(text):
        return "[visual-only clip; no transcript text available]"
    return text


def _transcript_backed_items(
    items: list[SelectedCandidate],
) -> list[tuple[int, SelectedCandidate]]:
    return [
        (index, item)
        for index, item in enumerate(items, start=1)
        if _has_transcript_text(item)
    ]


def _has_transcript_text(item: SelectedCandidate) -> bool:
    text = " ".join(item.text.split())
    return bool(text) and not _is_visual_only_text(text)


def _is_visual_only_text(text: str) -> bool:
    normalized = text.lower()
    return normalized.startswith("visual frame sampled at") or normalized.startswith(
        "on-screen text near"
    )


def _time_range(item: SelectedCandidate) -> str:
    if item.clip_start_seconds is not None and item.clip_end_seconds is not None:
        return f"{item.clip_start_seconds:.2f}s-{item.clip_end_seconds:.2f}s"
    return f"{item.timestamp_seconds:.2f}s"


def _path_to_string(path: Path | None) -> str | None:
    return str(path) if path is not None else None
