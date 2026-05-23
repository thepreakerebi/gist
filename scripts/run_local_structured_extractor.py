#!/usr/bin/env python3
import json
import re
import sys
from typing import Any


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        print(json.dumps(extract(payload)))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"run_local_structured_extractor.py: {exc}", file=sys.stderr)
        return 2
    return 0


def extract(payload: dict[str, Any]) -> dict[str, Any]:
    schema = payload.get("schema") or {}
    labels = [str(label) for label in schema.get("labels", [])]
    item_type = str(schema.get("item_type") or "event")
    items = []
    for evidence in payload.get("evidence", []):
        text = " ".join(str(evidence.get("text") or "").split())
        if not text:
            continue
        label, label_score = _best_label(labels, item_type, text)
        if labels and label_score == 0:
            continue
        items.append(_item_from_evidence(evidence, text, label, label_score))
    return {
        "provider": "local-reference-structured-extractor",
        "items": items,
    }


def _item_from_evidence(
    evidence: dict[str, Any],
    text: str,
    label: str,
    label_score: float,
) -> dict[str, Any]:
    timestamp = float(evidence.get("timestamp_seconds") or 0.0)
    start_seconds = _number(evidence.get("clip_start_seconds"), timestamp)
    end_seconds = _number(evidence.get("clip_end_seconds"), timestamp)
    return {
        "label": label,
        "description": text,
        "timestamp_start_seconds": min(start_seconds, end_seconds),
        "timestamp_end_seconds": max(start_seconds, end_seconds),
        "evidence_id": str(evidence.get("id") or ""),
        "evidence_rank": int(evidence.get("selection_rank") or 1),
        "confidence": max(0.0, min(label_score or 0.5, 1.0)),
        "support_text": text,
        "clip_path": evidence.get("clip_path"),
        "values": {
            "summary": text,
            "sentiment": _sentiment(text),
        },
    }


def _best_label(labels: list[str], item_type: str, text: str) -> tuple[str, float]:
    if not labels:
        return item_type, 1.0
    scored = [(_term_overlap(label, text), label) for label in labels]
    score, label = max(scored, key=lambda item: (item[0], item[1]))
    return label, score


def _sentiment(text: str) -> str:
    terms = _terms(text)
    if terms & {"bad", "expensive", "hate", "negative", "problem", "issue", "hard"}:
        return "negative"
    if terms & {"good", "great", "love", "strong", "positive", "works", "useful"}:
        return "positive"
    return "neutral"


def _term_overlap(left: str, right: str) -> float:
    left_terms = _terms(left)
    right_terms = _terms(right)
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms)


def _terms(value: str) -> set[str]:
    stopwords = {"a", "an", "and", "for", "in", "is", "of", "on", "or", "the", "to", "with"}
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if token not in stopwords
    }


def _number(value: Any, default: float) -> float:
    if value is None:
        return default
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
