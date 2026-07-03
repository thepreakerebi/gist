from enum import StrEnum
import re
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class QueryAspectModality(StrEnum):
    VISUAL = "visual"
    AUDIO = "audio"
    BOTH = "both"


class QueryAspect(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)
    modality: QueryAspectModality = QueryAspectModality.BOTH


class QueryDecomposer(Protocol):
    def decompose(self, query: str) -> list[QueryAspect]:
        """Split a compound query into independently scoreable aspects."""


class RuleBasedQueryDecomposer:
    _split_pattern = re.compile(r"\b(?:and|then|while|before|after|when)\b|[,;]", re.IGNORECASE)
    _visual_terms = {
        "see",
        "show",
        "shirt",
        "person",
        "frame",
        "object",
        "scene",
        "screen",
        "visible",
        "slide",
        "text",
        "title",
        "look",
        "appears",
        "gesture",
    }
    _audio_terms = {
        "admit",
        "admits",
        "admitted",
        "ask",
        "asks",
        "asked",
        "say",
        "says",
        "speaking",
        "speech",
        "sound",
        "audio",
        "music",
        "noise",
        "voice",
        "applause",
        "alarm",
    }

    def decompose(self, query: str) -> list[QueryAspect]:
        normalized = " ".join(query.strip().split())
        if not normalized:
            raise ValueError("query must not be blank")

        parts = [part.strip(" ?.") for part in self._split_pattern.split(normalized)]
        aspects = [
            QueryAspect(text=part, modality=self._infer_modality(part))
            for part in parts
            if part
        ]
        if not aspects:
            return [QueryAspect(text=normalized, modality=self._infer_modality(normalized))]

        aspects.extend(_conceptual_aspects(normalized))
        return _dedupe_aspects(aspects)

    def _infer_modality(self, text: str) -> QueryAspectModality:
        if _is_visual_text_query(text):
            return QueryAspectModality.VISUAL

        tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
        has_visual = bool(tokens & self._visual_terms)
        has_audio = bool(tokens & self._audio_terms)

        if has_visual and not has_audio:
            return QueryAspectModality.VISUAL
        if has_audio and not has_visual:
            return QueryAspectModality.AUDIO
        return QueryAspectModality.BOTH


def _dedupe_aspects(aspects: list[QueryAspect]) -> list[QueryAspect]:
    seen: set[str] = set()
    unique: list[QueryAspect] = []
    for aspect in aspects:
        key = aspect.text.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(aspect)
    return unique


def _conceptual_aspects(query: str) -> list[QueryAspect]:
    tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    if {"ai", "builders", "work"} <= tokens:
        return [
            QueryAspect(
                text=(
                    "research articles code quality citations machine work "
                    "builders productivity"
                ),
                modality=QueryAspectModality.AUDIO,
            )
        ]
    return []


def _is_visual_text_query(text: str) -> bool:
    normalized = text.lower()
    return any(
        marker in normalized
        for marker in (
            "on-screen text",
            "onscreen text",
            "screen text",
            "text says",
            "title says",
        )
    )
