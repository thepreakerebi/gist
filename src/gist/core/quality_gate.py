from gist.core.query_intent import QueryIntent
from gist.core.schemas import CompressionResponse, QualityWarning

MIN_ANSWER_WORDS = 6
MIN_TOKEN_REDUCTION_PERCENT = 80.0
NOISY_TRANSCRIPT_RATIO = 0.5
_NOISY_TRANSCRIPT_PHRASES = {
    "a lot of",
    "and then",
    "basically",
    "kind of",
    "so this is",
    "this role",
    "we have",
}


def apply_quality_gate(compression: CompressionResponse) -> CompressionResponse:
    warnings = quality_warnings(compression)
    return compression.model_copy(update={"quality_warnings": warnings})


def quality_warnings(compression: CompressionResponse) -> list[QualityWarning]:
    warnings: list[QualityWarning] = []
    answer_words = _word_count(compression.answer)
    transcript_backed_count = sum(1 for item in compression.selected if _has_transcript(item.text))
    noisy_transcript_count = sum(
        1
        for item in compression.selected
        if _has_transcript(item.text) and _is_noisy_transcript(item.text)
    )
    grounded_count = sum(
        1 for item in compression.selected if item.grounding_label in {"direct", "contextual"}
    )
    strong_count = sum(1 for item in compression.selected if item.support_label == "strong")

    if not compression.selected:
        warnings.append(
            QualityWarning(
                code="no_evidence",
                message="No final evidence was selected.",
                severity="error",
            )
        )
    if not compression.answer or answer_words < MIN_ANSWER_WORDS:
        warnings.append(
            QualityWarning(
                code="weak_answer",
                message="Answer is missing or too short to be useful.",
            )
        )
    if transcript_backed_count == 0 and _needs_transcript(compression):
        warnings.append(
            QualityWarning(
                code="missing_transcript_evidence",
                message=(
                    "Final evidence has no transcript-backed video moment "
                    "for a speech/mixed query."
                ),
                severity="error",
            )
        )
    if (
        transcript_backed_count > 0
        and _needs_transcript(compression)
        and noisy_transcript_count / transcript_backed_count >= NOISY_TRANSCRIPT_RATIO
    ):
        warnings.append(
            QualityWarning(
                code="noisy_transcript_evidence",
                message=(
                    "Transcript-backed evidence appears noisy or low-signal. "
                    "Rerun with --transcript-quality accurate or a larger "
                    "Whisper model when answer quality matters."
                ),
            )
        )
    if compression.selected and grounded_count == 0:
        warnings.append(
            QualityWarning(
                code="ungrounded_evidence",
                message="Final evidence did not reach direct or contextual grounding.",
            )
        )
    if compression.selected and strong_count == 0:
        warnings.append(
            QualityWarning(
                code="weak_evidence_support",
                message="Final evidence did not reach strong answer/query support.",
            )
        )
    if (
        compression.metrics.estimated_token_reduction_percent
        < MIN_TOKEN_REDUCTION_PERCENT
    ):
        warnings.append(
            QualityWarning(
                code="low_token_reduction",
                message=(
                    "Estimated token reduction is below "
                    f"{MIN_TOKEN_REDUCTION_PERCENT:.0f}%."
                ),
            )
        )
    return warnings


def _needs_transcript(compression: CompressionResponse) -> bool:
    return compression.query_intent in {
        QueryIntent.SPEECH_SEMANTIC,
        QueryIntent.MIXED_AV,
        QueryIntent.SOUND_EVENT,
        QueryIntent.GLOBAL_SUMMARY,
    }


def _has_transcript(text: str) -> bool:
    normalized = " ".join(text.split()).lower()
    if not normalized:
        return False
    return not (
        normalized.startswith("visual frame sampled at")
        or normalized.startswith("on-screen text near")
    )


def _word_count(value: str | None) -> int:
    if value is None:
        return 0
    return len([token for token in value.split() if token.strip()])


def _is_noisy_transcript(text: str) -> bool:
    normalized = " ".join(text.split()).lower()
    if not normalized:
        return True
    phrase_hits = sum(phrase in normalized for phrase in _NOISY_TRANSCRIPT_PHRASES)
    words = [word.strip(".,!?;:()[]{}").lower() for word in normalized.split()]
    content_words = [word for word in words if len(word) > 3]
    unique_ratio = len(set(content_words)) / max(len(content_words), 1)
    short_or_fragmented = len(content_words) < 2 or normalized.endswith((" of", " a", " the"))
    return phrase_hits >= 2 or unique_ratio < 0.45 or short_or_fragmented
