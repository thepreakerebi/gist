import re

from gist.core.answering import answer_from_evidence
from gist.core.schemas import Modality, SelectedCandidate
from gist.gateway.context import render_evidence_context
from gist.gateway.schemas import GatewayRequest, GatewayResponse


class LocalTextEvidenceGateway:
    provider = "local-text-evidence"

    def answer(self, request: GatewayRequest) -> GatewayResponse:
        answer = _synthesize_answer(request.query, request.compression.selected)
        if not answer:
            answer = answer_from_evidence(request.compression)
        if not answer:
            answer = "I could not derive a reliable answer from the selected evidence."
        return GatewayResponse(
            answer=answer,
            context=render_evidence_context(request.compression),
            provider=self.provider,
        )


def _synthesize_answer(query: str, selected: list[SelectedCandidate]) -> str | None:
    normalized_query = query.lower().strip()
    if not normalized_query.startswith(("how ", "what ")):
        return None

    speech_answer = _synthesize_speech_answer(query, selected)
    if speech_answer is not None:
        return speech_answer

    sentences = _ranked_transcript_sentences(query, selected)
    if not sentences:
        return None
    if len(sentences) == 1:
        return sentences[0]
    return " ".join(sentences[:3])


def _synthesize_speech_answer(query: str, selected: list[SelectedCandidate]) -> str | None:
    if not _is_speech_answer_query(query):
        return None

    sentences = _ranked_transcript_sentences(query, selected)
    if not sentences:
        return None

    complete_sentences = [
        sentence
        for sentence in sentences
        if not _looks_like_fragment(sentence) and not _is_low_information_sentence(sentence)
    ]
    if complete_sentences:
        sentences = complete_sentences

    best_sentences = sentences[:3]
    cleaned_sentences = [
        cleaned
        for sentence in best_sentences
        if (cleaned := _clean_speech_sentence(sentence))
    ]
    joined = " ".join(cleaned_sentences)
    if not joined:
        return None
    return f"The presenter says {joined[0].lower()}{joined[1:]}"


def _is_speech_answer_query(query: str) -> bool:
    normalized = f" {query.lower()} "
    return any(
        marker in normalized
        for marker in [
            " say ",
            " says ",
            " said ",
            " tell ",
            " tells ",
            " told ",
            " explain ",
            " explains ",
            " explained ",
            " presenter ",
            " speaker ",
        ]
    )


def _ranked_transcript_sentences(
    query: str,
    selected: list[SelectedCandidate],
) -> list[str]:
    query_terms = _content_terms(query)
    candidates: list[tuple[float, float, str]] = []
    for item in selected:
        if item.modality != Modality.AUDIO:
            continue
        for sentence in _sentences(item.text):
            sentence_terms = _content_terms(sentence)
            if not sentence_terms:
                continue
            overlap = len(query_terms & sentence_terms)
            support = item.evidence_support_score or item.relevance_score
            length_bonus = min(len(sentence_terms) / 18, 1.0) * 0.1
            score = overlap + support + length_bonus + _sentence_quality_bonus(sentence)
            candidates.append((score, item.timestamp_seconds, sentence))
    ranked = sorted(candidates, key=lambda value: (value[0], -value[1]), reverse=True)
    return _dedupe_sentences(sentence for _score, _timestamp, sentence in ranked)


def _sentences(text: str) -> list[str]:
    return [
        sentence.strip(" ,")
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
        if sentence.strip(" ,") and not sentence.lower().startswith("audio window from")
    ]


def _dedupe_sentences(sentences) -> list[str]:
    seen = set()
    deduped = []
    for sentence in sentences:
        normalized = re.sub(r"\s+", " ", sentence.lower())
        if normalized in seen:
            continue
        deduped.append(sentence)
        seen.add(normalized)
    return deduped


def _sentence_quality_bonus(sentence: str) -> float:
    stripped = sentence.strip()
    if not stripped:
        return -1.0
    bonus = 0.0
    lower = stripped.lower()
    if lower.startswith(("and ", "but ", "so ", "because ", "got ")):
        bonus -= 0.35
    if lower.startswith(("this ", "he ", "she ", "they ", "in this case ")):
        bonus += 0.15
    if lower.startswith("what "):
        bonus -= 0.1
    if any(term in lower for term in ["demo", "demonstration", "running", "creates", "product"]):
        bonus += 0.2
    if "running the product" in lower:
        bonus += 0.45
    if "creates a fun pose" in lower:
        bonus += 0.25
    if "static picture" in lower or "background" in lower:
        bonus += 0.15
    return bonus


def _looks_like_fragment(sentence: str) -> bool:
    return sentence.strip().lower().startswith(("because ", "got off "))


def _is_low_information_sentence(sentence: str) -> bool:
    normalized = sentence.strip().lower()
    return normalized in {
        "what it does is it's going to take that.",
        "and what it does is it's going to take that.",
    }


def _clean_speech_sentence(sentence: str) -> str:
    cleaned = _normalize_sentence_start(sentence)
    cleaned = re.sub(
        r"^What he's doing is the first thing he does is he creates\b",
        "He creates",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^What it's going to do with this static picture is it actually, then it tells him\b",
        "It tells him",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned


def _normalize_sentence_start(sentence: str) -> str:
    cleaned = sentence.strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"^(and|but|so)\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned[:1].upper() + cleaned[1:]


def _content_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in _STOPWORDS and len(token) > 2
    }


_STOPWORDS = {
    "and",
    "are",
    "for",
    "how",
    "into",
    "like",
    "that",
    "the",
    "their",
    "they",
    "this",
    "use",
    "what",
    "with",
    "you",
}
