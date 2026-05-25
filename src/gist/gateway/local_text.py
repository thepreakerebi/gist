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

    sentences = _ranked_transcript_sentences(query, selected)
    if not sentences:
        return None
    if len(sentences) == 1:
        return sentences[0]
    return " ".join(sentences[:3])


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
            score = overlap + support + length_bonus
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
