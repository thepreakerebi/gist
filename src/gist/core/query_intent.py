import re
from enum import StrEnum


class QueryIntent(StrEnum):
    SPEECH_SEMANTIC = "speech_semantic"
    VISUAL_OBJECT_ACTION = "visual_object_action"
    TEMPORAL_BEFORE_AFTER = "temporal_before_after"
    GLOBAL_SUMMARY = "global_summary"
    SOUND_EVENT = "sound_event"
    COUNTING_COMPARISON = "counting_comparison"
    NEGATIVE_EVIDENCE = "negative_evidence"
    MIXED_AV = "mixed_av"


_SPEECH_TERMS = {
    "say",
    "said",
    "says",
    "speak",
    "speaks",
    "speaker",
    "talk",
    "talks",
    "tell",
    "tells",
    "mention",
    "mentions",
    "voice",
    "transcript",
}
_CONCEPTUAL_SPEECH_TERMS = {
    "according",
    "answer",
    "claim",
    "claims",
    "explain",
    "explains",
    "idea",
    "mean",
    "means",
    "reason",
    "why",
}
_VISUAL_TERMS = {
    "appear",
    "appears",
    "display",
    "displayed",
    "image",
    "logo",
    "projection",
    "show",
    "shows",
    "see",
    "seen",
    "look",
    "looks",
    "wearing",
    "holding",
    "object",
    "person",
    "people",
    "text",
    "screen",
    "slide",
    "title",
    "frame",
}
_TEMPORAL_TERMS = {
    "after",
    "before",
    "then",
    "next",
    "prior",
    "following",
    "during",
    "when",
    "while",
}
_GLOBAL_TERMS = {
    "summarize",
    "summary",
    "overall",
    "main",
    "theme",
    "about",
    "describe",
    "overview",
}
_SOUND_TERMS = {
    "sound",
    "noise",
    "music",
    "applause",
    "alarm",
    "siren",
    "laugh",
    "laughing",
    "engine",
    "bang",
    "gunshot",
    "footsteps",
}
_COUNTING_TERMS = {
    "count",
    "counting",
    "many",
    "number",
    "largest",
    "smallest",
    "more",
    "most",
    "least",
    "total",
}
_NEGATIVE_TERMS = {
    "not",
    "except",
    "neither",
    "none",
    "without",
    "didn",
    "doesn",
    "isn",
    "aren",
    "absent",
    "missing",
    "undiscussed",
}
_NEGATIVE_PHRASES = {
    "not discussed",
    "not mentioned",
    "does not",
    "did not",
    "is not",
    "are not",
    "which of the following is not",
}


def route_query_intent(query: str) -> tuple[QueryIntent, str]:
    tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    if not tokens:
        return (
            QueryIntent.MIXED_AV,
            "blank or tokenless query defaults to mixed audio-visual routing",
        )

    has_speech = bool(tokens & _SPEECH_TERMS)
    has_conceptual_speech = bool(tokens & _CONCEPTUAL_SPEECH_TERMS) or (
        bool(tokens & {"what", "how"}) and not bool(tokens & _VISUAL_TERMS)
    )
    has_visual = bool(tokens & _VISUAL_TERMS)
    has_temporal = bool(tokens & _TEMPORAL_TERMS)
    has_global = bool(tokens & _GLOBAL_TERMS)
    has_sound = bool(tokens & _SOUND_TERMS)
    has_counting = bool(tokens & _COUNTING_TERMS) or "how many" in query.lower()
    has_negative = bool(tokens & _NEGATIVE_TERMS) or any(
        phrase in query.lower() for phrase in _NEGATIVE_PHRASES
    )

    if has_negative:
        return (
            QueryIntent.NEGATIVE_EVIDENCE,
            "negative query terms require coverage of mentioned and unmentioned alternatives",
        )
    if has_counting:
        return (
            QueryIntent.COUNTING_COMPARISON,
            "counting/comparison terms require denser visual evidence around relevant moments",
        )
    if has_global and not (has_speech or has_visual or has_sound):
        return QueryIntent.GLOBAL_SUMMARY, "global summary terms favor broad segment coverage"
    if has_temporal and (has_speech or has_visual or has_sound):
        return (
            QueryIntent.TEMPORAL_BEFORE_AFTER,
            "temporal markers require continuity around relevant moments",
        )
    if has_speech and has_visual:
        return QueryIntent.MIXED_AV, "speech and visual terms require cross-modal evidence"
    if has_conceptual_speech and not (has_visual or has_sound):
        return (
            QueryIntent.SPEECH_SEMANTIC,
            "conceptual question terms favor transcript-first retrieval",
        )
    if has_speech:
        return QueryIntent.SPEECH_SEMANTIC, "speech terms favor transcript-first retrieval"
    if has_sound:
        return QueryIntent.SOUND_EVENT, "sound-event terms favor audio event retrieval"
    if has_visual:
        return (
            QueryIntent.VISUAL_OBJECT_ACTION,
            "visual terms favor frame and object/action retrieval",
        )
    if has_temporal:
        return (
            QueryIntent.TEMPORAL_BEFORE_AFTER,
            "temporal markers require continuity around relevant moments",
        )
    if has_global:
        return QueryIntent.GLOBAL_SUMMARY, "global terms favor broad segment coverage"
    return QueryIntent.MIXED_AV, "no dominant route detected; using mixed audio-visual retrieval"
