from typing import Any


def build_video_answer_prompt(gateway_payload: dict[str, Any]) -> str:
    query = str(gateway_payload.get("query", "")).strip()
    context = str(gateway_payload.get("context", "")).strip()
    query_intent = _query_intent(gateway_payload)
    task_instruction = _task_instruction(query_intent)

    return (
        "Answer the video question using only the provided Gist evidence frames and "
        "evidence context. Treat evidence frames as ordered by relevance and timestamp. "
        "If the question is multiple choice, answer with the single best choice letter "
        "and a very short justification. Do not use outside knowledge.\n\n"
        f"Task guidance: {task_instruction}\n\n"
        f"Question: {query}\n\n"
        f"Evidence context:\n{context}"
    )


def _query_intent(gateway_payload: dict[str, Any]) -> str | None:
    compression = gateway_payload.get("compression")
    if isinstance(compression, dict):
        intent = compression.get("query_intent")
        if isinstance(intent, str) and intent.strip():
            return intent.strip()

    intent = gateway_payload.get("query_intent")
    if isinstance(intent, str) and intent.strip():
        return intent.strip()
    return None


def _task_instruction(query_intent: str | None) -> str:
    if query_intent == "counting_comparison":
        return (
            "Count or compare visible entities directly in the evidence frames. "
            "Use narration only as support; do not infer counts from transcript text alone."
        )
    if query_intent == "negative_evidence":
        return (
            "Check each choice against the evidence context and frames. Prefer the option "
            "that is not shown or not mentioned when the question asks for an exception."
        )
    if query_intent == "temporal_before_after":
        return (
            "Reason over the chronological order of evidence. Use pre-context and post-context "
            "clips to decide what happened before, after, or next."
        )
    if query_intent == "speech_semantic":
        return "Prioritize spoken/transcript evidence, then verify with the matching video frames."
    if query_intent == "visual_object_action":
        return "Prioritize visible objects, actions, scene changes, and on-screen text in the frames."
    if query_intent == "sound_event":
        return "Use transcript/audio evidence for sound events, then verify whether frames support it."
    if query_intent == "global_summary":
        return "Synthesize across all evidence clips instead of over-weighting a single moment."
    return "Use both visual frames and evidence context; prefer direct evidence over inference."
