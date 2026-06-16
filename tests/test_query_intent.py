from gist.core.query_intent import QueryIntent, route_query_intent


def test_routes_temporal_speech_query() -> None:
    intent, reason = route_query_intent("when does the speaker mention architecture")

    assert intent == QueryIntent.TEMPORAL_BEFORE_AFTER
    assert "temporal" in reason


def test_routes_temporal_slide_query_without_treating_it_as_speech() -> None:
    intent, reason = route_query_intent(
        "What slide appears immediately after the Worldwide Telescope slide?"
    )

    assert intent == QueryIntent.TEMPORAL_BEFORE_AFTER
    assert "temporal" in reason


def test_routes_visual_query() -> None:
    intent, reason = route_query_intent("show the person holding the tool")

    assert intent == QueryIntent.VISUAL_OBJECT_ACTION
    assert "visual" in reason


def test_routes_sound_event_query() -> None:
    intent, reason = route_query_intent("where is the applause loudest")

    assert intent == QueryIntent.SOUND_EVENT
    assert "sound" in reason


def test_routes_global_query() -> None:
    intent, reason = route_query_intent("summarize the overall video")

    assert intent == QueryIntent.GLOBAL_SUMMARY
    assert "global" in reason


def test_routes_counting_query() -> None:
    intent, reason = route_query_intent("how many red socks are above the fireplace")

    assert intent == QueryIntent.COUNTING_COMPARISON
    assert "counting" in reason


def test_routes_negative_evidence_query() -> None:
    intent, reason = route_query_intent("which option is not discussed in the video")

    assert intent == QueryIntent.NEGATIVE_EVIDENCE
    assert "negative" in reason


def test_routes_conceptual_question_to_speech_semantic() -> None:
    intent, reason = route_query_intent(
        "How do top builders use AI to do the work of hundreds of engineers?"
    )

    assert intent == QueryIntent.SPEECH_SEMANTIC
    assert "transcript-first" in reason


def test_routes_mixed_query_when_no_dominant_signal_exists() -> None:
    intent, reason = route_query_intent("pricing")

    assert intent == QueryIntent.MIXED_AV
    assert "mixed" in reason
