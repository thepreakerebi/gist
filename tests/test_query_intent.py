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


def test_routes_speech_while_showing_query_to_mixed_av() -> None:
    intent, reason = route_query_intent(
        "What does the woman say about robotics and space while showing her robot hand?"
    )

    assert intent == QueryIntent.MIXED_AV
    assert "cross-modal" in reason


def test_routes_ask_admit_visible_query_to_mixed_av() -> None:
    intent, reason = route_query_intent(
        "What does the woman ask about her robot hand, and what does the man admit while the robot hand is visible?"
    )

    assert intent == QueryIntent.MIXED_AV
    assert "cross-modal" in reason


def test_routes_visual_query() -> None:
    intent, reason = route_query_intent("show the person holding the tool")

    assert intent == QueryIntent.VISUAL_OBJECT_ACTION
    assert "visual" in reason


def test_routes_on_screen_text_says_query_to_visual() -> None:
    intent, reason = route_query_intent(
        "What on-screen text says Characterization and Modelling?"
    )

    assert intent == QueryIntent.VISUAL_OBJECT_ACTION
    assert "OCR" in reason


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


def test_counting_with_speech_signal_routes_to_speech() -> None:
    # A count about spoken content must not be forced to visual-only counting.
    intent, _reason = route_query_intent(
        "according to the narrator, how many solar cells does each panel contain"
    )
    assert intent == QueryIntent.SPEECH_SEMANTIC


def test_visual_counting_stays_counting_comparison() -> None:
    # Counting about on-screen objects should still favour visual evidence.
    intent, _reason = route_query_intent("how many people appear on screen in the frame")
    assert intent == QueryIntent.COUNTING_COMPARISON


def test_narration_terms_route_to_speech() -> None:
    intent, _reason = route_query_intent("what does the commentary describe about the mission")
    assert intent == QueryIntent.SPEECH_SEMANTIC
