import pytest

from gist.core.decomposition import QueryAspectModality, RuleBasedQueryDecomposer


def test_rule_based_decomposer_splits_compound_queries() -> None:
    aspects = RuleBasedQueryDecomposer().decompose(
        "show the person in red shirt and what does the speaker say"
    )

    assert [aspect.text for aspect in aspects] == [
        "show the person in red shirt",
        "what does the speaker say",
    ]
    assert aspects[0].modality == QueryAspectModality.VISUAL
    assert aspects[1].modality == QueryAspectModality.AUDIO


def test_rule_based_decomposer_rejects_blank_queries() -> None:
    with pytest.raises(ValueError, match="query must not be blank"):
        RuleBasedQueryDecomposer().decompose(" ")


def test_rule_based_decomposer_treats_ask_as_audio_signal() -> None:
    aspects = RuleBasedQueryDecomposer().decompose(
        "What does the woman ask while her robot hand is visible?"
    )

    assert aspects[0].modality == QueryAspectModality.AUDIO
    assert aspects[1].modality == QueryAspectModality.VISUAL


def test_rule_based_decomposer_routes_on_screen_text_says_to_visual() -> None:
    aspects = RuleBasedQueryDecomposer().decompose(
        "What on-screen text says Further Reading Materials?"
    )

    assert aspects[0].modality == QueryAspectModality.VISUAL
