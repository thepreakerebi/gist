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


def test_rule_based_decomposer_expands_ai_builder_productivity_queries() -> None:
    aspects = RuleBasedQueryDecomposer().decompose(
        "How do top builders use AI to do the work of hundreds of engineers?"
    )

    assert aspects[-1].modality == QueryAspectModality.AUDIO
    assert "research articles" in aspects[-1].text
    assert "machine work" in aspects[-1].text
