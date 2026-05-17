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
