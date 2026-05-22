from gist.core.schemas import Candidate
from gist.core.scoring import lexical_relevance


def test_lexical_relevance_ignores_stopword_only_overlap() -> None:
    candidate = Candidate(
        id="v1",
        timestamp_seconds=1,
        text="on-screen text near 58 seconds: of an the",
    )

    assert lexical_relevance(
        "How do top builders use AI to do the work of hundreds of engineers?",
        candidate,
    ) == 0.0
