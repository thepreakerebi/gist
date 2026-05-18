from gist.eval.answers import answer_score


def test_answer_score_matches_exact_normalized_answer() -> None:
    assert answer_score("The answer is Mars.", "Mars", []) == 1.0


def test_answer_score_matches_choice_letter() -> None:
    assert answer_score("B", "second option", ["first option", "second option"]) == 1.0


def test_answer_score_returns_none_without_expected_answer() -> None:
    assert answer_score("Mars", None, []) is None


def test_answer_score_rejects_wrong_choice() -> None:
    assert answer_score("A", "B", ["first", "second"]) == 0.0
