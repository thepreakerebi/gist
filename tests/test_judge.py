from gist.eval.judge import JudgeVerdict, _extract_verdict, build_judge_prompt


def test_prompt_includes_query_terms_and_answer():
    prompt = build_judge_prompt(
        query="who are the big three artists",
        expected_terms=["leonardo", "raphael", "renaissance"],
        answer="Leonardo, Michelangelo, and Raphael.",
    )
    assert "who are the big three artists" in prompt
    assert "leonardo, raphael, renaissance" in prompt
    assert "Leonardo, Michelangelo, and Raphael." in prompt
    assert "JSON" in prompt


def test_extract_verdict_parses_clean_json():
    v = _extract_verdict('{"correct": true, "score": 0.9, "reason": "covers the facts"}')
    assert v.correct is True
    assert v.score == 0.9
    assert "covers" in v.reason


def test_extract_verdict_handles_surrounding_prose():
    v = _extract_verdict('Sure! Here is my grade: {"correct": false, "score": 0.1, "reason": "off topic"} done')
    assert v.correct is False
    assert v.score == 0.1


def test_extract_verdict_clamps_and_defaults():
    v = _extract_verdict('{"correct": true, "score": 5.0, "reason": "x"}')
    assert v.score == 1.0  # clamped
    missing = _extract_verdict("no json here at all")
    assert missing.correct is False
    assert missing.score == 0.0


def test_verdict_model_roundtrip():
    v = JudgeVerdict(correct=True, score=0.8, reason="ok")
    assert v.model_dump()["correct"] is True
