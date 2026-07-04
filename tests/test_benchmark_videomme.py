import json
from pathlib import Path

from gist.core.schemas import SelectedCandidate, Modality
from gist.eval.benchmark_videomme import (
    CONDITIONS,
    BenchConditionResult,
    BenchQuestionResult,
    BenchReport,
    _mc_prompt,
    _parse_letter,
    _parse_options,
    load_questions,
    render_markdown,
)


def test_parse_options_from_stringified_list():
    opts = _parse_options("['A. Apples.', 'B. Candles.', 'C. Berries.', 'D. None.']")
    assert opts[0].startswith("A.")
    assert len(opts) == 4


def test_parse_letter_extracts_choice():
    assert _parse_letter("The answer is C.") == "C"
    assert _parse_letter("B") == "B"
    assert _parse_letter("no letter here") == ""


def test_mc_prompt_contains_question_options_and_evidence():
    ev = [SelectedCandidate(
        id="w1", modality=Modality.AUDIO, timestamp_seconds=10.0, text="they decorate with candles",
        scene_start_seconds=8.0, scene_end_seconds=12.0, selection_rank=1,
        relevance_score=0.5, normalized_score=0.5, mmr_score=0.5,
        source_score_type="t", reason="r")]
    prompt = _mc_prompt("what decoration is used", ["A. Apples", "B. Candles"], ev)
    assert "what decoration is used" in prompt
    assert "B. Candles" in prompt
    assert "candles" in prompt
    assert "ONLY the letter" in prompt


def test_load_questions_filters_by_video(tmp_path):
    rows = [
        {"question_id": "1", "videoID": "vidA", "question": "q1",
         "options": "['A. x', 'B. y']", "answer": "A"},
        {"question_id": "2", "videoID": "vidB", "question": "q2",
         "options": ["A. x", "B. y"], "answer": "b"},
    ]
    p = tmp_path / "q.json"
    p.write_text(json.dumps(rows))
    only_a = load_questions(p, {"vidA"})
    assert len(only_a) == 1 and only_a[0].videoID == "vidA"
    both = load_questions(p, None)
    assert len(both) == 2
    assert both[1].answer == "B"  # normalized to uppercase letter


def test_render_markdown_lists_conditions():
    def cr(c, correct):
        return BenchConditionResult(condition=c, predicted="A", correct=correct, context_tokens=100)
    results = [BenchQuestionResult(question_id="1", videoID="v", gold="A",
               conditions={c: cr(c, c == "gist") for c in CONDITIONS})]
    from gist.eval.benchmark_videomme import BenchConditionSummary
    summaries = {c: BenchConditionSummary(condition=c, cases=1,
                 accuracy=1.0 if c == "gist" else 0.0, avg_context_tokens=100.0)
                 for c in CONDITIONS}
    report = BenchReport(answerer="ollama:llama3.2:3b", questions=1, videos=1,
                         summaries=summaries, results=results)
    md = render_markdown(report)
    assert "Video-MME" in md
    assert "Gist-compressed" in md
