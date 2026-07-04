from pathlib import Path

from gist.core.schemas import Candidate, Modality
from gist.eval.downstream import (
    CONDITIONS,
    CaseConditionResult,
    DownstreamCaseResult,
    DownstreamReport,
    _approx_tokens,
    _build_condition_selected,
    _response_for,
    _summarize,
    _to_selected,
    render_downstream_markdown,
)
from gist.gateway.evidence_package import build_evidence_prompt
from gist.eval.quality import load_quality_cases


def _audio(i: int, ts: float, text: str) -> Candidate:
    return Candidate(id=f"a{i}", timestamp_seconds=ts, text=text, saliency_score=0.5)


def test_whole_uses_all_audio_uniform_matches_gist_budget():
    audio = [_audio(i, i * 30.0, f"chunk {i}") for i in range(10)]
    gist_selected = [_to_selected(audio[3], Modality.AUDIO, 1)]  # budget of 1

    uniform = _build_condition_selected("uniform", audio, gist_selected)
    whole = _build_condition_selected("whole", audio, gist_selected)

    assert len(uniform) == 1  # matches gist evidence count
    assert len(whole) == 10  # all audio windows


def test_response_and_prompt_render_transcript():
    audio = [_audio(0, 15.0, "today's lecture is about behavioral finance")]
    selected = [_to_selected(audio[0], Modality.AUDIO, 1)]
    response = _response_for("what is the topic", selected)
    assert response.metrics.audio_selected == 1
    prompt = build_evidence_prompt(response)
    assert "behavioral finance" in prompt
    assert "what is the topic" in prompt


def test_approx_tokens_grows_with_context():
    small = _approx_tokens("short")
    big = _approx_tokens("word " * 500)
    assert big > small > 0


def test_summarize_computes_condition_averages():
    def result(recall_whole, recall_gist):
        return DownstreamCaseResult(
            case_id="c",
            query="q",
            conditions={
                "whole": CaseConditionResult(
                    condition="whole", answer="a", answer_term_recall=recall_whole,
                    correct=recall_whole >= 0.5, context_tokens=4000, evidence_items=100,
                ),
                "uniform": CaseConditionResult(
                    condition="uniform", answer="a", answer_term_recall=0.0,
                    correct=False, context_tokens=100, evidence_items=1,
                ),
                "gist": CaseConditionResult(
                    condition="gist", answer="a", answer_term_recall=recall_gist,
                    correct=recall_gist >= 0.5, context_tokens=120, evidence_items=1,
                ),
            },
        )

    results = [result(1.0, 1.0), result(1.0, 1.0)]
    gist = _summarize("gist", results)
    uniform = _summarize("uniform", results)
    whole = _summarize("whole", results)
    assert gist.correct_rate == 1.0
    assert uniform.correct_rate == 0.0
    # Gist reaches whole-context accuracy at far fewer tokens.
    assert gist.avg_context_tokens < whole.avg_context_tokens
    assert gist.avg_answer_term_recall == whole.avg_answer_term_recall


def test_render_markdown_lists_all_conditions():
    results = [
        DownstreamCaseResult(
            case_id="c", query="q",
            conditions={
                cond: CaseConditionResult(
                    condition=cond, answer="a", answer_term_recall=1.0,
                    correct=True, context_tokens=100, evidence_items=1,
                )
                for cond in CONDITIONS
            },
        )
    ]
    report = DownstreamReport(
        cases=1, answerer="ollama:llama3.2:3b", correct_threshold=0.5,
        summaries={c: _summarize(c, results) for c in CONDITIONS}, results=results,
    )
    md = render_downstream_markdown(report)
    assert "Whole transcript" in md
    assert "Gist-compressed evidence" in md
    assert "Uniform sampling" in md


def test_dataset_has_transcript_answerable_cases():
    cases = load_quality_cases(Path("data/eval/long-video-quality.jsonl"))
    speech = [c for c in cases if c.query_category and c.query_category.value == "speech_semantic"]
    assert len(speech) >= 10
