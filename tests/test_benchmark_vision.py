from pathlib import Path

from gist.eval.benchmark_videomme_vision import (
    CONDITIONS,
    VisionConditionResult,
    VisionConditionSummary,
    VisionQuestionResult,
    VisionReport,
    _mc_vision_prompt,
    _uniform_frame_paths,
    render_markdown,
)


def test_uniform_frame_paths_spacing():
    frames = [Path(f"f{i}.jpg") for i in range(100)]
    picked = _uniform_frame_paths(frames, 5)
    assert len(picked) == 5
    assert picked[0] == frames[0]
    assert picked[-1] == frames[99]  # spans start to end


def test_uniform_frame_paths_fewer_than_budget():
    frames = [Path("a.jpg"), Path("b.jpg")]
    assert _uniform_frame_paths(frames, 6) == frames
    assert _uniform_frame_paths([], 6) == []


def test_mc_vision_prompt_is_frame_based_not_transcript():
    prompt = _mc_vision_prompt("what color is the car", ["A. Red", "B. Blue"])
    assert "frames" in prompt.lower()
    assert "what color is the car" in prompt
    assert "A. Red" in prompt
    assert "ONLY the letter" in prompt


def test_render_markdown_lists_frame_budgets():
    def cr(c, correct, frames):
        return VisionConditionResult(condition=c, predicted="A", correct=correct, frames=frames)
    results = [VisionQuestionResult(question_id="1", videoID="v", gold="A",
               conditions={c: cr(c, c == "gist", 6 if c != "dense" else 12) for c in CONDITIONS})]
    summaries = {c: VisionConditionSummary(condition=c, cases=1,
                 accuracy=1.0 if c == "gist" else 0.0,
                 avg_frames=12.0 if c == "dense" else 6.0) for c in CONDITIONS}
    report = VisionReport(answerer="ollama:llava:7b", questions=1, videos=1,
                          dense_budget=12, small_budget=6, summaries=summaries, results=results)
    md = render_markdown(report)
    assert "VISION setting" in md
    assert "Gist-selected (6 frames)" in md
    assert "Dense uniform (12 frames)" in md
