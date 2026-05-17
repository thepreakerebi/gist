from gist.core.presets import CompressionPreset
from gist.core.schemas import Candidate
from gist.eval.baselines import uniform_baseline
from gist.eval.reporting import render_markdown_report
from gist.eval.runner import EvalRunner
from gist.eval.schemas import EvalExample, EvalSettings


def test_uniform_baseline_selects_evenly_spaced_candidates() -> None:
    example = EvalExample(
        id="case-1",
        video_id="v1",
        query="pricing",
        duration_seconds=60,
        relevant_timestamps=[0, 50],
        visual_candidates=[
            Candidate(id=f"v-{index}", timestamp_seconds=float(index * 10), text="frame")
            for index in range(6)
        ],
    )

    result = uniform_baseline(example, CompressionPreset.AGGRESSIVE)

    assert result.name == "uniform"
    assert result.selected_candidates == 6
    assert result.timestamp_hit_rate == 1


def test_eval_runner_builds_report_summary() -> None:
    example = EvalExample(
        id="case-1",
        video_id="v1",
        query="pricing",
        duration_seconds=60,
        relevant_timestamps=[10],
        visual_candidates=[
            Candidate(id="v-1", timestamp_seconds=10, text="pricing slide"),
            Candidate(id="v-2", timestamp_seconds=30, text="closing slide"),
        ],
        audio_candidates=[
            Candidate(id="a-1", timestamp_seconds=11, text="speaker explains pricing"),
        ],
    )

    report = EvalRunner().run(
        [example],
        EvalSettings(preset=CompressionPreset.AGGRESSIVE),
    )

    assert report.summary.examples == 1
    assert report.summary.avg_gist_timestamp_hit_rate == 1
    assert report.results[0].baselines[0].name == "uniform"


def test_render_markdown_report_includes_summary() -> None:
    example = EvalExample(
        id="case-1",
        video_id="v1",
        query="pricing",
        duration_seconds=60,
        visual_candidates=[Candidate(id="v-1", timestamp_seconds=10, text="pricing slide")],
    )
    report = EvalRunner().run([example], EvalSettings())

    markdown = render_markdown_report(report)

    assert "# Gist Evaluation Report" in markdown
    assert "case-1" in markdown

