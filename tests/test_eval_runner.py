from pathlib import Path

from gist.core.modes import AudioScoringMode, VisualScoringMode
from gist.core.presets import CompressionPreset
from gist.core.schemas import Candidate, Modality
from gist.eval.baselines import score_topk_baseline, uniform_baseline
from gist.eval.reporting import render_html_report, render_markdown_report
from gist.eval.runner import EvalRunner
from gist.eval.schemas import EvalExample, EvalSettings
from gist.gateway.schemas import GatewayRequest, GatewayResponse


class FixedGateway:
    provider = "fixed"

    def __init__(self, answer: str) -> None:
        self.fixed_answer = answer

    def answer(self, request: GatewayRequest) -> GatewayResponse:
        return GatewayResponse(
            answer=self.fixed_answer,
            context="fixed context",
            provider=self.provider,
        )


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


def test_score_topk_baseline_selects_salient_candidates() -> None:
    example = EvalExample(
        id="case-1",
        video_id="v1",
        query="pricing",
        duration_seconds=60,
        visual_candidates=[
            Candidate(
                id=f"v-{index}",
                timestamp_seconds=float(index),
                text="frame",
                saliency_score=0.1,
            )
            for index in range(8)
        ]
        + [
            Candidate(id="v-high", timestamp_seconds=20, text="high", saliency_score=0.9),
        ],
    )

    result = score_topk_baseline(example, CompressionPreset.AGGRESSIVE)

    assert result.name == "score_topk"
    assert result.selected_candidates == 6
    assert "v-high" in {item.id for item in result.selected}


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

    report = EvalRunner().run([example], EvalSettings(preset=CompressionPreset.AGGRESSIVE))

    assert report.summary.examples == 1
    assert report.summary.variants["gist_configured"].avg_timestamp_hit_rate == 1
    assert "avg_token_reduction_percent" in report.summary.variants[
        "gist_configured"
    ].model_dump()
    assert [baseline.name for baseline in report.results[0].baselines] == [
        "uniform",
        "score_topk",
    ]
    assert report.results[0].variants[0].name == "gist_configured"


def test_eval_runner_scores_gateway_answers() -> None:
    example = EvalExample(
        id="case-1",
        video_id="v1",
        query="which planet is discussed",
        duration_seconds=60,
        expected_answer="Mars",
        visual_candidates=[
            Candidate(id="v-1", timestamp_seconds=10, text="Mars appears on screen"),
        ],
    )

    report = EvalRunner(gateway=FixedGateway("Mars")).run(
        [example],
        EvalSettings(preset=CompressionPreset.AGGRESSIVE),
    )

    variant = report.results[0].variants[0]
    assert variant.predicted_answer == "Mars"
    assert variant.answer_score == 1
    assert variant.answer_provider == "fixed"
    assert [baseline.answer_score for baseline in report.results[0].baselines] == [1, 1]
    assert [baseline.answer_provider for baseline in report.results[0].baselines] == [
        "fixed",
        "fixed",
    ]
    assert report.summary.variants["gist_configured"].avg_answer_score == 1


def test_eval_runner_passes_configured_scorers_to_single_variant() -> None:
    example = EvalExample(
        id="case-1",
        video_id="v1",
        query="pricing",
        duration_seconds=60,
        visual_candidates=[Candidate(id="v-1", timestamp_seconds=10, text="pricing slide")],
    )

    report = EvalRunner().run(
        [example],
        EvalSettings(
            visual_scorer=VisualScoringMode.CLIP,
            audio_scorer=AudioScoringMode.WHISPER,
        ),
    )

    variant = report.variants[0]
    assert variant.visual_scorer == VisualScoringMode.CLIP
    assert variant.audio_scorer == AudioScoringMode.WHISPER


def test_eval_runner_runs_default_variant_sweep() -> None:
    example = EvalExample(
        id="case-1",
        video_id="v1",
        query="pricing",
        duration_seconds=60,
        visual_candidates=[Candidate(id="v-1", timestamp_seconds=10, text="pricing slide")],
    )

    report = EvalRunner().run([example])

    assert len(report.variants) == 5
    assert "gist_decomposed_adaptive" in report.summary.variants
    assert len(report.results[0].variants) == 5


def test_eval_runner_supports_real_video_examples(tmp_path) -> None:
    video_path = tmp_path / "video.mp4"
    import subprocess

    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x120:rate=10:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:duration=1",
            "-shortest",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    example = EvalExample(
        id="video-case",
        video_id="video-1",
        video_path=video_path,
        query="audio",
        duration_seconds=1,
        relevant_timestamps=[0],
        sample_count=2,
        audio_window_seconds=0.5,
    )

    report = EvalRunner(output_root=tmp_path / "eval").run(
        [example],
        EvalSettings(preset=CompressionPreset.AGGRESSIVE, spatial_pruning=True),
    )

    assert report.results[0].variants[0].response.metrics.input_candidates == 4
    assert report.summary.variants["gist_configured"].avg_timestamp_hit_rate == 1
    for item in report.results[0].variants[0].response.selected:
        assert item.clip_path is not None
        assert item.clip_path.exists()
        assert item.clip_start_seconds is not None
        assert item.clip_end_seconds is not None
        assert item.clip_end_seconds > item.clip_start_seconds
        if item.modality == Modality.VISUAL:
            assert item.spatial_mask_path is not None
            assert item.spatial_mask_path.exists()
            assert item.spatial_mask_preview_path is not None
            assert item.spatial_mask_preview_path.exists()
            if item.asset_path is not None:
                assert item.spatial_mask_overlay_path is not None
                assert item.spatial_mask_overlay_path.exists()
    assert (
        report.results[0]
        .variants[0]
        .response.metrics.estimated_spatial_token_reduction_percent
        > 0
    )


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
    assert "Baselines" in markdown
    assert "Variant" in markdown


def test_render_html_report_includes_evidence() -> None:
    frame_path = Path(__file__)
    example = EvalExample(
        id="case-1",
        video_id="v1",
        query="pricing",
        duration_seconds=60,
        visual_candidates=[
            Candidate(
                id="v-1",
                timestamp_seconds=10,
                text="pricing slide",
                asset_path=frame_path,
            )
        ],
    )
    report = EvalRunner().run([example], EvalSettings())

    html = render_html_report(report)

    assert "<html" in html
    assert "Baselines" in html
    assert "pricing slide" in html
    assert "evidence-frame" in html
    assert frame_path.resolve().as_uri() in html


def test_render_html_report_prefers_video_clip_over_frame() -> None:
    clip_path = Path(__file__)
    example = EvalExample(
        id="case-1",
        video_id="v1",
        query="pricing",
        duration_seconds=60,
        visual_candidates=[Candidate(id="v-1", timestamp_seconds=10, text="pricing slide")],
    )
    report = EvalRunner().run([example], EvalSettings())
    selected = [
        item.model_copy(update={"clip_path": clip_path})
        for item in report.results[0].variants[0].response.selected
    ]
    report.results[0].variants[0].response = report.results[0].variants[
        0
    ].response.model_copy(update={"selected": selected})

    html = render_html_report(report)

    assert "<video" in html
    assert "evidence-clip" in html
    assert clip_path.resolve().as_uri() in html


def test_render_html_report_merges_overlapping_audio_visual_evidence() -> None:
    clip_path = Path(__file__)
    example = EvalExample(
        id="case-1",
        video_id="v1",
        query="architecture",
        duration_seconds=60,
        visual_candidates=[
            Candidate(id="v-near", timestamp_seconds=10.2, text="visual frame near topic")
        ],
        audio_candidates=[
            Candidate(id="a-topic", timestamp_seconds=10.0, text="speaker says architecture")
        ],
    )
    report = EvalRunner().run([example], EvalSettings())
    selected = [
        item.model_copy(update={"clip_path": clip_path})
        for item in report.results[0].variants[0].response.selected
    ]
    report.results[0].variants[0].response = report.results[0].variants[
        0
    ].response.model_copy(update={"selected": selected})

    html = render_html_report(report)

    assert "rendered 1 video evidence clips" in html
    assert html.count("<video") == 1
    assert "speaker says architecture" in html
    assert "visual frame near topic" in html


def test_render_html_report_includes_spatial_debug_image(tmp_path: Path) -> None:
    preview_path = tmp_path / "mask.svg"
    preview_path.write_text("<svg></svg>")
    example = EvalExample(
        id="case-1",
        video_id="v1",
        query="show pricing slide",
        duration_seconds=60,
        visual_candidates=[
            Candidate(id="v-near", timestamp_seconds=10.2, text="visual frame near topic")
        ],
    )
    report = EvalRunner().run([example], EvalSettings())
    selected = [
        item.model_copy(
            update={
                "spatial_mask_preview_path": preview_path,
                "spatial_mask_overlay_path": preview_path,
            }
        )
        for item in report.results[0].variants[0].response.selected
    ]
    report.results[0].variants[0].response = report.results[0].variants[
        0
    ].response.model_copy(update={"selected": selected})

    html = render_html_report(report)

    assert "Spatial mask overlay" in html
    assert preview_path.resolve().as_uri() in html


def test_render_html_report_keeps_answer_and_precontext_but_drops_weak_late_context() -> None:
    clip_path = Path(__file__)
    example = EvalExample(
        id="case-1",
        video_id="v1",
        query="architecture missions",
        duration_seconds=140,
        audio_candidates=[
            Candidate(
                id="a-pre",
                timestamp_seconds=37,
                text="building the next chapter returning to the moon",
            ),
            Candidate(
                id="a-answer",
                timestamp_seconds=58,
                text="the architecture for these missions is taking shape",
            ),
            Candidate(
                id="a-theme",
                timestamp_seconds=94,
                text="sustainable science and human spirit",
            ),
            Candidate(
                id="a-outro",
                timestamp_seconds=122,
                text="every day every mission we advance",
            ),
        ],
    )
    report = EvalRunner().run([example], EvalSettings())
    selected = [
        item.model_copy(update={"clip_path": clip_path})
        for item in report.results[0].variants[0].response.selected
    ]
    report.results[0].variants[0].response = report.results[0].variants[
        0
    ].response.model_copy(update={"selected": selected})

    html = render_html_report(report)

    assert "rendered 2 video evidence clips" in html
    assert "building the next chapter" in html
    assert "the architecture for these missions" in html
    assert "sustainable science" not in html
    assert "every day every mission" not in html
