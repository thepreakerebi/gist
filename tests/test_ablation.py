import json
from pathlib import Path

from gist.core.compressor import GistCompressor
from gist.core.presets import CompressionPreset
from gist.core.schemas import Candidate, CompressionRequest
from gist.eval.ablation import (
    MODES,
    ModeOutcome,
    AblationCaseResult,
    _compress_mode,
    _summarize,
    _uniform_mode,
    render_ablation_markdown,
    resolve_case_config,
    run_ablation_suite,
)
from gist.eval.quality import QualityCase, load_quality_cases


def _visual(idx: int, timestamp: float, text: str, saliency: float) -> Candidate:
    return Candidate(
        id=f"v{idx}",
        timestamp_seconds=timestamp,
        text=text,
        saliency_score=saliency,
        scene_start_seconds=timestamp,
        scene_end_seconds=timestamp + 1.0,
    )


def _audio(idx: int, timestamp: float, text: str, saliency: float) -> Candidate:
    return Candidate(
        id=f"a{idx}",
        timestamp_seconds=timestamp,
        text=text,
        saliency_score=saliency,
        scene_start_seconds=timestamp,
        scene_end_seconds=timestamp + 5.0,
    )


def _request(query: str) -> CompressionRequest:
    return CompressionRequest(
        video_id="vid-1",
        query=query,
        duration_seconds=3600.0,
        preset=CompressionPreset.BALANCED,
        adaptive_budget=True,
        decompose_query=True,
        task_aware_selection=True,
    )


def test_uniform_mode_respects_budget_and_ignores_scores():
    visual = [_visual(i, i * 10.0, "on-screen text: noise", 0.9) for i in range(6)]
    audio = [_audio(i, i * 10.0 + 5.0, "spoken words here", 0.9) for i in range(6)]
    response = _uniform_mode(
        _request("what does the slide say"),
        visual,
        audio,
        budget=3,
        raw_candidate_count=12,
        raw_visual_count=6,
        raw_audio_count=6,
    )
    assert response.metrics.selected_candidates == 3
    assert response.metrics.budget_mode == "uniform"
    assert all(item.relevance_score == 0.0 for item in response.selected)
    # Raw-count token reduction is reported against the full candidate pool.
    assert response.metrics.estimated_token_reduction_percent > 0.0


def test_compress_modes_restrict_modality():
    visual = [
        _visual(0, 100.0, "on-screen text: Course Overview", 1.0),
        _visual(1, 900.0, "on-screen text: unrelated diagram", 0.2),
    ]
    audio = [_audio(0, 300.0, "welcome to the course overview lecture", 0.8)]
    compressor = GistCompressor()
    template = _request("what on-screen text says course overview")

    visual_only = _compress_mode(compressor, template, visual, [], 3, 2, 1)
    transcript_only = _compress_mode(compressor, template, [], audio, 3, 2, 1)
    full = _compress_mode(compressor, template, visual, audio, 3, 2, 1)

    assert all(item.modality.value == "visual" for item in visual_only.selected)
    assert all(item.modality.value == "audio" for item in transcript_only.selected)
    assert full.metrics.selected_candidates >= 1


def test_run_ablation_case_scores_full_gist_pass(tmp_path):
    # Build a synthetic committed-style artifact + candidate pool via monkey-free path:
    # exercise the scoring wiring by writing a full_gist response and evaluating it.
    visual = [_visual(0, 100.0, "on-screen text: Course Overview", 1.0)]
    compressor = GistCompressor()
    template = _request("what on-screen text says course overview")
    response = _compress_mode(compressor, template, visual, [], 100, 100, 0)

    mode_path = tmp_path / "full_gist" / "compression.json"
    mode_path.parent.mkdir(parents=True)
    mode_path.write_text(json.dumps({"compression": response.model_dump(mode="json")}))

    case = QualityCase(
        id="synthetic-course-overview",
        compression_path=mode_path,
        expected_answer_terms=["course", "overview"],
        expected_evidence_terms=["course", "overview"],
        relevant_ranges=[{"start_seconds": 100.0, "end_seconds": 101.0}],
        timestamp_tolerance_seconds=2,
        min_answer_term_recall=1.0,
        min_evidence_term_coverage=1.0,
        min_evidence_relevance_rate=1.0,
        min_timestamp_hit_rate=1.0,
        min_token_reduction_percent=99.0,
        max_selected_evidence=1,
        min_visual_evidence=1,
    )
    from gist.eval.quality import evaluate_quality_case

    result = evaluate_quality_case(case, output_root=tmp_path)
    assert result.passed
    assert result.timestamp_hit_rate == 1.0
    assert result.answer_term_recall == 1.0
    assert result.token_reduction_percent >= 99.0


def test_resolve_case_config_detects_whisper(tmp_path):
    artifact = tmp_path / "compression.json"
    artifact.write_text(
        json.dumps(
            {
                "ingestion": {
                    "source_path": "video.mp4",
                    "metadata": {"duration_seconds": 3660.0},
                },
                "compression": {
                    "query": "what does she say",
                    "preset": "aggressive",
                    "audio_scorer_used": None,
                    "transcript_metadata": {
                        "quality": "fast",
                        "model_size": "tiny",
                        "device": "cpu",
                        "compute_type": "int8",
                        "beam_size": 1,
                    },
                },
            }
        )
    )
    case = QualityCase(id="c", compression_path=artifact)
    config = resolve_case_config(case)
    assert config.audio_scorer.value == "whisper"
    assert config.whisper_model_size == "tiny"
    assert config.transcript_quality == "fast"
    assert config.whisper_beam_size == 1
    assert config.preset == CompressionPreset.AGGRESSIVE


def test_resolve_case_config_defaults_to_baseline(tmp_path):
    artifact = tmp_path / "compression.json"
    artifact.write_text(
        json.dumps(
            {
                "ingestion": {
                    "source_path": "video.ogv",
                    "metadata": {"duration_seconds": 5000.0},
                },
                "compression": {"query": "what slide", "preset": "balanced"},
            }
        )
    )
    case = QualityCase(id="c", compression_path=artifact)
    config = resolve_case_config(case)
    assert config.audio_scorer.value == "baseline"


def test_render_markdown_includes_mode_labels():
    outcome = ModeOutcome(
        mode="full_gist",
        passed=True,
        answer_term_recall=1.0,
        evidence_term_coverage=1.0,
        evidence_relevance_rate=1.0,
        timestamp_hit_rate=1.0,
        grounded_evidence_rate=1.0,
        token_reduction_percent=99.8,
        selected_evidence=1,
        visual_evidence=1,
        audio_evidence=0,
        failures=[],
    )
    outcomes = {mode: outcome.model_copy(update={"mode": mode}) for mode in MODES}
    result = AblationCaseResult(case_id="c", query_category="visual_object_action", outcomes=outcomes)

    from gist.eval.ablation import AblationReport

    summaries = {mode: _summarize(mode, [result]) for mode in MODES}
    report = AblationReport(cases=1, summaries=summaries, results=[result])
    markdown = render_ablation_markdown(report)
    assert "Full Gist (audio+visual)" in markdown
    assert "Uniform sampling" in markdown
    assert "| Case |" in markdown


def test_dataset_cases_load_for_ablation():
    dataset = Path("data/eval/long-video-quality.jsonl")
    cases = load_quality_cases(dataset)
    assert len(cases) >= 30
    assert all(case.compression_path is not None for case in cases)
