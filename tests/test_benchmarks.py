from pathlib import Path

from gist.core.modes import VisualScoringMode
from gist.eval.benchmarks import (
    BenchmarkName,
    SOTA_BENCHMARK_VARIANTS,
    load_benchmark_jsonl,
)


def test_load_benchmark_jsonl_accepts_video_mme_style_rows(tmp_path: Path) -> None:
    path = tmp_path / "video_mme.jsonl"
    path.write_text(
        "\n".join(
            [
                (
                    '{"question_id":"q1","video_id":"video-1","question":"What happens",'
                    '"duration":120,"options":["A","B"],"answer":"A",'
                    '"relevant_timestamps":[12,18]}'
                )
            ]
        )
    )

    examples = load_benchmark_jsonl(path, BenchmarkName.VIDEO_MME)

    assert len(examples) == 1
    assert examples[0].id == "q1"
    assert examples[0].query == "What happens"
    assert examples[0].choices == ["A", "B"]
    assert examples[0].answer == "A"
    assert examples[0].to_eval_example().relevant_timestamps == [12.0, 18.0]


def test_sota_benchmark_variants_include_scene_and_spatial_runs() -> None:
    variants = {variant.name: variant for variant in SOTA_BENCHMARK_VARIANTS}

    assert variants["gist_core"].visual_scorer == VisualScoringMode.BASELINE
    assert variants["gist_scene_clip"].visual_scorer == VisualScoringMode.CLIP_SCENE
    assert variants["gist_scene_spatial"].spatial_pruning is True
