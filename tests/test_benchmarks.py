from pathlib import Path

from gist.core.modes import AudioScoringMode, VisualScoringMode
from gist.eval.benchmarks import (
    BenchmarkName,
    SOTA_BENCHMARK_VARIANTS,
    benchmark_readiness_issues,
    load_benchmark_jsonl,
    resolve_benchmark_video_paths,
    write_benchmark_jsonl,
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
    assert examples[0].to_eval_example().expected_answer == "A"
    assert examples[0].to_eval_example().choices == ["A", "B"]
    assert examples[0].to_eval_example().sample_count == 128
    assert examples[0].to_eval_example().relevant_timestamps == [12.0, 18.0]


def test_sota_benchmark_variants_include_scene_and_spatial_runs() -> None:
    variants = {variant.name: variant for variant in SOTA_BENCHMARK_VARIANTS}

    assert variants["gist_core"].visual_scorer == VisualScoringMode.BASELINE
    assert variants["gist_scene_clip"].visual_scorer == VisualScoringMode.CLIP_SCENE
    assert variants["gist_scene_router_adaptive_whisper"].audio_scorer == AudioScoringMode.WHISPER
    assert variants["gist_task_router_adaptive_whisper"].task_aware_selection is True
    assert variants["gist_scene_spatial"].spatial_pruning is True


def test_resolve_benchmark_video_paths_from_video_root(tmp_path: Path) -> None:
    dataset = tmp_path / "video_mme.jsonl"
    video_root = tmp_path / "videos"
    video_root.mkdir()
    video_path = video_root / "video-1.mp4"
    video_path.write_bytes(b"video")
    dataset.write_text(
        '{"id":"q1","video_id":"video-1","query":"What happens",'
        '"duration_seconds":10,"answer":"A"}\n'
    )
    examples = load_benchmark_jsonl(dataset, BenchmarkName.VIDEO_MME)

    resolved = resolve_benchmark_video_paths(examples, video_root)

    assert resolved[0].video_path == video_path
    assert benchmark_readiness_issues(resolved) == []


def test_write_benchmark_jsonl_preserves_resolved_paths(tmp_path: Path) -> None:
    dataset = tmp_path / "video_mme.jsonl"
    output = tmp_path / "prepared.jsonl"
    dataset.write_text(
        '{"id":"q1","video_id":"video-1","query":"What happens",'
        '"duration_seconds":10,"video_path":"video.mp4","answer":"A"}\n'
    )
    examples = load_benchmark_jsonl(dataset, BenchmarkName.VIDEO_MME)

    write_benchmark_jsonl(examples, output)

    assert '"video_path": "video.mp4"' in output.read_text()
    assert '"sample_count": 128' in output.read_text()
