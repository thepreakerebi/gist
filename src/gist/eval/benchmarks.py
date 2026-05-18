from enum import StrEnum
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from gist.core.modes import VisualScoringMode
from gist.core.presets import CompressionPreset
from gist.eval.schemas import EvalExample, EvalVariant


class BenchmarkName(StrEnum):
    VIDEO_MME = "video_mme"
    LONG_VIDEO_BENCH = "long_video_bench"
    MLVU = "mlvu"
    EGO_SCHEMA = "egoschema"


class BenchmarkExample(BaseModel):
    benchmark: BenchmarkName
    id: str
    video_id: str
    query: str
    duration_seconds: float = Field(gt=0)
    video_path: Path | None = None
    choices: list[str] = Field(default_factory=list)
    answer: str | None = None
    relevant_timestamps: list[float] = Field(default_factory=list)
    sample_count: int = 128
    audio_window_seconds: float = 1.0

    def to_eval_example(self) -> EvalExample:
        return EvalExample(
            id=self.id,
            video_id=self.video_id,
            query=self.query,
            duration_seconds=self.duration_seconds,
            video_path=self.video_path,
            relevant_timestamps=self.relevant_timestamps,
            expected_answer=self.answer,
            choices=self.choices,
            sample_count=self.sample_count,
            audio_window_seconds=self.audio_window_seconds,
        )

    def with_video_path(self, video_path: Path | None) -> "BenchmarkExample":
        return self.model_copy(update={"video_path": video_path})

    def with_ingestion_settings(
        self,
        sample_count: int | None = None,
        audio_window_seconds: float | None = None,
    ) -> "BenchmarkExample":
        updates = {}
        if sample_count is not None:
            updates["sample_count"] = sample_count
        if audio_window_seconds is not None:
            updates["audio_window_seconds"] = audio_window_seconds
        return self.model_copy(update=updates)


SOTA_BENCHMARK_VARIANTS = [
    EvalVariant(
        name="gist_core",
        preset=CompressionPreset.BALANCED,
    ),
    EvalVariant(
        name="gist_scene_clip",
        preset=CompressionPreset.BALANCED,
        visual_scorer=VisualScoringMode.CLIP_SCENE,
    ),
    EvalVariant(
        name="gist_scene_router_adaptive",
        preset=CompressionPreset.BALANCED,
        visual_scorer=VisualScoringMode.CLIP_SCENE,
        adaptive_budget=True,
    ),
    EvalVariant(
        name="gist_scene_spatial",
        preset=CompressionPreset.BALANCED,
        visual_scorer=VisualScoringMode.CLIP_SCENE,
        adaptive_budget=True,
        spatial_pruning=True,
    ),
]


def load_benchmark_jsonl(path: Path, benchmark: BenchmarkName) -> list[BenchmarkExample]:
    examples: list[BenchmarkExample] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
        examples.append(_benchmark_example_from_record(record, benchmark, line_number))
    return examples


def resolve_benchmark_video_paths(
    examples: list[BenchmarkExample],
    video_root: Path,
    extensions: tuple[str, ...] = (".mp4", ".mov", ".mkv", ".webm"),
) -> list[BenchmarkExample]:
    return [
        example.with_video_path(
            example.video_path if example.video_path is not None else _find_video_path(
                video_root=video_root,
                video_id=example.video_id,
                extensions=extensions,
            )
        )
        for example in examples
    ]


def benchmark_readiness_issues(examples: list[BenchmarkExample]) -> list[str]:
    issues: list[str] = []
    for example in examples:
        if example.video_path is None:
            issues.append(f"{example.id}: missing video_path")
        elif not example.video_path.exists():
            issues.append(f"{example.id}: video file does not exist: {example.video_path}")
        if example.answer is None:
            issues.append(f"{example.id}: missing expected answer")
    return issues


def write_benchmark_jsonl(examples: list[BenchmarkExample], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "id": example.id,
                "video_id": example.video_id,
                "query": example.query,
                "duration_seconds": example.duration_seconds,
                "video_path": str(example.video_path) if example.video_path else None,
                "choices": example.choices,
                "answer": example.answer,
                "relevant_timestamps": example.relevant_timestamps,
                "sample_count": example.sample_count,
                "audio_window_seconds": example.audio_window_seconds,
            }
        )
        for example in examples
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return path


def _benchmark_example_from_record(
    record: dict[str, Any],
    benchmark: BenchmarkName,
    line_number: int,
) -> BenchmarkExample:
    query = _first_string(record, "query", "question", "prompt")
    video_id = _first_string(record, "video_id", "video", "video_name", "youtube_id")
    identifier = _first_string(record, "id", "question_id", "uid") or f"{benchmark}-{line_number}"
    duration = _first_float(record, "duration_seconds", "duration", "video_duration")
    if query is None:
        raise ValueError(f"benchmark row {line_number} is missing query/question")
    if video_id is None:
        raise ValueError(f"benchmark row {line_number} is missing video_id/video")
    if duration is None:
        raise ValueError(f"benchmark row {line_number} is missing duration_seconds/duration")

    return BenchmarkExample(
        benchmark=benchmark,
        id=identifier,
        video_id=video_id,
        query=query,
        duration_seconds=duration,
        video_path=_optional_path(record.get("video_path") or record.get("path")),
        choices=_choices(record),
        answer=_first_string(record, "answer", "correct_answer", "label"),
        relevant_timestamps=_timestamps(record),
        sample_count=int(_first_float(record, "sample_count") or 128),
        audio_window_seconds=_first_float(record, "audio_window_seconds") or 1.0,
    )


def _first_string(record: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _first_float(record: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    return None


def _optional_path(value: Any) -> Path | None:
    if isinstance(value, str) and value.strip():
        return Path(value)
    return None


def _find_video_path(
    video_root: Path,
    video_id: str,
    extensions: tuple[str, ...],
) -> Path | None:
    direct = Path(video_id)
    if direct.exists():
        return direct

    for extension in extensions:
        candidate = video_root / f"{video_id}{extension}"
        if candidate.exists():
            return candidate

    for path in video_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions and path.stem == video_id:
            return path
    return None


def _choices(record: dict[str, Any]) -> list[str]:
    choices = record.get("choices") or record.get("options")
    if isinstance(choices, list):
        return [str(choice) for choice in choices]
    return []


def _timestamps(record: dict[str, Any]) -> list[float]:
    timestamps = record.get("relevant_timestamps") or record.get("timestamps") or []
    if not isinstance(timestamps, list):
        return []
    parsed: list[float] = []
    for timestamp in timestamps:
        if isinstance(timestamp, int | float):
            parsed.append(float(timestamp))
    return parsed
