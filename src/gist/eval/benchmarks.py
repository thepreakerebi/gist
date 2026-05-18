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

    def to_eval_example(self) -> EvalExample:
        return EvalExample(
            id=self.id,
            video_id=self.video_id,
            query=self.query,
            duration_seconds=self.duration_seconds,
            video_path=self.video_path,
            relevant_timestamps=self.relevant_timestamps,
        )


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
