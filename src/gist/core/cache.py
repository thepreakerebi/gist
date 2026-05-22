from dataclasses import dataclass, field
from pathlib import Path

from gist.candidates.baseline import CandidateSet
from gist.core.modes import AudioScoringMode, VisualScoringMode
from gist.core.schemas import Candidate
from gist.media.models import IngestedVideo


CANDIDATE_CACHE_VERSION = "v9"


@dataclass(slots=True)
class CandidateCache:
    _visual: dict[str, list[Candidate]] = field(default_factory=dict)
    _audio: dict[str, list[Candidate]] = field(default_factory=dict)

    def get_visual(self, video_id: str) -> list[Candidate] | None:
        return self._visual.get(video_id)

    def set_visual(self, video_id: str, candidates: list[Candidate]) -> None:
        self._visual[video_id] = candidates

    def get_audio(self, video_id: str) -> list[Candidate] | None:
        return self._audio.get(video_id)

    def set_audio(self, video_id: str, candidates: list[Candidate]) -> None:
        self._audio[video_id] = candidates


class DiskCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    def get_ingestion(self, key: str) -> IngestedVideo | None:
        path = self._path("ingestions", key)
        if not path.exists():
            return None
        return IngestedVideo.model_validate_json(path.read_text())

    def set_ingestion(self, key: str, ingestion: IngestedVideo) -> None:
        path = self._path("ingestions", key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(ingestion.model_dump_json(indent=2))

    def get_candidates(self, key: str) -> CandidateSet | None:
        path = self._path("candidates", key)
        if not path.exists():
            return None
        payload = path.read_text()
        return CandidateSet.model_validate_json(payload)

    def set_candidates(self, key: str, candidates: CandidateSet) -> None:
        path = self._path("candidates", key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(candidates.model_dump_json(indent=2))

    def _path(self, namespace: str, key: str) -> Path:
        return self.root / namespace / f"{key}.json"


def ingestion_cache_key(
    video_path: Path,
    sample_count: int | None,
    audio_window_seconds: float | None,
    processing_mode: str = "short",
) -> str:
    absolute_path = video_path.expanduser().resolve(strict=False)
    fingerprint = (
        f"{absolute_path}|mode={processing_mode}|samples={sample_count}|"
        f"audio_window={audio_window_seconds}"
    )
    return _sha256_short(fingerprint)


def candidate_cache_key(
    ingestion: IngestedVideo,
    query: str,
    visual_scorer: VisualScoringMode,
    audio_scorer: AudioScoringMode,
    audio_context_window_count: int | None = None,
    visual_ocr: bool = False,
) -> str:
    context_count = (
        audio_context_window_count
        if audio_context_window_count is not None
        else (
            ingestion.settings.audio_context_window_count
            if ingestion.settings is not None
            else 1
        )
    )
    fingerprint = (
        f"{CANDIDATE_CACHE_VERSION}|{ingestion.video_id}|query={query.strip()}|"
        f"visual={visual_scorer}|audio={audio_scorer}|audio_context={context_count}|"
        f"visual_ocr={visual_ocr}"
    )
    return _sha256_short(fingerprint)


def _sha256_short(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
