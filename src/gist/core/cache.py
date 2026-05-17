from dataclasses import dataclass, field

from gist.core.schemas import Candidate


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

