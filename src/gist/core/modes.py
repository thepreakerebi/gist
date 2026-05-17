from enum import StrEnum


class VisualScoringMode(StrEnum):
    BASELINE = "baseline"
    CLIP = "clip"


class AudioScoringMode(StrEnum):
    BASELINE = "baseline"
    WHISPER = "whisper"
