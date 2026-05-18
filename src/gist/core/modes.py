from enum import StrEnum


class VisualScoringMode(StrEnum):
    BASELINE = "baseline"
    CLIP = "clip"
    CLIP_SCENE = "clip_scene"


class AudioScoringMode(StrEnum):
    BASELINE = "baseline"
    WHISPER = "whisper"
    CLAP = "clap"
