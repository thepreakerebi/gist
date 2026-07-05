from enum import StrEnum


class VisualScoringMode(StrEnum):
    BASELINE = "baseline"
    CLIP = "clip"
    CLIP_SCENE = "clip_scene"


class AudioScoringMode(StrEnum):
    AUTO = "auto"
    BASELINE = "baseline"
    WHISPER = "whisper"
    CLAP = "clap"
    # Per-window speech-vs-sound routing (CLAP probe -> Whisper or CLAP scoring).
    DISPATCHER = "dispatcher"
