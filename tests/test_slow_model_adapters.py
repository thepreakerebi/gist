import subprocess
from pathlib import Path

import pytest
from PIL import Image

from gist.audio.clap import HuggingFaceClapAudioScorer
from gist.audio.whisper import FasterWhisperTranscriber
from gist.media.models import AudioWindow, ExtractedFrame
from gist.vision.clip import HuggingFaceClipFrameScorer
from slow_utils import require_slow_model_tests


@pytest.mark.slow
def test_slow_clip_scores_generated_frame(tmp_path: Path) -> None:
    require_slow_model_tests("torch", "transformers", "PIL")
    frame_path = tmp_path / "frame.jpg"
    Image.new("RGB", (64, 64), color="red").save(frame_path)

    scores = HuggingFaceClipFrameScorer(batch_size=1).score_frames(
        [ExtractedFrame(index=0, timestamp_seconds=0, path=frame_path)],
        "red image",
    )

    assert frame_path in scores
    assert isinstance(scores[frame_path], float)


@pytest.mark.slow
def test_slow_whisper_transcribes_generated_audio(tmp_path: Path) -> None:
    require_slow_model_tests("faster_whisper")
    audio_path = _generate_sine_wav(tmp_path)

    transcripts = FasterWhisperTranscriber(model_size="tiny").transcribe_windows(
        [AudioWindow(index=0, start_seconds=0, duration_seconds=1, path=audio_path)]
    )

    assert audio_path in transcripts


@pytest.mark.slow
def test_slow_clap_scores_generated_audio(tmp_path: Path) -> None:
    require_slow_model_tests("torch", "transformers", "numpy")
    audio_path = _generate_sine_wav(tmp_path)

    scores = HuggingFaceClapAudioScorer(batch_size=1).score_windows(
        [AudioWindow(index=0, start_seconds=0, duration_seconds=1, path=audio_path)],
        "sine tone",
    )

    assert audio_path in scores
    assert isinstance(scores[audio_path], float)


def _generate_sine_wav(tmp_path: Path) -> Path:
    audio_path = tmp_path / "tone.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(audio_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return audio_path
