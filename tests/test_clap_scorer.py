from pathlib import Path

import pytest

from gist.audio.clap import HuggingFaceClapAudioScorer
from gist.audio.errors import AudioTranscriptionError
from gist.media.models import AudioWindow


def test_clap_scorer_requires_non_blank_query(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake")

    with pytest.raises(ValueError, match="query must not be blank"):
        HuggingFaceClapAudioScorer().score_windows(
            [
                AudioWindow(
                    index=0,
                    start_seconds=0,
                    duration_seconds=1,
                    path=audio_path,
                )
            ],
            " ",
        )


def test_clap_scorer_reports_missing_optional_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake")
    original_import = __import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch":
            raise ImportError("missing torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(AudioTranscriptionError, match="optional sound dependencies"):
        HuggingFaceClapAudioScorer().score_windows(
            [
                AudioWindow(
                    index=0,
                    start_seconds=0,
                    duration_seconds=1,
                    path=audio_path,
                )
            ],
            "applause",
        )
