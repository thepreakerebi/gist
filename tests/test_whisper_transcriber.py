from pathlib import Path

import pytest

from gist.audio.errors import AudioTranscriptionError
import gist.audio.whisper as whisper
from gist.audio.whisper import FasterWhisperTranscriber
from gist.media.models import AudioWindow


def test_whisper_transcriber_returns_empty_mapping_without_windows() -> None:
    assert FasterWhisperTranscriber().transcribe_windows([]) == {}


def test_whisper_transcriber_reports_missing_optional_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake")
    original_import = __import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "faster_whisper":
            raise ImportError("missing faster-whisper")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(AudioTranscriptionError, match="optional audio dependencies"):
        FasterWhisperTranscriber().transcribe_windows(
            [
                AudioWindow(
                    index=0,
                    start_seconds=0.0,
                    duration_seconds=1.0,
                    path=audio_path,
                )
            ]
        )


def test_whisper_transcriber_reuses_cached_window_transcripts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake")
    calls = {"count": 0}

    class FakeSegment:
        text = "hello"

    class FakeModel:
        def transcribe(self, *_args: object, **_kwargs: object) -> tuple[list[FakeSegment], None]:
            calls["count"] += 1
            return [FakeSegment()], None

    monkeypatch.setattr(whisper, "_TRANSCRIPT_CACHE", {})
    transcriber = FasterWhisperTranscriber()
    transcriber._model = FakeModel()
    window = AudioWindow(index=0, start_seconds=0.0, duration_seconds=1.0, path=audio_path)

    assert transcriber.transcribe_windows([window]) == {audio_path: "hello"}
    assert transcriber.transcribe_windows([window]) == {audio_path: "hello"}
    assert calls["count"] == 1
