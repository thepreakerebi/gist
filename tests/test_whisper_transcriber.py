from pathlib import Path

import pytest

import gist.audio.whisper as whisper
from gist.audio.errors import AudioTranscriptionError
from gist.audio.whisper import (
    FasterWhisperTranscriber,
    TranscriptQuality,
    resolve_whisper_settings,
)
from gist.media.models import AudioWindow


def test_whisper_transcriber_returns_empty_mapping_without_windows() -> None:
    assert FasterWhisperTranscriber().transcribe_windows([]) == {}


def test_resolve_whisper_settings_uses_quality_presets() -> None:
    fast = resolve_whisper_settings(TranscriptQuality.FAST)
    balanced = resolve_whisper_settings(TranscriptQuality.BALANCED)
    accurate = resolve_whisper_settings(TranscriptQuality.ACCURATE)

    assert fast.model_size == "tiny"
    assert fast.beam_size == 1
    assert balanced.model_size == "base"
    assert balanced.beam_size == 3
    assert accurate.model_size == "small"
    assert accurate.beam_size == 5


def test_resolve_whisper_settings_allows_explicit_overrides() -> None:
    settings = resolve_whisper_settings(
        TranscriptQuality.FAST,
        model_size="medium",
        device="cuda",
        compute_type="float16",
        beam_size=7,
    )

    assert settings.cache_fingerprint == (
        "model=medium|device=cuda|compute=float16|beam=7"
    )


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
        def transcribe(
            self,
            *_args: object,
            **kwargs: object,
        ) -> tuple[list[FakeSegment], None]:
            calls["count"] += 1
            assert kwargs["beam_size"] == 3
            return [FakeSegment()], None

    monkeypatch.setattr(whisper, "_TRANSCRIPT_CACHE", {})
    transcriber = FasterWhisperTranscriber(beam_size=3)
    transcriber._model = FakeModel()
    window = AudioWindow(index=0, start_seconds=0.0, duration_seconds=1.0, path=audio_path)

    assert transcriber.transcribe_windows([window]) == {audio_path: "hello"}
    assert transcriber.transcribe_windows([window]) == {audio_path: "hello"}
    assert calls["count"] == 1


def test_whisper_transcriber_persists_window_transcripts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake")
    calls = {"count": 0}

    class FakeSegment:
        text = "cached hello"

    class FakeModel:
        def transcribe(self, *_args: object, **_kwargs: object) -> tuple[list[FakeSegment], None]:
            calls["count"] += 1
            return [FakeSegment()], None

    monkeypatch.setattr(whisper, "_TRANSCRIPT_CACHE", {})
    first = FasterWhisperTranscriber(cache_dir=tmp_path / "cache")
    first._model = FakeModel()
    window = AudioWindow(index=0, start_seconds=0.0, duration_seconds=1.0, path=audio_path)

    assert first.transcribe_windows([window]) == {audio_path: "cached hello"}

    monkeypatch.setattr(whisper, "_TRANSCRIPT_CACHE", {})
    second = FasterWhisperTranscriber(cache_dir=tmp_path / "cache")

    assert second.transcribe_windows([window]) == {audio_path: "cached hello"}
    assert second._model is None
    assert calls["count"] == 1
