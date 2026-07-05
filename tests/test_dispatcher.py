from pathlib import Path

import numpy as np

from gist.audio.dispatcher import SpeechSoundDispatcher
from gist.media.models import AudioWindow


def _window(index: int) -> AudioWindow:
    return AudioWindow(
        index=index, start_seconds=float(index * 30), duration_seconds=30.0,
        path=Path(f"w{index}.wav"),
    )


class FakeClap:
    """Returns [speech_probe, ambient_probe, sound_query] per window by index.

    Window 0 = speech-dominant (speech>ambient); window 1 = sound-dominant.
    """

    def score_windows_against(self, windows, prompts):
        rows = {0: [0.80, 0.20, 0.30], 1: [0.10, 0.60, 0.75]}
        return {w.path: rows[w.index] for w in windows}


class FakeTranscriber:
    def transcribe_windows(self, windows):
        return {w.path: "the professor explains behavioral finance" for w in windows}


class FakeEmbedder:
    """query -> [1,0]; transcript -> [0.6,0.8] so cosine(query,transcript)=0.6."""

    def encode(self, texts):
        vecs = []
        for t in texts:
            vecs.append([1.0, 0.0] if "?" in t or t.islower() is False else [0.6, 0.8])
        # first text is the query (has '?'); rest are transcripts
        out = [[1.0, 0.0]] + [[0.6, 0.8]] * (len(texts) - 1)
        return np.array(out, dtype=float)


def test_dispatcher_routes_speech_to_whisper_embedding_and_sound_to_clap():
    disp = SpeechSoundDispatcher(
        clap=FakeClap(), transcriber=FakeTranscriber(), embedder=FakeEmbedder(),
        speech_threshold=0.0,
    )
    windows = [_window(0), _window(1)]
    scores = disp.score_windows(windows, "what is the topic?")

    # window 0: speech (0.80 > 0.20) -> transcript embedding cosine with query = 0.6
    assert disp.last_routing[windows[0].path] == "speech"
    assert abs(scores[windows[0].path] - 0.6) < 1e-6
    # window 1: sound (0.10 < 0.60) -> CLAP sound-query score = 0.75
    assert disp.last_routing[windows[1].path] == "sound"
    assert abs(scores[windows[1].path] - 0.75) < 1e-6


def test_dispatcher_speech_window_without_transcript_keeps_clap_fallback():
    class EmptyTranscriber:
        def transcribe_windows(self, windows):
            return {w.path: "" for w in windows}

    disp = SpeechSoundDispatcher(
        clap=FakeClap(), transcriber=EmptyTranscriber(), embedder=FakeEmbedder(),
    )
    windows = [_window(0)]
    scores = disp.score_windows(windows, "what is the topic?")
    # routed speech but no transcript -> retains the CLAP sound-query provisional (0.30)
    assert abs(scores[windows[0].path] - 0.30) < 1e-6


def test_dispatcher_empty_and_blank_query():
    disp = SpeechSoundDispatcher(clap=FakeClap(), transcriber=FakeTranscriber(), embedder=FakeEmbedder())
    assert disp.score_windows([], "q") == {}
    import pytest
    with pytest.raises(ValueError):
        disp.score_windows([_window(0)], "   ")
