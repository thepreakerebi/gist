import numpy as np

from gist.audio.clap import HuggingFaceClapAudioScorer


def _scorer_with_numpy() -> HuggingFaceClapAudioScorer:
    scorer = HuggingFaceClapAudioScorer()
    scorer._numpy = np  # avoid loading the real model for the pure-numpy helper
    return scorer


def test_resample_16k_to_48k_triples_length():
    scorer = _scorer_with_numpy()
    audio = np.sin(np.linspace(0, 6.28, 16000, dtype=np.float32))  # 1s @ 16k
    out = scorer._resample_48k(audio, orig_sr=16000)
    assert abs(len(out) - 48000) <= 1  # ~3x upsample
    assert out.dtype == np.float32


def test_resample_noop_when_already_48k_or_empty():
    scorer = _scorer_with_numpy()
    audio = np.ones(48000, dtype=np.float32)
    assert scorer._resample_48k(audio, orig_sr=48000) is audio
    empty = np.array([], dtype=np.float32)
    assert len(scorer._resample_48k(empty, orig_sr=16000)) == 0
