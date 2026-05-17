from gist.core.cache import CandidateCache
from gist.core.schemas import Candidate


def test_candidate_cache_stores_visual_and_audio_candidates_separately() -> None:
    cache = CandidateCache()
    visual = [Candidate(id="v1", timestamp_seconds=1, text="speaker enters")]
    audio = [Candidate(id="a1", timestamp_seconds=2, text="speaker talks")]

    cache.set_visual("video-1", visual)
    cache.set_audio("video-1", audio)

    assert cache.get_visual("video-1") == visual
    assert cache.get_audio("video-1") == audio

