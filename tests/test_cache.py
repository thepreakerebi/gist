from gist.core.cache import CandidateCache
from gist.candidates.baseline import CandidateSet
from gist.core.cache import (
    DiskCache,
    candidate_cache_key,
    ingestion_cache_key,
)
from gist.core.modes import AudioScoringMode, VisualScoringMode
from gist.core.schemas import Candidate
from gist.media.models import IngestedVideo, VideoMetadata


def test_candidate_cache_stores_visual_and_audio_candidates_separately() -> None:
    cache = CandidateCache()
    visual = [Candidate(id="v1", timestamp_seconds=1, text="speaker enters")]
    audio = [Candidate(id="a1", timestamp_seconds=2, text="speaker talks")]

    cache.set_visual("video-1", visual)
    cache.set_audio("video-1", audio)

    assert cache.get_visual("video-1") == visual
    assert cache.get_audio("video-1") == audio


def test_disk_cache_round_trips_ingestion_manifest(tmp_path) -> None:
    cache = DiskCache(tmp_path)
    ingestion = IngestedVideo(
        video_id="video-1",
        source_path=tmp_path / "video.mp4",
        metadata=VideoMetadata(duration_seconds=1.0),
        frames=[],
        audio_windows=[],
    )

    cache.set_ingestion("key", ingestion)

    assert cache.get_ingestion("key") == ingestion


def test_disk_cache_round_trips_candidate_set(tmp_path) -> None:
    cache = DiskCache(tmp_path)
    candidates = CandidateSet(
        visual=[Candidate(id="v1", timestamp_seconds=0, text="frame")],
        audio=[Candidate(id="a1", timestamp_seconds=0, text="audio")],
    )

    cache.set_candidates("key", candidates)

    assert cache.get_candidates("key") == candidates


def test_cache_keys_change_when_parameters_change(tmp_path) -> None:
    video_path = tmp_path / "video.mp4"
    ingestion = IngestedVideo(
        video_id="video-1",
        source_path=video_path,
        metadata=VideoMetadata(duration_seconds=1.0),
        frames=[],
        audio_windows=[],
    )

    assert ingestion_cache_key(video_path, 4, 1.0) != ingestion_cache_key(video_path, 8, 1.0)
    assert candidate_cache_key(
        ingestion,
        "pricing",
        VisualScoringMode.BASELINE,
        AudioScoringMode.BASELINE,
    ) != candidate_cache_key(
        ingestion,
        "pricing",
        VisualScoringMode.CLIP,
        AudioScoringMode.BASELINE,
    )
    assert candidate_cache_key(
        ingestion,
        "pricing",
        VisualScoringMode.BASELINE,
        AudioScoringMode.WHISPER,
    ) != candidate_cache_key(
        ingestion,
        "pricing",
        VisualScoringMode.BASELINE,
        AudioScoringMode.CLAP,
    )
