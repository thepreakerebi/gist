from gist.core.schemas import Modality, SelectedCandidate
from gist.eval.metrics import modality_coverage, reduction_percent, timestamp_hit_rate


def test_timestamp_hit_rate_counts_relevant_timestamp_matches() -> None:
    selected = [
        SelectedCandidate(
            id="v1",
            modality=Modality.VISUAL,
            timestamp_seconds=10,
            text="frame",
            selection_rank=1,
            relevance_score=1,
            normalized_score=1,
            mmr_score=1,
            source_score_type="test",
            reason="test",
        )
    ]

    assert timestamp_hit_rate(selected, [8, 30], tolerance_seconds=3) == 0.5


def test_reduction_percent_and_modality_coverage() -> None:
    selected = [
        SelectedCandidate(
            id="a1",
            modality=Modality.AUDIO,
            timestamp_seconds=0,
            text="audio",
            selection_rank=1,
            relevance_score=1,
            normalized_score=1,
            mmr_score=1,
            source_score_type="test",
            reason="test",
        )
    ]

    assert reduction_percent(10, 4) == 60
    assert modality_coverage(selected) == {Modality.VISUAL: 0, Modality.AUDIO: 1}

