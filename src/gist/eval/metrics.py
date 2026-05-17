from gist.core.schemas import Modality, SelectedCandidate


def timestamp_hit_rate(
    selected: list[SelectedCandidate],
    relevant_timestamps: list[float],
    tolerance_seconds: float,
) -> float:
    if not relevant_timestamps:
        return 0.0

    selected_timestamps = [item.timestamp_seconds for item in selected]
    hits = sum(
        any(abs(selected - relevant) <= tolerance_seconds for selected in selected_timestamps)
        for relevant in relevant_timestamps
    )
    return hits / len(relevant_timestamps)


def reduction_percent(input_candidates: int, selected_candidates: int) -> float:
    if input_candidates == 0:
        return 0.0
    return (1.0 - (selected_candidates / input_candidates)) * 100


def modality_coverage(selected: list[SelectedCandidate]) -> dict[Modality, int]:
    return {
        Modality.VISUAL: sum(item.modality == Modality.VISUAL for item in selected),
        Modality.AUDIO: sum(item.modality == Modality.AUDIO for item in selected),
    }

