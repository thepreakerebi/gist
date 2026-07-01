from gist.core.schemas import Modality, SelectedCandidate


def visual_evidence_count(selected: list[SelectedCandidate]) -> int:
    return sum(_is_visual_evidence(item) for item in selected)


def audio_evidence_count(selected: list[SelectedCandidate]) -> int:
    return sum(item.modality == Modality.AUDIO for item in selected)


def _is_visual_evidence(item: SelectedCandidate) -> bool:
    if item.modality == Modality.VISUAL:
        return True
    if item.modality != Modality.AUDIO:
        return False
    item_id = item.id.lower()
    return ":visual:" in item_id or "+visual" in item_id
