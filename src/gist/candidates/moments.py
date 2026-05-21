from gist.candidates.baseline import CandidateSet
from gist.core.schemas import Candidate
from gist.core.scoring import lexical_relevance


DEFAULT_VISUAL_RADIUS_SECONDS = 12.0
DEFAULT_MIN_AUDIO_RELEVANCE = 0.12
DEFAULT_MAX_AUDIO_MOMENTS = 8


def fuse_transcript_moments(
    candidates: CandidateSet,
    query: str,
    visual_radius_seconds: float = DEFAULT_VISUAL_RADIUS_SECONDS,
    min_audio_relevance: float = DEFAULT_MIN_AUDIO_RELEVANCE,
    max_audio_moments: int = DEFAULT_MAX_AUDIO_MOMENTS,
) -> CandidateSet:
    """Build transcript-centered evidence moments with nearby visual grounding.

    Long videos should not expose audio and visual candidates as unrelated units.
    This keeps transcript evidence as the selectable unit, but attaches the best
    nearby video frame/scene metadata so downstream clip rendering and reports
    behave like moment selection.
    """

    if visual_radius_seconds <= 0:
        raise ValueError("visual_radius_seconds must be greater than zero")
    if min_audio_relevance < 0:
        raise ValueError("min_audio_relevance must be non-negative")
    if max_audio_moments <= 0:
        raise ValueError("max_audio_moments must be greater than zero")

    scored_audio: list[tuple[float, Candidate]] = []
    used_visual_ids: set[str] = set()
    for audio in candidates.audio:
        audio_relevance = lexical_relevance(query, audio)
        if audio_relevance < min_audio_relevance:
            continue
        visual = _nearest_visual(audio, candidates.visual, visual_radius_seconds)
        if visual is not None:
            used_visual_ids.add(visual.id)
            scored_audio.append((audio_relevance, _audio_with_visual_grounding(audio, visual)))
        else:
            scored_audio.append((audio_relevance, audio))

    fused_audio = [
        candidate
        for _score, candidate in sorted(
            scored_audio,
            key=lambda item: (item[0], -item[1].timestamp_seconds),
            reverse=True,
        )[:max_audio_moments]
    ]

    # Keep high-saliency orphan visuals as a fallback for genuinely visual questions.
    orphan_visuals = [
        visual
        for visual in candidates.visual
        if visual.id not in used_visual_ids and lexical_relevance(query, visual) >= min_audio_relevance
    ]
    return CandidateSet(visual=orphan_visuals, audio=fused_audio)


def _nearest_visual(
    audio: Candidate,
    visuals: list[Candidate],
    radius_seconds: float,
) -> Candidate | None:
    nearby = [
        visual
        for visual in visuals
        if abs(visual.timestamp_seconds - audio.timestamp_seconds) <= radius_seconds
    ]
    if not nearby:
        return None
    return min(
        nearby,
        key=lambda visual: (
            abs(visual.timestamp_seconds - audio.timestamp_seconds),
            -float(visual.saliency_score or 0.0),
        ),
    )


def _audio_with_visual_grounding(audio: Candidate, visual: Candidate) -> Candidate:
    return audio.model_copy(
        update={
            "id": f"{audio.id}+{visual.id}",
            "asset_path": visual.asset_path or audio.asset_path,
            "spatial_mask_path": visual.spatial_mask_path,
            "segment_id": visual.segment_id or audio.segment_id,
            "scene_start_seconds": audio.scene_start_seconds,
            "scene_end_seconds": audio.scene_end_seconds,
        }
    )
