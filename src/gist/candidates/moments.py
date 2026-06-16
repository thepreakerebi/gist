from gist.candidates.baseline import CandidateSet
from gist.core.answering import WHY_ANSWER_TERMS
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
    preserve_visual_context_audio: bool = False,
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
            scored_audio.append(
                (
                    _moment_relevance(query, audio, audio_relevance),
                    _audio_with_visual_grounding(audio, visual),
                )
            )
        else:
            scored_audio.append((_moment_relevance(query, audio, audio_relevance), audio))

    if preserve_visual_context_audio:
        _add_visual_context_audio(
            scored_audio=scored_audio,
            used_visual_ids=used_visual_ids,
            candidates=candidates,
            query=query,
            visual_radius_seconds=max(visual_radius_seconds, 30.0),
        )

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
        if visual.id not in used_visual_ids
        and lexical_relevance(query, visual) >= min_audio_relevance
    ]
    return CandidateSet(visual=orphan_visuals, audio=fused_audio)


def _add_visual_context_audio(
    scored_audio: list[tuple[float, Candidate]],
    used_visual_ids: set[str],
    candidates: CandidateSet,
    query: str,
    visual_radius_seconds: float,
) -> None:
    existing_audio_ids = {
        candidate.id.split("+", maxsplit=1)[0]
        for _score, candidate in scored_audio
    }
    scored_visuals = sorted(
        (
            (_visual_relevance(query, visual), visual)
            for visual in candidates.visual
        ),
        key=lambda item: (item[0], -item[1].timestamp_seconds),
        reverse=True,
    )
    if not scored_visuals:
        return

    minimum_visual_score = max(
        DEFAULT_MIN_AUDIO_RELEVANCE,
        scored_visuals[0][0] * 0.6,
    )
    relevant_visuals = [
        (score, visual)
        for score, visual in scored_visuals
        if score >= minimum_visual_score
    ]
    for visual_score, visual in relevant_visuals[:8]:
        nearby_audio = [
            audio
            for audio in candidates.audio
            if audio.id not in existing_audio_ids
            and abs(audio.timestamp_seconds - visual.timestamp_seconds)
            <= visual_radius_seconds
        ]
        if not nearby_audio:
            continue
        audio = min(
            nearby_audio,
            key=lambda item: abs(item.timestamp_seconds - visual.timestamp_seconds),
        )
        existing_audio_ids.add(audio.id)
        used_visual_ids.add(visual.id)
        scored_audio.append(
            (
                visual_score,
                _audio_with_visual_grounding(audio, visual),
            )
        )


def _visual_relevance(query: str, visual: Candidate) -> float:
    return max(
        lexical_relevance(query, visual),
        float(visual.saliency_score or 0.0),
    )


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


def _moment_relevance(query: str, audio: Candidate, base_relevance: float) -> float:
    if not query.lower().strip().startswith("why"):
        return base_relevance
    text = audio.text.lower()
    answer_signal = sum(0.2 for term in WHY_ANSWER_TERMS if term in text)
    return base_relevance + answer_signal


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
