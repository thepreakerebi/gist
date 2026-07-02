from dataclasses import dataclass
from pathlib import Path
import re

from pydantic import BaseModel, ConfigDict

from gist.audio.scorers import AudioWindowScorer
from gist.audio.transcribers import AudioTranscriber
from gist.core.progress import ProgressCallback
from gist.core.schemas import Candidate
from gist.core.scoring import text_similarity
from gist.core.temporal_query import parse_temporal_query
from gist.media.models import AudioWindow, ExtractedFrame, IngestedVideo
from gist.vision.ocr_protocol import FrameOcr
from gist.vision.scene import SceneSegment, detect_scene_segments, scene_by_frame_index
from gist.vision.scorers import VisualFrameScorer

_COMMON_TRANSCRIPT_WORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "as",
    "because",
    "but",
    "can",
    "different",
    "for",
    "from",
    "have",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "their",
    "they",
    "this",
    "to",
    "we",
    "what",
    "with",
    "you",
}


class CandidateSet(BaseModel):
    model_config = ConfigDict(frozen=True)

    visual: list[Candidate]
    audio: list[Candidate]


@dataclass(frozen=True, slots=True)
class AudioTranscriptContext:
    text: str | None
    start_seconds: float | None
    end_seconds: float | None


class BaselineCandidateGenerator:
    """Build deterministic timestamp candidates before model-based scoring is available."""

    def __init__(
        self,
        visual_scorer: VisualFrameScorer | None = None,
        audio_transcriber: AudioTranscriber | None = None,
        audio_scorer: AudioWindowScorer | None = None,
        frame_ocr: FrameOcr | None = None,
        audio_context_window_count: int = 1,
        scene_aware_visuals: bool = False,
    ) -> None:
        if audio_context_window_count < 0:
            raise ValueError("audio_context_window_count must be non-negative")
        self.visual_scorer = visual_scorer
        self.audio_transcriber = audio_transcriber
        self.audio_scorer = audio_scorer
        self.frame_ocr = frame_ocr
        self.audio_context_window_count = audio_context_window_count
        self.scene_aware_visuals = scene_aware_visuals

    def generate(
        self,
        ingested_video: IngestedVideo,
        query: str,
        progress: ProgressCallback | None = None,
    ) -> CandidateSet:
        if progress is not None:
            progress("scoring visual frames")
        visual_scores = self._score_visual_frames(ingested_video, query)
        temporal_anchor_scores, temporal_target_scores = self._score_temporal_frames(
            ingested_video,
            query,
        )
        temporal_query = parse_temporal_query(query)
        if progress is not None:
            progress("detecting visual scenes")
        visual_scenes = self._scene_segments(ingested_video, visual_scores)
        scene_by_frame = scene_by_frame_index(visual_scenes)
        if progress is not None:
            progress("extracting visual OCR")
        frame_ocr_text = self._extract_frame_ocr(ingested_video)
        if progress is not None:
            progress("transcribing audio windows")
        audio_transcripts = self._transcribe_audio_windows(ingested_video)
        if progress is not None:
            progress("scoring audio windows")
        audio_scores = self._score_audio_windows(ingested_video, query)
        candidate_set = CandidateSet(
            visual=[
                self._visual_candidate(
                    ingested_video.video_id,
                    frame,
                    visual_scores.get(frame.path),
                    temporal_anchor_scores.get(frame.path),
                    temporal_target_scores.get(frame.path),
                    temporal_query.direction if temporal_query is not None else None,
                    temporal_query.anchor if temporal_query is not None else None,
                    temporal_query.target if temporal_query is not None else None,
                    frame_ocr_text.get(frame.path),
                    scene_by_frame.get(frame.index),
                )
                for frame in ingested_video.frames
            ],
            audio=self._audio_candidates(
                ingested_video=ingested_video,
                audio_transcripts=audio_transcripts,
                audio_scores=audio_scores,
            ),
        )
        if progress is not None:
            progress(
                f"candidates ready: visual={len(candidate_set.visual)}, "
                f"audio={len(candidate_set.audio)}"
            )
        return candidate_set

    def _score_visual_frames(
        self,
        ingested_video: IngestedVideo,
        query: str,
    ) -> dict[Path, float]:
        if self.visual_scorer is None:
            return {}
        return self.visual_scorer.score_frames(ingested_video.frames, query=query)

    def _scene_segments(
        self,
        ingested_video: IngestedVideo,
        visual_scores: dict[Path, float],
    ) -> list[SceneSegment]:
        if not self.scene_aware_visuals or self.visual_scorer is None:
            return []
        embed_frames = getattr(self.visual_scorer, "embed_frames", None)
        if embed_frames is None:
            return []

        embeddings = embed_frames(ingested_video.frames)
        relevance_by_frame = {
            frame.index: visual_scores.get(frame.path, 0.0)
            for frame in ingested_video.frames
        }
        return detect_scene_segments(
            embeddings=embeddings,
            relevance_by_frame=relevance_by_frame,
        )

    def _score_temporal_frames(
        self,
        ingested_video: IngestedVideo,
        query: str,
    ) -> tuple[dict[Path, float], dict[Path, float]]:
        temporal_query = parse_temporal_query(query)
        if temporal_query is None or self.visual_scorer is None:
            return {}, {}
        return (
            self.visual_scorer.score_frames(
                ingested_video.frames,
                query=temporal_query.anchor,
            ),
            self.visual_scorer.score_frames(
                ingested_video.frames,
                query=temporal_query.target,
            ),
        )

    def _transcribe_audio_windows(self, ingested_video: IngestedVideo) -> dict[Path, str]:
        if self.audio_transcriber is None:
            return {}
        return self.audio_transcriber.transcribe_windows(ingested_video.audio_windows)

    def _score_audio_windows(
        self,
        ingested_video: IngestedVideo,
        query: str,
    ) -> dict[Path, float]:
        if self.audio_scorer is None:
            return {}
        return self.audio_scorer.score_windows(ingested_video.audio_windows, query=query)

    def _extract_frame_ocr(self, ingested_video: IngestedVideo) -> dict[Path, str]:
        if self.frame_ocr is None:
            return {}
        return self.frame_ocr.extract_text(ingested_video.frames)

    def _audio_candidates(
        self,
        ingested_video: IngestedVideo,
        audio_transcripts: dict[Path, str],
        audio_scores: dict[Path, float],
    ) -> list[Candidate]:
        candidates: list[Candidate] = []
        for window in ingested_video.audio_windows:
            transcript_context = self._audio_transcript_context(
                ingested_video.audio_windows,
                audio_transcripts,
                window.index,
            )
            if self.audio_transcriber is not None:
                if transcript_context.text is None:
                    continue
                if _looks_like_noisy_transcript(transcript_context.text):
                    continue
            candidates.append(
                self._audio_candidate(
                    ingested_video.video_id,
                    window,
                    transcript_context,
                    audio_scores.get(window.path),
                )
            )
        return candidates

    def _audio_transcript_context(
        self,
        windows: list[AudioWindow],
        transcripts: dict[Path, str],
        center_index: int,
    ) -> AudioTranscriptContext:
        if not transcripts:
            return AudioTranscriptContext(text=None, start_seconds=None, end_seconds=None)

        start_index = max(center_index - self.audio_context_window_count, 0)
        end_index = min(center_index + self.audio_context_window_count + 1, len(windows))
        snippets: list[str] = []
        for window in windows[start_index:end_index]:
            transcript = transcripts.get(window.path, "").strip()
            if transcript:
                snippets.append(transcript)
        text = " ".join(snippets).strip() or None
        if text is None:
            return AudioTranscriptContext(text=None, start_seconds=None, end_seconds=None)

        first_window = windows[start_index]
        last_window = windows[end_index - 1]
        return AudioTranscriptContext(
            text=text,
            start_seconds=first_window.start_seconds,
            end_seconds=last_window.start_seconds + last_window.duration_seconds,
        )

    def _visual_candidate(
        self,
        video_id: str,
        frame: ExtractedFrame,
        saliency_score: float | None,
        temporal_anchor_score: float | None,
        temporal_target_score: float | None,
        temporal_direction: str | None,
        temporal_anchor_query: str | None,
        temporal_target_query: str | None,
        ocr_text: str | None,
        scene: SceneSegment | None,
    ) -> Candidate:
        text = f"visual frame sampled at {frame.timestamp_seconds:.2f} seconds"
        if ocr_text:
            text = f"on-screen text near {frame.timestamp_seconds:.2f} seconds: {ocr_text}"
        temporal_text = ocr_text or text
        if temporal_anchor_query is not None:
            temporal_anchor_score = max(
                float(temporal_anchor_score or 0.0),
                text_similarity(temporal_anchor_query, temporal_text),
            )
        if temporal_target_query is not None:
            temporal_target_score = max(
                float(temporal_target_score or 0.0),
                text_similarity(temporal_target_query, temporal_text),
            )
        return Candidate(
            id=f"{video_id}:visual:{frame.index}",
            timestamp_seconds=frame.timestamp_seconds,
            text=text,
            saliency_score=saliency_score,
            asset_path=frame.path,
            segment_id=scene.id if scene else None,
            scene_start_seconds=scene.start_seconds if scene else None,
            scene_end_seconds=scene.end_seconds if scene else None,
            temporal_anchor_score=temporal_anchor_score,
            temporal_target_score=temporal_target_score,
            temporal_direction=temporal_direction,
        )

    def _audio_candidate(
        self,
        video_id: str,
        window: AudioWindow,
        transcript_context: AudioTranscriptContext,
        saliency_score: float | None,
    ) -> Candidate:
        end_seconds = window.start_seconds + window.duration_seconds
        midpoint_seconds = window.start_seconds + (window.duration_seconds / 2)
        text = transcript_context.text or (
            "audio window "
            f"from {window.start_seconds:.2f} to {end_seconds:.2f} seconds"
        )
        clip_start_seconds = (
            transcript_context.start_seconds
            if transcript_context.start_seconds is not None
            else window.start_seconds
        )
        clip_end_seconds = (
            transcript_context.end_seconds
            if transcript_context.end_seconds is not None
            else end_seconds
        )
        return Candidate(
            id=f"{video_id}:audio:{window.index}",
            timestamp_seconds=midpoint_seconds,
            text=text,
            saliency_score=saliency_score,
            asset_path=window.path,
            segment_id=f"audio-window-{window.index}",
            scene_start_seconds=clip_start_seconds,
            scene_end_seconds=clip_end_seconds,
        )


def _looks_like_noisy_transcript(text: str) -> bool:
    words = re.findall(r"[a-z]+", text.lower())
    if len(words) < 8:
        return False
    ascii_ratio = sum(1 for char in text if char.isascii()) / max(len(text), 1)
    if ascii_ratio < 0.92:
        return True
    common_ratio = sum(1 for word in words if word in _COMMON_TRANSCRIPT_WORDS) / len(words)
    return common_ratio < 0.08
