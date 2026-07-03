import json
import math
import shutil
import subprocess
from pathlib import Path

from gist.media.errors import MediaProcessingError
from gist.media.models import AudioWindow, ExtractedFrame, VideoMetadata


class FfmpegMediaProcessor:
    def __init__(self, ffmpeg_bin: str = "ffmpeg", ffprobe_bin: str = "ffprobe") -> None:
        self.ffmpeg_bin = ffmpeg_bin
        self.ffprobe_bin = ffprobe_bin

    def probe(self, video_path: Path) -> VideoMetadata:
        self._require_binary(self.ffprobe_bin)
        self._require_file(video_path)

        payload = self._run_json(
            [
                self.ffprobe_bin,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(video_path),
            ]
        )
        return self._parse_metadata(payload)

    def extract_frames(
        self,
        video_path: Path,
        output_dir: Path,
        sample_count: int,
    ) -> list[ExtractedFrame]:
        if sample_count <= 0:
            raise ValueError("sample_count must be greater than zero")

        self._require_binary(self.ffmpeg_bin)
        self._require_file(video_path)
        metadata = self.probe(video_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamps = evenly_spaced_timestamps(metadata.duration_seconds, sample_count)
        _remove_stale_outputs(output_dir, "frame_*.jpg", expected_count=len(timestamps))
        frames: list[ExtractedFrame] = []
        skipped = 0
        for index, timestamp in enumerate(timestamps):
            frame_path = output_dir / f"frame_{index:04d}.jpg"
            if not _usable_file(frame_path):
                try:
                    self._run(
                        [
                            self.ffmpeg_bin,
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-y",
                            "-ss",
                            f"{timestamp:.3f}",
                            "-i",
                            str(video_path),
                            "-frames:v",
                            "1",
                            "-q:v",
                            "2",
                            # Normalize to JPEG full-range YUV so sources with
                            # non-standard color ranges still encode as MJPEG.
                            "-pix_fmt",
                            "yuvj420p",
                            str(frame_path),
                        ]
                    )
                except MediaProcessingError:
                    # A corrupt region at this timestamp should not abort the
                    # whole ingestion; skip the frame and keep going.
                    skipped += 1
                    continue
            # ffmpeg can exit 0 while writing nothing over a damaged region, so
            # only keep frames that were actually produced.
            if not _usable_file(frame_path):
                skipped += 1
                continue
            frames.append(
                ExtractedFrame(index=index, timestamp_seconds=timestamp, path=frame_path)
            )

        if not frames:
            raise MediaProcessingError(
                f"no frames could be extracted from {video_path} "
                f"({skipped} timestamps failed)"
            )

        return frames

    def extract_audio_windows(
        self,
        video_path: Path,
        output_dir: Path,
        window_seconds: float = 1.0,
    ) -> list[AudioWindow]:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be greater than zero")

        self._require_binary(self.ffmpeg_bin)
        self._require_file(video_path)
        metadata = self.probe(video_path)
        if not metadata.has_audio:
            return []

        output_dir.mkdir(parents=True, exist_ok=True)
        expected_windows = _audio_window_count(metadata.duration_seconds, window_seconds)
        _remove_stale_outputs(output_dir, "audio_*.wav", expected_count=expected_windows)
        windows: list[AudioWindow] = []
        start = 0.0
        index = 0
        while start < metadata.duration_seconds:
            duration = min(window_seconds, metadata.duration_seconds - start)
            audio_path = output_dir / f"audio_{index:04d}.wav"
            if not _usable_file(audio_path):
                self._run(
                    [
                        self.ffmpeg_bin,
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-ss",
                        f"{start:.3f}",
                        "-i",
                        str(video_path),
                        "-t",
                        f"{duration:.3f}",
                        "-vn",
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        str(audio_path),
                    ]
                )
            windows.append(
                AudioWindow(
                    index=index,
                    start_seconds=start,
                    duration_seconds=duration,
                    path=audio_path,
                )
            )
            start += window_seconds
            index += 1

        return windows

    def extract_clip(
        self,
        video_path: Path,
        output_path: Path,
        center_seconds: float | None = None,
        duration_seconds: float = 8.0,
        start_seconds: float | None = None,
    ) -> Path:
        if center_seconds is None and start_seconds is None:
            raise ValueError("center_seconds or start_seconds must be provided")
        if center_seconds is not None and center_seconds < 0:
            raise ValueError("center_seconds must be non-negative")
        if start_seconds is not None and start_seconds < 0:
            raise ValueError("start_seconds must be non-negative")
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be greater than zero")

        self._require_binary(self.ffmpeg_bin)
        self._require_file(video_path)
        metadata = self.probe(video_path)

        start = (
            start_seconds
            if start_seconds is not None
            else max((center_seconds or 0.0) - (duration_seconds / 2), 0.0)
        )
        if start + duration_seconds > metadata.duration_seconds:
            start = max(metadata.duration_seconds - duration_seconds, 0.0)
        actual_duration = min(duration_seconds, metadata.duration_seconds - start)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._run(
            [
                self.ffmpeg_bin,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{start:.3f}",
                "-i",
                str(video_path),
                "-t",
                f"{actual_duration:.3f}",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "28",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
        )
        return output_path

    def _parse_metadata(self, payload: dict) -> VideoMetadata:
        streams = payload.get("streams", [])
        format_info = payload.get("format", {})
        duration = _safe_float(format_info.get("duration"))

        video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
        audio_stream = any(stream.get("codec_type") == "audio" for stream in streams)
        if video_stream is None:
            raise MediaProcessingError("input does not contain a video stream")

        stream_duration = _safe_float(video_stream.get("duration"))
        duration_seconds = duration or stream_duration
        if duration_seconds is None or duration_seconds <= 0:
            raise MediaProcessingError("could not determine positive video duration")

        return VideoMetadata(
            duration_seconds=duration_seconds,
            width=video_stream.get("width"),
            height=video_stream.get("height"),
            frame_rate=_parse_frame_rate(video_stream.get("avg_frame_rate")),
            has_audio=audio_stream,
        )

    def _run_json(self, command: list[str]) -> dict:
        result = self._run(command)
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise MediaProcessingError("ffprobe returned invalid JSON") from exc

    def _run(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise MediaProcessingError(f"missing required binary: {command[0]}") from exc
        except subprocess.CalledProcessError as exc:
            message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
            raise MediaProcessingError(message) from exc

    def _require_binary(self, binary: str) -> None:
        if shutil.which(binary) is None:
            raise MediaProcessingError(f"missing required binary: {binary}")

    def _require_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            raise MediaProcessingError(f"video file does not exist: {path}")


def evenly_spaced_timestamps(duration_seconds: float, sample_count: int) -> list[float]:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be greater than zero")
    if sample_count <= 0:
        raise ValueError("sample_count must be greater than zero")
    if sample_count == 1:
        return [0.0]

    step = duration_seconds / sample_count
    return [min(duration_seconds, step * index) for index in range(sample_count)]


def _audio_window_count(duration_seconds: float, window_seconds: float) -> int:
    return math.ceil(duration_seconds / window_seconds)


def _remove_stale_outputs(output_dir: Path, pattern: str, expected_count: int) -> None:
    for path in output_dir.glob(pattern):
        index = _output_index(path)
        if index is None or index >= expected_count:
            path.unlink(missing_ok=True)


def _output_index(path: Path) -> int | None:
    try:
        return int(path.stem.rsplit("_", maxsplit=1)[-1])
    except ValueError:
        return None


def _usable_file(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_frame_rate(value: object) -> float | None:
    if not isinstance(value, str) or value in {"", "0/0"}:
        return None
    if "/" not in value:
        return _safe_float(value)

    numerator, denominator = value.split("/", maxsplit=1)
    numerator_value = _safe_float(numerator)
    denominator_value = _safe_float(denominator)
    if numerator_value is None or denominator_value in {None, 0}:
        return None
    return numerator_value / denominator_value
