import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from gist.media.errors import MediaProcessingError
from gist.media.ffmpeg import FfmpegMediaProcessor, evenly_spaced_timestamps


def test_evenly_spaced_timestamps_are_deterministic() -> None:
    assert evenly_spaced_timestamps(duration_seconds=10, sample_count=4) == [0.0, 2.5, 5.0, 7.5]


def test_probe_parses_video_metadata(tmp_path: Path) -> None:
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"fake")
    ffprobe_output = """
    {
      "streams": [
        {
          "codec_type": "video",
          "width": 1920,
          "height": 1080,
          "avg_frame_rate": "30000/1001"
        },
        {"codec_type": "audio"}
      ],
      "format": {"duration": "12.5"}
    }
    """

    with patch("gist.media.ffmpeg.shutil.which", return_value="/usr/bin/tool"):
        with patch("gist.media.ffmpeg.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess(
                args=["ffprobe"],
                returncode=0,
                stdout=ffprobe_output,
                stderr="",
            )

            metadata = FfmpegMediaProcessor().probe(video_path)

    assert metadata.duration_seconds == 12.5
    assert metadata.width == 1920
    assert metadata.height == 1080
    assert metadata.frame_rate == pytest.approx(29.97, rel=0.01)
    assert metadata.has_audio is True


def test_extract_audio_windows_returns_empty_when_video_has_no_audio(tmp_path: Path) -> None:
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"fake")
    processor = FfmpegMediaProcessor()

    with patch("gist.media.ffmpeg.shutil.which", return_value="/usr/bin/tool"):
        with patch.object(
            processor,
            "probe",
            return_value=processor._parse_metadata(
                {
                    "streams": [{"codec_type": "video", "avg_frame_rate": "30/1"}],
                    "format": {"duration": "3.0"},
                }
            ),
        ):
            windows = processor.extract_audio_windows(video_path, tmp_path / "audio")

    assert windows == []


def test_missing_ffprobe_raises_clear_error(tmp_path: Path) -> None:
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"fake")

    with patch("gist.media.ffmpeg.shutil.which", return_value=None):
        with pytest.raises(MediaProcessingError, match="missing required binary"):
            FfmpegMediaProcessor().probe(video_path)

