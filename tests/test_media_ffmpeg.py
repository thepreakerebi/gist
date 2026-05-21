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


def test_extract_frames_reuses_existing_outputs_and_removes_stale_files(tmp_path: Path) -> None:
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"fake")
    output_dir = tmp_path / "frames"
    output_dir.mkdir()
    existing = output_dir / "frame_0000.jpg"
    stale = output_dir / "frame_0002.jpg"
    existing.write_bytes(b"existing")
    stale.write_bytes(b"stale")
    processor = FfmpegMediaProcessor()

    with patch("gist.media.ffmpeg.shutil.which", return_value="/usr/bin/tool"):
        with patch.object(
            processor,
            "probe",
            return_value=processor._parse_metadata(
                {
                    "streams": [{"codec_type": "video", "avg_frame_rate": "30/1"}],
                    "format": {"duration": "4.0"},
                }
            ),
        ):
            with patch.object(processor, "_run") as run:
                frames = processor.extract_frames(
                    video_path=video_path,
                    output_dir=output_dir,
                    sample_count=2,
                )

    assert [frame.path.name for frame in frames] == ["frame_0000.jpg", "frame_0001.jpg"]
    assert stale.exists() is False
    assert run.call_count == 1
    assert run.call_args.args[0][-1] == str(output_dir / "frame_0001.jpg")


def test_extract_audio_windows_reuses_existing_outputs_and_removes_stale_files(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"fake")
    output_dir = tmp_path / "audio"
    output_dir.mkdir()
    existing = output_dir / "audio_0000.wav"
    stale = output_dir / "audio_0003.wav"
    existing.write_bytes(b"existing")
    stale.write_bytes(b"stale")
    processor = FfmpegMediaProcessor()

    with patch("gist.media.ffmpeg.shutil.which", return_value="/usr/bin/tool"):
        with patch.object(
            processor,
            "probe",
            return_value=processor._parse_metadata(
                {
                    "streams": [
                        {"codec_type": "video", "avg_frame_rate": "30/1"},
                        {"codec_type": "audio"},
                    ],
                    "format": {"duration": "4.0"},
                }
            ),
        ):
            with patch.object(processor, "_run") as run:
                windows = processor.extract_audio_windows(
                    video_path=video_path,
                    output_dir=output_dir,
                    window_seconds=2.0,
                )

    assert [window.path.name for window in windows] == ["audio_0000.wav", "audio_0001.wav"]
    assert stale.exists() is False
    assert run.call_count == 1
    assert run.call_args.args[0][-1] == str(output_dir / "audio_0001.wav")


def test_extract_clip_clamps_to_video_bounds(tmp_path: Path) -> None:
    video_path = tmp_path / "input.mp4"
    output_path = tmp_path / "clips" / "evidence.mp4"
    video_path.write_bytes(b"fake")
    processor = FfmpegMediaProcessor()

    with patch("gist.media.ffmpeg.shutil.which", return_value="/usr/bin/tool"):
        with patch.object(
            processor,
            "probe",
            return_value=processor._parse_metadata(
                {
                    "streams": [{"codec_type": "video", "avg_frame_rate": "30/1"}],
                    "format": {"duration": "5.0"},
                }
            ),
        ):
            with patch.object(processor, "_run") as run:
                result = processor.extract_clip(
                    video_path=video_path,
                    output_path=output_path,
                    center_seconds=4.5,
                    duration_seconds=8.0,
                )

    command = run.call_args.args[0]
    assert result == output_path
    assert output_path.parent.exists()
    assert command[command.index("-ss") + 1] == "0.000"
    assert command[command.index("-t") + 1] == "5.000"


def test_missing_ffprobe_raises_clear_error(tmp_path: Path) -> None:
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"fake")

    with patch("gist.media.ffmpeg.shutil.which", return_value=None):
        with pytest.raises(MediaProcessingError, match="missing required binary"):
            FfmpegMediaProcessor().probe(video_path)
