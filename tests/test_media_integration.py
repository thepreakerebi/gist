import shutil
import subprocess
from pathlib import Path

import pytest

from gist.media.ingestion import MediaIngestor


pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required for media integration tests",
)


def test_media_ingestor_extracts_real_synthetic_video(tmp_path: Path) -> None:
    video_path = tmp_path / "synthetic.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x120:rate=10:duration=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:duration=2",
            "-shortest",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = MediaIngestor(output_root=tmp_path / "ingested").ingest(
        video_path=video_path,
        sample_count=3,
        audio_window_seconds=1.0,
    )

    assert manifest.metadata.duration_seconds == pytest.approx(2.0, rel=0.1)
    assert manifest.metadata.has_audio is True
    assert len(manifest.frames) == 3
    assert len(manifest.audio_windows) == 2
    assert all(frame.path.exists() for frame in manifest.frames)
    assert all(window.path.exists() for window in manifest.audio_windows)
