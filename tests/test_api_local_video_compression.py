import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gist.api.app import create_app


pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required for local video compression API tests",
)


def test_local_video_compression_endpoint_runs_end_to_end(tmp_path: Path) -> None:
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
            "testsrc=size=160x120:rate=10:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:duration=1",
            "-shortest",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    response = TestClient(create_app()).post(
        "/v1/local-video-compressions",
        json={
            "video_path": str(video_path),
            "output_root": str(tmp_path / "ingested"),
            "query": "audio",
            "preset": "balanced",
            "visual_scorer": "baseline",
            "audio_scorer": "baseline",
            "decompose_query": True,
            "sample_count": 2,
            "audio_window_seconds": 0.5,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["ingestion"]["frames"]) == 2
    assert len(body["ingestion"]["audio_windows"]) == 2
    assert body["compression"]["metrics"]["input_candidates"] == 4
