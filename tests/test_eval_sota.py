import json
import subprocess
import sys

import pytest

from gist.eval.sota import run


def test_sota_cli_runs_benchmark_sweep_with_gateway(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    dataset = tmp_path / "video_mme.jsonl"
    output_dir = tmp_path / "sota"
    video_root = tmp_path / "videos"
    video_root.mkdir()
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
            str(video_root / "v1.mp4"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    dataset.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "video_id": "v1",
                "question": "What happens?",
                "duration": 60,
                "answer": "What happens?",
            }
        )
        + "\n"
    )

    run(
        [
            "--dataset",
            str(dataset),
            "--benchmark",
            "video_mme",
            "--output-dir",
            str(output_dir),
            "--video-root",
            str(video_root),
            "--sample-count",
            "4",
            "--gateway-command",
            (
                f"{sys.executable} -c "
                "\"import json,sys; payload=json.load(sys.stdin); "
                "print(json.dumps({'answer': payload['query'], 'provider': 'fake'}))\""
            ),
        ]
    )

    payload = json.loads((output_dir / "sota-report.json").read_text())
    assert (output_dir / "sota-report.md").exists()
    assert (output_dir / "sota-report.html").exists()
    assert [variant["name"] for variant in payload["variants"]] == [
        "gist_core",
        "gist_scene_clip",
        "gist_scene_router_adaptive",
        "gist_scene_spatial",
    ]
    assert payload["results"][0]["variants"][0]["answer_score"] == 1
    assert payload["results"][0]["baselines"][0]["answer_score"] == 1


def test_sota_cli_dry_run_reports_readiness_issues(tmp_path, capsys) -> None:
    dataset = tmp_path / "video_mme.jsonl"
    output_dir = tmp_path / "sota"
    dataset.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "video_id": "missing-video",
                "question": "What happens?",
                "duration": 60,
            }
        )
        + "\n"
    )

    run(
        [
            "--dataset",
            str(dataset),
            "--benchmark",
            "video_mme",
            "--output-dir",
            str(output_dir),
            "--video-root",
            str(tmp_path / "videos"),
            "--gateway-command",
            "unused",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert "examples=1" in captured.out
    assert "issues=2" in captured.out
    assert "missing video_path" in captured.out
    assert "missing expected answer" in captured.out
    assert (output_dir / "prepared-benchmark.jsonl").exists()
