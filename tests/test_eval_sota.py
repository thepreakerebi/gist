import json
import sys

from gist.eval.sota import run


def test_sota_cli_runs_benchmark_sweep_with_gateway(tmp_path) -> None:
    dataset = tmp_path / "video_mme.jsonl"
    output_dir = tmp_path / "sota"
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
