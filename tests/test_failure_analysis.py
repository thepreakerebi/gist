import json

from gist.eval.failure_analysis import render_failure_analysis


def test_render_failure_analysis_joins_expected_answers(tmp_path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    report = tmp_path / "report.json"
    dataset.write_text(
        json.dumps(
            {
                "id": "q1",
                "query": "What color?",
                "choices": ["A. Red", "B. Blue"],
                "answer": "B",
            }
        )
        + "\n"
    )
    report.write_text(
        json.dumps(
            {
                "variants": [{"name": "gist_core"}],
                "results": [
                    {
                        "id": "q1",
                        "query": "What color?",
                        "baselines": [
                            {
                                "name": "uniform",
                                "answer_score": 0,
                                "predicted_answer": "A",
                                "selected_candidates": 1,
                                "selected": [
                                    {
                                        "selection_rank": 1,
                                        "modality": "visual",
                                        "timestamp_seconds": 1,
                                        "text": "red object",
                                    }
                                ],
                            }
                        ],
                        "variants": [
                            {
                                "name": "gist_core",
                                "answer_score": 0,
                                "predicted_answer": "A",
                                "response": {
                                    "query_intent": "mixed_av",
                                    "routing_reason": "test",
                                    "metrics": {
                                        "selected_candidates": 1,
                                        "estimated_token_reduction_percent": 75,
                                    },
                                    "selected": [
                                        {
                                            "selection_rank": 1,
                                            "modality": "visual",
                                            "timestamp_seconds": 1,
                                            "normalized_score": 0.5,
                                            "text": "red object",
                                            "reason": "top match",
                                        }
                                    ],
                                },
                            }
                        ],
                    }
                ],
            }
        )
    )

    markdown = render_failure_analysis(report_path=report, dataset_path=dataset)

    assert "Expected: B" in markdown
    assert "gist_core" in markdown
    assert "red object" in markdown
