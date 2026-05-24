import argparse
from pathlib import Path

from gist.cli import main as gist_main
from gist.gateway.structured import schema_name_for_extraction_preset, suggest_extraction_preset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Label a video with a suggested Gist extraction preset."
    )
    parser.add_argument("video_path", type=Path)
    parser.add_argument("--task", required=True, help="Natural-language labeling task.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--query", help="Override the compression query. Defaults to task.")
    parser.add_argument("--preset", help="Override the suggested extraction preset.")
    parser.add_argument("--processing-mode", default="auto")
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--audio-window-seconds", type=float)
    parser.add_argument("--visual-scorer", default="clip_scene")
    parser.add_argument("--audio-scorer", default="baseline")
    parser.add_argument(
        "--answer-with",
        choices=["extractive", "local-text", "ollama"],
        default="extractive",
    )
    parser.add_argument("--ollama-model")
    parser.add_argument("--ollama-url")
    parser.add_argument("--no-clips", action="store_true")
    parser.add_argument("--no-answer-prune", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    suggestion = suggest_extraction_preset(args.task)
    extraction_preset = args.preset or suggestion.recommended_preset
    schema_name = schema_name_for_extraction_preset(extraction_preset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extraction_json = args.output_dir / "extraction.json"
    extraction_csv = args.output_dir / "extraction.csv"
    output_root = args.output_dir / "runs"

    gist_args = [
        str(args.video_path),
        "--query",
        args.query or args.task,
        "--output-root",
        str(output_root),
        "--processing-mode",
        args.processing_mode,
        "--visual-scorer",
        args.visual_scorer,
        "--audio-scorer",
        args.audio_scorer,
        "--adaptive-budget",
        "--decompose-query",
        "--extraction-preset",
        extraction_preset,
        "--extraction-output",
        str(extraction_json),
        "--extraction-csv-output",
        str(extraction_csv),
    ]
    if args.sample_count is not None:
        gist_args.extend(["--sample-count", str(args.sample_count)])
    if args.audio_window_seconds is not None:
        gist_args.extend(["--audio-window-seconds", str(args.audio_window_seconds)])
    if args.answer_with:
        gist_args.extend(["--answer-with", args.answer_with])
    if args.ollama_model:
        gist_args.extend(["--ollama-model", args.ollama_model])
    if args.ollama_url:
        gist_args.extend(["--ollama-url", args.ollama_url])
    if args.no_clips:
        gist_args.append("--no-clips")
    if args.no_answer_prune:
        gist_args.append("--no-answer-prune")
    if args.quiet:
        gist_args.append("--quiet")

    exit_code = gist_main(gist_args)
    print(f"recommended_preset={suggestion.recommended_preset}")
    print(f"extraction_preset={extraction_preset}")
    print(f"schema_name={schema_name}")
    print(f"reason={suggestion.reason}")
    print(f"extraction={extraction_json}")
    print(f"extraction_csv={extraction_csv}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
