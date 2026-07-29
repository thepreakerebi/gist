"""Pre-bake a flagship Gist run into the web frontend's cached-runs directory.

The live demo depends on a running API host plus the OpenAI/Anthropic API and
venue WiFi. Any of those can stall mid-presentation. A cached run is the safety
net: it captures the exact `scored` + `done` SSE payloads the live stream would
emit, so the same UI can replay a known-good result identically when the API is
unreachable.

The captured JSON is byte-identical in shape to the live stream because it
reuses `_candidate_point` and the same payload construction as
`gist.api.demo`. Regenerate with `--answerer openai|claude` once API keys are
set so the fallback shows a real hosted-LLM answer.

Usage:
    uv run python scripts/bake_cached_run.py \
        --slug paul-graham \
        --label "Paul Graham talk" \
        --video .gist/videos/youtube/paul-graham-y-combinator.mp4 \
        --query "How do founders get startup ideas unconsciously?" \
        --answerer extractive
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from gist.api.demo import _candidate_point, _load_api_key
from gist.gateway.hosted_answerer import answer_with_hosted_llm
from gist.gateway.schemas import GatewayRequest
from gist.pipeline import LocalCompressionPipeline

# Written into gist/web/public/cached-runs/ so Vercel serves them statically.
OUTPUT_DIR = Path("web/public/cached-runs")


def bake(args: argparse.Namespace) -> dict[str, Any]:
    scored: dict[str, Any] = {}

    def progress(message: str) -> None:
        print(f"  [pipeline] {message}", flush=True)

    def on_candidates(candidate_set) -> None:
        scored["visual"] = [_candidate_point(c) for c in candidate_set.visual]
        scored["audio"] = [_candidate_point(c) for c in candidate_set.audio]

    print(f"[bake] running pipeline for {args.slug} ...", flush=True)
    ingestion, compression = LocalCompressionPipeline(
        output_root=Path(args.output_root)
    ).run(
        video_path=Path(args.video),
        query=args.query,
        sample_count=args.sample_count,
        visual_scorer=args.visual_scorer,
        audio_scorer=args.audio_scorer,
        adaptive_budget=True,
        decompose_query=True,
        progress=progress,
        on_candidates=on_candidates,
    )

    answer = compression.answer
    provider = compression.answer_provider or "extractive"
    if args.answerer in ("openai", "claude"):
        print(f"[bake] asking {args.answerer} ...", flush=True)
        gateway_response = answer_with_hosted_llm(
            GatewayRequest(query=args.query, compression=compression),
            answerer=args.answerer,
            model=args.answerer_model,
            api_key=_load_api_key(args.answerer),
            max_frames=args.max_frames,
            # Mirror the live demo: grounded, transcript-prioritizing prompt.
            prompt_strategy="intent",
        )
        answer = gateway_response.answer
        provider = gateway_response.provider

    done = {
        "answer": answer,
        "provider": provider,
        "compression": compression.model_dump(mode="json"),
        "video": {
            "duration_seconds": ingestion.metadata.duration_seconds,
            "frame_count": len(ingestion.frames),
            "audio_window_count": len(ingestion.audio_windows),
        },
    }

    return {
        "slug": args.slug,
        "label": args.label,
        "query": args.query,
        "scored": scored,
        "done": done,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", required=True, help="filename stem, e.g. paul-graham")
    parser.add_argument("--label", required=True, help="human label shown in the UI")
    parser.add_argument("--video", required=True, help="local video path")
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--answerer",
        default="extractive",
        choices=["extractive", "openai", "claude"],
    )
    parser.add_argument("--answerer-model", default=None)
    parser.add_argument("--visual-scorer", default="clip_scene")
    # dispatcher (not auto): auto only scores transcripts on >=10min videos, so
    # short demo clips need the dispatcher to select any audio. Mirrors the demo.
    parser.add_argument("--audio-scorer", default="dispatcher")
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--output-root", default=".gist/demo-web")
    args = parser.parse_args()

    payload = bake(args)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{args.slug}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[bake] wrote {out_path}", flush=True)

    # Maintain a manifest the frontend reads to list cached runs.
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest: list[dict[str, str]] = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    manifest = [e for e in manifest if e.get("slug") != args.slug]
    manifest.append(
        {
            "slug": args.slug,
            "label": args.label,
            "query": args.query,
            "provider": payload["done"]["provider"],
        }
    )
    manifest.sort(key=lambda e: e["label"])
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[bake] updated {manifest_path} ({len(manifest)} runs)", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
