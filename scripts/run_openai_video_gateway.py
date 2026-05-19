#!/usr/bin/env python3
import argparse
import json
import sys

from gist.gateway.openai_vision import (
    DEFAULT_MODEL,
    OpenAIVisionGatewayError,
    answer_from_gateway_payload,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="OpenAI vision gateway for Gist subprocess evaluation."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--detail", choices=["low", "high", "auto"], default="low")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--frame-sampling", choices=["start", "anchor"], default="start")
    parser.add_argument("--prompt-strategy", choices=["default", "task_aware"], default="default")
    args = parser.parse_args()

    try:
        payload = json.load(sys.stdin)
        result = answer_from_gateway_payload(
            payload=payload,
            model=args.model,
            max_frames=args.max_frames,
            detail=args.detail,
            timeout_seconds=args.timeout,
            ffmpeg_bin=args.ffmpeg_bin,
            frame_sampling=args.frame_sampling,
            prompt_strategy=args.prompt_strategy,
        )
    except (json.JSONDecodeError, OpenAIVisionGatewayError, OSError, ValueError) as exc:
        print(f"run_openai_video_gateway.py: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
