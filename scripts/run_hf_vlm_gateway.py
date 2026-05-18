#!/usr/bin/env python3
import argparse
import json
import sys

from gist.gateway.hf_vision import (
    DEFAULT_HF_MODEL,
    HuggingFaceVisionGatewayError,
    answer_from_gateway_payload,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Local Hugging Face VLM gateway for Gist SOTA benchmarking."
    )
    parser.add_argument("--model", default=DEFAULT_HF_MODEL)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    args = parser.parse_args()

    try:
        payload = json.load(sys.stdin)
        result = answer_from_gateway_payload(
            payload=payload,
            model=args.model,
            max_frames=args.max_frames,
            max_new_tokens=args.max_new_tokens,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            trust_remote_code=args.trust_remote_code,
            ffmpeg_bin=args.ffmpeg_bin,
        )
    except (json.JSONDecodeError, HuggingFaceVisionGatewayError, OSError, ValueError) as exc:
        print(f"run_hf_vlm_gateway.py: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
