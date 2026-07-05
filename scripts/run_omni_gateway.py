#!/usr/bin/env python3
"""Qwen2.5-Omni gateway for Gist (persistent subprocess protocol).

Loads a real audio-visual Omni-LLM once, then answers Gist's compressed
evidence: it reads one JSON request per stdin line and writes one JSON response
line ({"answer","provider"}). Visual evidence is fed as frame images
(``asset_path``), audio evidence as the window WAV (``asset_path``/``clip_path``),
so the model sees exactly the frames+audio Gist selected — the compression layer
sitting in front of the Omni-LLM, as the capstone plan specifies.

Designed to be driven by gist.gateway.subprocess.PersistentSubprocessVideoLlmGateway.
Run on a GPU box:  python scripts/run_omni_gateway.py --model Qwen/Qwen2.5-Omni-3B
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _collect_media(evidence: list[dict], max_images: int, max_audios: int):
    images, audios = [], []
    for item in evidence:
        for key in ("asset_path", "clip_path"):
            p = item.get(key)
            if not p:
                continue
            ext = Path(p).suffix.lower()
            if ext in IMAGE_EXTS and len(images) < max_images and p not in images:
                images.append(p)
            elif ext in AUDIO_EXTS and len(audios) < max_audios and p not in audios:
                audios.append(p)
    return images, audios


def _build_conversation(query: str, images: list[str], audios: list[str], context: str):
    content: list[dict] = []
    for a in audios:
        content.append({"type": "audio", "audio": a})
    for img in images:
        content.append({"type": "image", "image": img})
    instruction = (
        "You are given frames and audio sampled from a video. Using ONLY what you "
        "see and hear, answer the question concisely.\n"
        f"Question: {query}"
    )
    content.append({"type": "text", "text": instruction})
    return [{"role": "user", "content": content}]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Omni-3B")
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--max-audios", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    import torch
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    try:
        from qwen_omni_utils import process_mm_info
    except Exception:  # pragma: no cover - optional helper
        process_mm_info = None

    _log(f"[omni] loading {args.model} ...")
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda", enable_audio_output=False
    )
    model.eval()
    _log(f"[omni] ready; mem={torch.cuda.memory_allocated() / 1e9:.1f}GB")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps({"error": "bad json"}), flush=True)
            continue
        if req.get("type") == "shutdown":
            break

        query = req.get("query", "")
        evidence = req.get("evidence", [])
        context = req.get("context", "")
        images, audios = _collect_media(evidence, args.max_images, args.max_audios)
        conv = _build_conversation(query, images, audios, context)

        try:
            text = processor.apply_chat_template(
                conv, add_generation_prompt=True, tokenize=False
            )
            if process_mm_info is not None:
                audio_in, image_in, video_in = process_mm_info(conv, use_audio_in_video=False)
            else:
                from PIL import Image
                import soundfile as sf

                image_in = [Image.open(p).convert("RGB") for p in images] or None
                audio_in = [sf.read(p)[0] for p in audios] or None
                video_in = None
            inputs = processor(
                text=text, audio=audio_in, images=image_in, videos=video_in,
                return_tensors="pt", padding=True,
            ).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens,
                    do_sample=False, return_audio=False,
                )
            ans = processor.batch_decode(
                out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )[0].strip()
            print(json.dumps({
                "answer": ans or "(empty)",
                "provider": f"omni:{args.model}",
                "n_images": len(images), "n_audios": len(audios),
            }), flush=True)
        except Exception as exc:  # keep the gateway alive across a bad request
            _log(f"[omni] error: {exc}")
            print(json.dumps({"answer": f"(error: {exc})", "provider": f"omni:{args.model}"}),
                  flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
