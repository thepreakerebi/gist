"""CPU plumbing test for `omnipack_h2h.py`, run before paying for a pod.

Builds a tiny, randomly initialised Qwen2.5-Omni from the real checkpoint's
config (same architecture, same tokenizer, same processor, shrunk widths and
depths), makes a short synthetic video with an audio track, and runs the real
runner end to end on CPU: encoders, the attention readout, stage 1, stage 2 at
the insertion layer, the hand-written decode loop, and the check of the `full`
arm against stock `generate`.

Random weights answer at random, so **nothing this prints is a result**. What it
proves is that every shape lines up, the token layout is read correctly, and
the control arm reproduces stock generation token for token. Those are the
failures that would otherwise surface on a paid GPU.

Needs transformers 4.52.3 (the pod pin), torch, qwen-omni-utils, ffmpeg, and
network access once to fetch the config and tokenizer (a few MB, no weights):

    python results/omnipack-replication/smoke_cpu.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

WORK = Path(os.environ.get("OMNIPACK_SMOKE_DIR", tempfile.mkdtemp(prefix="omnipack-smoke-")))
TINY = WORK / "tiny-omni"
VIDS = WORK / "vids"
QFILE = WORK / "questions.json"
VIDEO_ID = "smoke01"


def build_tiny_model() -> None:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import (
        Qwen2_5OmniConfig,
        Qwen2_5OmniForConditionalGeneration,
        Qwen2_5OmniProcessor,
    )

    meta = snapshot_download(
        "Qwen/Qwen2.5-Omni-7B",
        allow_patterns=["*.json", "*.txt", "merges.txt", "vocab.json"],
        local_dir=WORK / "meta",
    )
    raw = json.loads((Path(meta) / "config.json").read_text())
    raw["enable_audio_output"] = False
    thinker = raw["thinker_config"]
    thinker["text_config"].update(
        hidden_size=64, intermediate_size=128, num_hidden_layers=6,
        num_attention_heads=4, num_key_value_heads=2, max_window_layers=6,
    )
    # head_dim 16 -> 8 rotary pairs, split across time, height, width.
    thinker["text_config"]["rope_scaling"]["mrope_section"] = [2, 3, 3]
    thinker["vision_config"].update(
        depth=4, hidden_size=32, embed_dim=32, num_heads=2,
        intermediate_size=64, out_hidden_size=64, fullatt_block_indexes=[1, 3],
    )
    thinker["audio_config"].update(
        d_model=32, encoder_layers=2, encoder_attention_heads=2,
        encoder_ffn_dim=64, output_dim=64, num_hidden_layers=2,
    )
    config = Qwen2_5OmniConfig(**raw)
    torch.manual_seed(0)
    model = Qwen2_5OmniForConditionalGeneration(config)
    model.save_pretrained(TINY)
    Qwen2_5OmniProcessor.from_pretrained(meta).save_pretrained(TINY)
    # from_pretrained always looks for the speaker table, even with audio output off.
    torch.save({"Chelsie": {}}, TINY / "spk_dict.pt")


def build_video() -> None:
    VIDS.mkdir(parents=True, exist_ok=True)
    out = VIDS / f"videomme-{VIDEO_ID}.mp4"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=size=640x360:rate=10:duration=12",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out),
        ],
        check=True,
    )
    QFILE.write_text(json.dumps([
        {
            "videoID": VIDEO_ID, "question_id": f"{VIDEO_ID}-{i}", "answer": "A",
            "question": question, "options": ["A. one", "B. two", "C. three", "D. four"],
        }
        for i, question in enumerate(["How many tones play?", "What colour is the top bar?"])
    ]))


def main() -> int:
    if shutil.which("ffmpeg") is None:
        print("ffmpeg is required")
        return 2
    print(f"workdir: {WORK}", flush=True)
    if not TINY.exists():
        build_tiny_model()
    build_video()

    os.environ.update(
        QWEN_MODEL_PATH=str(TINY),
        GIST_VIDEOS=str(VIDS),
        GIST_MEASUREMENTS=str(WORK / "measurements.jsonl"),
        GIST_RESULTS=str(WORK / "results.jsonl"),
        OMNIPACK_DEVICE="cpu",
        OMNIPACK_DTYPE="float32",
        OMNIPACK_INNER_LAYER="3",
        OMNIPACK_FRAMES="8",
        OMNIPACK_RATIOS=os.environ.get("OMNIPACK_RATIOS", "0.25,0.10"),
    )
    sys.argv = ["omnipack_h2h.py", str(QFILE)]
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import omnipack_h2h

    return omnipack_h2h.main()


if __name__ == "__main__":
    raise SystemExit(main())
