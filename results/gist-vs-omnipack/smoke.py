"""CPU plumbing test for `h2h.py`, run before paying for a pod.

Reuses the tiny random-weight Qwen2.5-Omni and the synthetic video from
`omnipack-replication/smoke_cpu.py`, and runs the real runner end to end on CPU
with the real CLIP, CLAP, Whisper and MiniLM: Gist's ingestion and selection,
unit-only encoding, OmniPack's stages, every arm's answer, and the control
check. Random Qwen weights answer at random, so **nothing printed is a result**.

What it proves: the unit plan matches the encoders token for token, encoding a
subset of units reproduces the full encoding exactly (the exactness note), the
kept tokens land on the right prompt positions, and every stage is measured.

    python results/gist-vs-omnipack/smoke.py          # in a transformers 4.52.3 env
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "omnipack-replication"))

import smoke_cpu  # noqa: E402

WORK = smoke_cpu.WORK


def main() -> int:
    if not smoke_cpu.TINY.exists():
        smoke_cpu.build_tiny_model()
    smoke_cpu.build_video()
    # A longer clip than smoke_cpu's, with the picture and the tone changing over
    # time, so Gist keeps a subset of units and the splice path is exercised.
    import subprocess

    # H2H_SMOKE_LONG=1 exercises the long-input protocol: a clip past Qwen's 300 s
    # audio cap, the cap raised, stage 1 chunked, and a context limit low enough
    # that the uncompressed arm is recorded as over context rather than run.
    long_mode = os.environ.get("H2H_SMOKE_LONG") == "1"
    duration = 400 if long_mode else 150

    out = smoke_cpu.VIDS / f"videomme-{smoke_cpu.VIDEO_ID}.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=640x360:rate=5:duration={duration}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=300:beep_factor=4:duration={duration}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(out),
        ],
        check=True,
    )
    os.environ.update(
        QWEN_MODEL_PATH=str(smoke_cpu.TINY),
        GIST_VIDEOS=str(smoke_cpu.VIDS),
        H2H_OUT=str(WORK / "out"),
        H2H_WORK=str(WORK / "work"),
        OMNIPACK_DEVICE="cpu",
        OMNIPACK_DTYPE="float32",
        OMNIPACK_INNER_LAYER="3",
        OMNIPACK_FRAMES="32",
        H2H_RATIOS="0.25,0.10",
        H2H_SCORER_DEVICE="cpu",
        H2H_COUNT_FLOPS=os.environ.get("H2H_COUNT_FLOPS", "0"),
    )
    if long_mode:
        os.environ.update(
            H2H_AUDIO_CAP_SECONDS="600",
            H2H_STAGE1_CHUNK_SECONDS="120",
            H2H_MAX_CONTEXT="8000",
            H2H_RATIOS="0.25,0.04",
            H2H_VERIFY="0",
            H2H_PRODUCT_WHISPER="0",
        )
    sys.argv = ["h2h.py", str(smoke_cpu.QFILE)]
    sys.path.insert(0, str(HERE))
    import h2h

    code = h2h.main()
    notes = [
        json.loads(line) for line in (WORK / "out" / "measurements.jsonl").read_text().splitlines()
    ]
    exact = [n for n in notes if n.get("record") == "exactness"]
    print("exactness:", exact)
    if not exact or exact[0]["visual_max_abs_diff"] > 1e-5 or exact[0]["audio_max_abs_diff"] > 1e-5:
        print("FAIL: unit encoding does not reproduce the full encoding")
        return 1
    return code


if __name__ == "__main__":
    raise SystemExit(main())
