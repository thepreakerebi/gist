"""Run on the pod after the timed run: how far unit-by-unit encoding is from the full
encoding in fp16 (exact.py), and how far each is from an fp32 reference (exact32.py).
The printed lines are kept in exactness.txt."""

import sys

import torch

sys.argv = ["x", "/root/data/videomme_av_all.json"]
for p in (
    "/root/gist/results/gist-vs-omnipack",
    "/root/gist/results/omnipack-replication",
    "/root/gist/src",
):
    sys.path.insert(0, p)
import h2h  # noqa: E402
import omnipack_h2h as op  # noqa: E402

model, proc = op.load_model()
th = model.thinker
for vid in ("ZLRyTFIZOIs", "eZ5lFBPpRkM"):
    media = op.load_media(f"/content/vids/videomme-{vid}.mp4")
    inp = op.build_inputs(proc, media, "q")
    v, a, info = op.encode(model, inp)
    plan = h2h.audio_units(th, int(inp["feature_attention_mask"][0].sum()))
    vs, as_ = h2h.encode_units(
        th, inp, list(range(int(inp["video_grid_thw"][0, 0]))), list(range(len(plan))), plan
    )
    for name, x, y in (("visual", vs, v), ("audio", as_, a)):
        x, y = x.float(), y.float()
        d = x - y
        print(
            "EXACT",
            vid,
            name,
            "max_abs",
            round(d.abs().max().item(), 4),
            "rel_fro",
            f"{(d.norm() / y.norm()).item():.2e}",
            "min_cos",
            round(torch.nn.functional.cosine_similarity(x, y, dim=-1).min().item(), 6),
            "max_abs_value",
            round(y.abs().max().item(), 2),
        )
