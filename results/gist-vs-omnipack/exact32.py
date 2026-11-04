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
import omnipack_h2h as op  # noqa: E402

model, proc = op.load_model()
th = model.thinker


def chunked(pixels, grid, size):
    t, h, w = (int(v) for v in grid[0])
    per = h * w
    out = []
    for s in range(0, t, size):
        c = min(size, t - s)
        out.append(
            th.get_video_features(
                pixels[s * per : (s + c) * per], torch.tensor([[c, h, w]], device=grid.device)
            )
        )
    return torch.cat(out).float()


for vid in ("ZLRyTFIZOIs", "eZ5lFBPpRkM"):
    media = op.load_media(f"/content/vids/videomme-{vid}.mp4")
    inp = op.build_inputs(proc, media, "q")
    px, grid = inp["pixel_values_videos"].to("cuda"), inp["video_grid_thw"].to("cuda")
    with torch.no_grad():
        th.visual.half()
        f4, f1 = chunked(px, grid, 4), chunked(px, grid, 1)
        th.visual.float()
        r4, r1 = chunked(px.float(), grid, 4), chunked(px.float(), grid, 1)

    def rel(x, y):
        return f"{((x - y).norm() / y.norm()).item():.2e}"

    print(
        "FP32REF",
        vid,
        "fp32 4v1",
        rel(r1, r4),
        "| fp16 groups-of-4 vs fp32",
        rel(f4, r4),
        "| fp16 one-at-a-time vs fp32",
        rel(f1, r4),
    )
    th.visual.half()
