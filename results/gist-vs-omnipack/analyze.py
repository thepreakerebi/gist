"""Turn the head-to-head logs into accuracy, cost and break-even tables.

    python results/gist-vs-omnipack/analyze.py results/gist-vs-omnipack

Reads `measurements.jsonl` (timed run), `answers.jsonl`, and, if present,
`measurements_flops.jsonl` (the FLOP-counting pass). Writes `results.md`,
`results.json` and `break_even.png` beside them.

How each method's cost is assembled, from stages measured on the same GPU:

  per video     full / omnipack: Qwen's encoders on the whole input (+ stage 1)
                gist: CLIP on the frames, CLAP on the audio segments, Whisper
                on the speech segments, MiniLM on the transcripts
  per question  full / omnipack: the answer (prefill + decode; stage 2 inside)
                gist: scoring the question, Qwen's encoders on the chosen
                units, the answer

Decoding the video is shared by every arm and reported on its own row. Energy
is the GPU board counter, so it includes Whisper, which runs outside PyTorch.
FLOPs for PyTorch stages are counted in the FLOP pass and carried to every
video and question through per-unit rates fitted on the counted ones; Whisper's
are derived from its architecture and labelled so.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from gist.eval.bootstrap import paired_difference, proportion_interval  # noqa: E402

DIR = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent)

PER_VIDEO = {
    "full": ["encode"],
    "gist": [
        "gist_write_media",
        "gist_clip_frames",
        "gist_clap_windows",
        "gist_whisper",
        "gist_minilm",
    ],
    "uniform": [],
}
PER_QUESTION = {
    "full": ["answer"],
    "gist": ["gist_select", "encode_units", "answer"],
    "uniform": ["encode_units", "answer"],
}

# Whisper architectures (Radford et al., 2023), for derived FLOPs.
WHISPER = {"tiny": dict(d=384, enc=4, dec=4), "base": dict(d=512, enc=6, dec=6)}
VOCAB, CTX, MELS = 51865, 1500, 80


def whisper_flops(model: str, windows: int, words: int, beam: int) -> float:
    """2 x multiply-accumulates. Every window is padded to 30 s (1500 positions)."""
    c = WHISPER[model]
    d, le, ld = c["d"], c["enc"], c["dec"]
    conv = 2 * 3000 * MELS * d * 3 + 2 * CTX * d * d * 3
    enc = conv + le * (24 * CTX * d * d + 4 * CTX * CTX * d)
    cross_kv = ld * 2 * 2 * CTX * d * d
    per_token = (
        ld * (2 * 4 * d * d + 2 * 2 * d * d + 2 * 8 * d * d + 4 * CTX * d + 4 * 64 * d)
        + 2 * d * VOCAB
    )
    tokens = 1.4 * words + 4 * windows
    return windows * (enc + cross_kv) + tokens * beam * per_token


def load(path: Path):
    stages, notes = [], defaultdict(list)
    if not path.exists():
        return stages, notes
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        kind = r.get("record")
        if kind == "stage":
            stages.append(r)
        else:
            notes[kind].append(r)
    return stages, notes


def method_of(r):
    return r["extra"].get("method")


def per_video_stages(method: str) -> list[str]:
    if method.startswith("omnipack_r"):
        return ["encode", "omnipack_stage1"]
    return PER_VIDEO[method]


def per_question_stages(method: str) -> list[str]:
    if method.startswith("omnipack_r"):
        return ["answer"]
    return PER_QUESTION[method]


def stage_rows(stages, name, method, scope_key, key):
    """Rows of a stage that belong to `method`: shared stages count for every arm."""
    out = []
    for r in stages:
        if r["stage"] != name or r.get("error"):
            continue
        m = method_of(r)
        own = m == method or (
            name == "encode"
            and m == "post_encoder"
            and (method == "full" or method.startswith("omnipack"))
        )
        if own and r["extra"].get(scope_key) == key:
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# FLOP rates from the counted pass
# ---------------------------------------------------------------------------


def fit_rates(fstages):
    if not fstages:
        return None
    rates = {}
    units = [r for r in fstages if r["stage"] == "encode_units" and r["flops"] is not None]
    if units:
        X = np.array(
            [[r["extra"]["patches"], r["extra"]["mel_frames"]] for r in units], dtype=float
        )
        y = np.array([r["flops"] for r in units], dtype=float)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        rates["per_patch"], rates["per_mel_frame"] = float(coef[0]), float(coef[1])
    enc = [r for r in fstages if r["stage"] == "encode" and r["flops"] is not None]
    if enc and "per_patch" in rates:
        plain = [
            rates["per_patch"] * r["extra"]["patches"]
            + rates["per_mel_frame"] * r["extra"]["mel_frames"]
            for r in enc
        ]
        # >1 because OmniPack's encode also reads the final-layer attention.
        rates["post_encoder_factor"] = float(
            np.mean([r["flops"] / p for r, p in zip(enc, plain, strict=True)])
        )
    for name, key in (
        ("gist_clip_frames", "items"),
        ("gist_clap_windows", "items"),
        ("gist_minilm", "items"),
    ):
        rows = [
            r
            for r in fstages
            if r["stage"] == name and r["flops"] is not None and r["extra"].get(key)
        ]
        if rows:
            rates[name] = float(np.mean([r["flops"] / r["extra"][key] for r in rows]))
    sel = [r["flops"] for r in fstages if r["stage"] == "gist_select" and r["flops"] is not None]
    if sel:
        rates["gist_select"] = float(np.mean(sel))
    s1 = [r for r in fstages if r["stage"] == "omnipack_stage1" and r["flops"] is not None]
    for r in s1:
        rates.setdefault(f"stage1_{method_of(r)}", []).append(r["flops"])
    for k in [k for k in rates if k.startswith("stage1_")]:
        rates[k] = float(np.mean(rates[k]))
    answers = defaultdict(list)
    for r in fstages:
        if r["stage"] == "answer" and r["flops"] is not None:
            family = method_of(r) if method_of(r).startswith("omnipack") else "plain"
            answers[family].append((r["extra"]["prefill_tokens"], r["flops"]))
    rates["answer"] = {}
    for family, pts in answers.items():
        L = np.array([p[0] for p in pts], dtype=float)
        X = np.stack([L, L * L], axis=1)
        coef, *_ = np.linalg.lstsq(X, np.array([p[1] for p in pts], dtype=float), rcond=None)
        rates["answer"][family] = [float(coef[0]), float(coef[1])]
    return rates


def stage_flops(r, rates):
    if r["flops"] is not None and r.get("_counted"):
        return r["flops"]
    if rates is None:
        return None
    e, name = r["extra"], r["stage"]
    if name == "encode_units" and "per_patch" in rates:
        return rates["per_patch"] * e["patches"] + rates["per_mel_frame"] * e["mel_frames"]
    if name == "encode" and "post_encoder_factor" in rates:
        return rates["post_encoder_factor"] * (
            rates["per_patch"] * e["patches"] + rates["per_mel_frame"] * e["mel_frames"]
        )
    if name in ("gist_clip_frames", "gist_clap_windows", "gist_minilm"):
        return rates.get(name, 0.0) * e.get("items", 0)
    if name == "gist_whisper":
        return whisper_flops(e["model"], e["items"], e.get("transcript_words", 0), e["beam"])
    if name == "gist_select":
        return rates.get("gist_select")
    if name == "omnipack_stage1":
        return rates.get(f"stage1_{method_of(r)}")
    if name == "answer":
        family = method_of(r) if method_of(r).startswith("omnipack") else "plain"
        coef = rates["answer"].get(family)
        if coef is None:
            return None
        L = e["prefill_tokens"]
        return coef[0] * L + coef[1] * L * L
    if name == "gist_write_media":
        return 0.0
    return None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def main() -> int:
    stages, notes = load(DIR / "measurements.jsonl")
    fstages, _ = load(DIR / "measurements_flops.jsonl")
    for r in fstages:
        r["_counted"] = True
    rates = fit_rates(fstages)
    idle = (notes.get("idle_power") or [{}])[0].get("watts")
    answers = [
        json.loads(line)
        for line in (DIR / "answers.jsonl").read_text().splitlines()
        if line.strip()
    ]

    arms = (
        [k for k in answers[0] if k not in ("qid", "video", "gold", "gist_units")]
        if answers
        else []
    )
    complete = [a for a in answers if all(arm in a for arm in arms)]
    outcomes = {arm: [a[arm] == a["gold"] for a in complete] for arm in arms}

    media_tokens = defaultdict(list)
    for r in stages:
        if r["stage"] == "answer" and not r.get("error"):
            media_tokens[method_of(r)].append(r["extra"]["media_tokens"])
    full_tokens = statistics.median(media_tokens["full"]) if media_tokens["full"] else None

    videos = sorted({r["extra"]["video"] for r in stages if r["extra"].get("scope") == "per_video"})
    qids = sorted({r["extra"]["qid"] for r in stages if r["extra"].get("scope") == "per_question"})

    def cost(method, scope, metric):
        keys, names, key_name = (
            (videos, per_video_stages(method), "video")
            if scope == "video"
            else (qids, per_question_stages(method), "qid")
        )
        totals = []
        for k in keys:
            total, seen = 0.0, False
            for name in names:
                for r in stage_rows(stages, name, method, key_name, k):
                    if metric == "seconds":
                        v = r["wall_seconds"]
                    elif metric == "energy_j":
                        v = r["energy_j"]
                    elif metric == "above_idle_j":
                        v = (
                            None
                            if r["energy_j"] is None or idle is None
                            else r["energy_j"] - idle * r["wall_seconds"]
                        )
                    else:
                        v = stage_flops(r, rates)
                    if v is None:
                        return None
                    total += v
                    seen = True
            if seen or not names:
                totals.append(total)
        return float(np.mean(totals)) if totals else 0.0

    shared_decode = [r["wall_seconds"] for r in stages if r["stage"] == "load_media"]
    methods = [m for m in arms if m != "uniform"] + (["uniform"] if "uniform" in arms else [])
    table = {}
    for m in methods:
        table[m] = {
            f"{scope}_{metric}": cost(m, scope, metric)
            for scope in ("video", "question")
            for metric in ("seconds", "energy_j", "above_idle_j", "flops")
        }
        peaks = [
            r["peak_reserved_gb"]
            for r in stages
            if (
                method_of(r) == m
                or (r["stage"] == "encode" and (m == "full" or m.startswith("omnipack")))
            )
            and r["peak_reserved_gb"] is not None
        ]
        table[m]["peak_reserved_gb"] = max(peaks) if peaks else None
        acts = [
            r["activation_gb"]
            for r in stages
            if r["stage"] in ("answer", "encode_units", "encode")
            and (
                method_of(r) == m
                or (r["stage"] == "encode" and (m == "full" or m.startswith("omnipack")))
            )
            and r["activation_gb"] is not None
        ]
        table[m]["max_activation_gb"] = max(acts) if acts else None

    def n_star(a, b, metric):
        """Questions per video beyond which `a` costs less in total than `b`."""
        pva, pqa = table[a][f"video_{metric}"], table[a][f"question_{metric}"]
        pvb, pqb = table[b][f"video_{metric}"], table[b][f"question_{metric}"]
        if None in (pva, pqa, pvb, pqb):
            return None
        if pqa >= pqb:
            return "never" if pva >= pvb else "always"
        return max(0.0, (pva - pvb) / (pqb - pqa))

    product = [r for r in stages if r["stage"] == "gist_whisper_product"]
    evalw = [r for r in stages if r["stage"] == "gist_whisper"]

    lines = ["# Gist against the OmniPack rebuild, same input, model, prompt and GPU", ""]
    dev = (notes.get("device") or [{}])[0]
    lines += [
        f"Device: {dev.get('name') or 'cpu'}, torch {dev.get('torch_version')}, "
        f"CUDA {dev.get('cuda_version')}.",
        "",
    ]
    lines += [
        f"Questions answered by every arm: {len(complete)} over "
        f"{len({a['video'] for a in complete})} videos. "
        f"Idle board power: {idle} W.",
        "",
    ]
    lines += [
        "## Accuracy",
        "",
        "| Arm | Correct | 95% CI | Media tokens to the LLM (median) | Share of full "
        "| Gist minus arm [95% CI] |",
        "| :-- | --: | :-- | --: | --: | :-- |",
    ]
    for arm in arms:
        o = outcomes[arm]
        ci = proportion_interval(o)
        tok = statistics.median(media_tokens[arm]) if media_tokens[arm] else None
        share = f"{tok / full_tokens:.1%}" if tok and full_tokens else ""
        diff = (
            ""
            if arm == "gist" or "gist" not in outcomes
            else paired_difference(outcomes["gist"], o).as_points()
        )
        lines.append(
            f"| {arm} | {sum(o)}/{len(o)} | [{ci.low:.1%}, {ci.high:.1%}] | "
            f"{tok:,.0f} | {share} | {diff} |"
        )
    lines.append("")

    def fmt(v, kind):
        if v is None:
            return "n/a"
        if kind == "flops":
            return f"{v / 1e12:,.2f}"
        if kind == "seconds":
            return f"{v:,.2f}"
        return f"{v:,.0f}"

    lines += [
        "## Cost on the same GPU",
        "",
        "Per video is paid once for every question about that video; "
        "per question is paid each time.",
        "Decoding the video is shared by every arm and not included: median "
        f"{statistics.median(shared_decode):.1f} s per video."
        if shared_decode
        else "",
        "",
        "| Method | Per video: GPU s | J | TFLOPs | Per question: GPU s | J | TFLOPs "
        "| Peak reserved GB |",
        "| :-- | --: | --: | --: | --: | --: | --: | --: |",
    ]
    for m in methods:
        t = table[m]
        lines.append(
            f"| {m} | {fmt(t['video_seconds'], 'seconds')} | {fmt(t['video_energy_j'], 'j')} | "
            f"{fmt(t['video_flops'], 'flops')} | {fmt(t['question_seconds'], 'seconds')} | "
            f"{fmt(t['question_energy_j'], 'j')} | {fmt(t['question_flops'], 'flops')} | "
            f"{t['peak_reserved_gb'] if t['peak_reserved_gb'] is not None else 'n/a'} |"
        )
    lines.append("")
    lines += [
        "## Break-even: questions per video beyond which Gist costs less in total",
        "",
        "| Against | GPU time | Energy | FLOPs |",
        "| :-- | :-- | :-- | :-- |",
    ]
    for m in methods:
        if m in ("gist", "uniform"):
            continue
        row = [n_star("gist", m, k) for k in ("seconds", "energy_j", "flops")]
        lines.append(
            f"| {m} | "
            + " | ".join(
                "n/a" if v is None else (v if isinstance(v, str) else f"{v:.1f}") for v in row
            )
            + " |"
        )
    lines.append("")
    if product and evalw:
        lines += [
            "## Whisper setting (sensitivity)",
            "",
            f"Selection used Whisper {evalw[0]['extra']['model']} "
            f"(beam {evalw[0]['extra']['beam']}) on speech segments only: "
            f"mean {np.mean([r['wall_seconds'] for r in evalw]):.2f} s and "
            f"{np.mean([r['energy_j'] or 0 for r in evalw]):,.0f} J per video. "
            f"The product's setting, {product[0]['extra']['model']} "
            f"(beam {product[0]['extra']['beam']}) on every segment: "
            f"mean {np.mean([r['wall_seconds'] for r in product]):.2f} s and "
            f"{np.mean([r['energy_j'] or 0 for r in product]):,.0f} J per video.",
            "",
        ]
    ex = notes.get("exactness")
    if ex:
        lines += [
            "Exactness: encoding only the chosen units matched the full encoding to max |diff| "
            f"{ex[0]['visual_max_abs_diff']:.3g} (visual) and "
            f"{ex[0]['audio_max_abs_diff']:.3g} (audio).",
            "",
        ]
    if rates is None:
        lines += ["FLOPs: no counting pass found, so FLOP columns are n/a.", ""]
    else:
        lines += [
            "FLOPs: PyTorch stages counted with `FlopCounterMode` on the counting pass and "
            "carried to every video and question by fitted per-unit rates; Whisper's are "
            "derived from its architecture.",
            "",
        ]

    (DIR / "results.md").write_text("\n".join(lines))
    (DIR / "results.json").write_text(
        json.dumps(
            {
                "idle_w": idle,
                "n_complete": len(complete),
                "accuracy": {a: {"correct": sum(o), "n": len(o)} for a, o in outcomes.items()},
                "median_media_tokens": {
                    a: statistics.median(v) for a, v in media_tokens.items() if v
                },
                "cost": table,
                "rates": rates,
            },
            indent=2,
            default=str,
        )
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=200)
        ax = axes[0]
        for arm in arms:
            if not media_tokens[arm]:
                continue
            ci = proportion_interval(outcomes[arm])
            x = statistics.median(media_tokens[arm])
            ax.errorbar(
                x,
                ci.point * 100,
                yerr=[[100 * (ci.point - ci.low)], [100 * (ci.high - ci.point)]],
                fmt="o",
                capsize=3,
                color="#c0392b" if arm == "gist" else "#555555",
            )
            ax.annotate(
                arm.replace("omnipack_r", "OmniPack "),
                (x, ci.point * 100),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Media tokens to the language model (median, log)")
        ax.set_ylabel("Correct (%)")
        ax.set_title("(a) Accuracy against tokens", fontsize=9)
        ax = axes[1]
        n = np.arange(1, 31)
        metric = (
            "energy_j" if table.get("gist", {}).get("video_energy_j") is not None else "seconds"
        )
        for m, colour in (
            ("full", "#999999"),
            ("omnipack_r0.25", "#2c7fb8"),
            ("omnipack_r0.1", "#7fcdbb"),
            ("gist", "#c0392b"),
        ):
            if m in table and table[m][f"video_{metric}"] is not None:
                ax.plot(
                    n,
                    table[m][f"video_{metric}"] + n * table[m][f"question_{metric}"],
                    label=m.replace("omnipack_r", "OmniPack "),
                    color=colour,
                )
        ax.set_xlabel("Questions asked about one video")
        ax.set_ylabel("GPU energy (J)" if metric == "energy_j" else "GPU time (s)")
        ax.set_title("(b) Total cost per video", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(DIR / "break_even.png")
    except Exception as exc:  # noqa: BLE001 - the tables are the result; the figure is a convenience
        print(f"figure skipped: {exc}")
    print((DIR / "results.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
