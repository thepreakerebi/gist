"""System-efficiency accounting for Gist: analytic FLOPs and measured cost.

Two different kinds of number live here and the distinction is load bearing, so
it is made in the type system rather than left to prose:

*Analytic* (``build_report``). Encoder FLOPs and downstream token counts,
computed from the target encoders' transformer configs. The item counts feeding
them are real — how many frames and audio windows Gist actually encodes (K)
against a full/uniform baseline (N) — and the relative saving (1 - K/N) is exact.
The absolute GFLOPs are *derived*, not profiled. Anything quoting them must say
so; "measured GFLOPs" is a claim this module does not support.

*Measured* (``build_measured_report``). Wall clock and peak GPU memory taken
from a real run by ``gist.eval.profiling``. This is what an efficiency claim
needs in order to survive review, because Gist's whole thesis is that selecting
before the encoders run reduces encoder cost and peak memory — and peak memory
in particular cannot be inferred from a FLOP count at all.

The two are reported in separate tables under separate headings. They are never
summed, averaged, or presented as if one corroborated the other.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from gist.eval.profiling import DeviceInfo, StageMeasurement, load_measurements


def transformer_encoder_gflops(
    num_tokens: int, dim: int, depth: int, mlp_ratio: float = 4.0
) -> float:
    """Analytic forward-pass GFLOPs for a ViT/transformer encoder on one item.

    Per layer MACs = attention (4*N*D^2 + 2*N^2*D) + MLP (2*mlp_ratio*N*D^2).
    FLOPs = 2 * MACs. Standard accounting (e.g. FastV / EViT FLOP tables).
    """
    n, d = num_tokens, dim
    per_layer_macs = 4 * n * d * d + 2 * n * n * d + 2 * mlp_ratio * n * d * d
    total_flops = 2 * depth * per_layer_macs
    return total_flops / 1e9


@dataclass(frozen=True)
class EncoderProfile:
    name: str
    # vision: per-frame encoder cost + downstream tokens the LLM consumes
    vision_gflops_per_frame: float
    vision_tokens_per_frame: int
    # audio: per-window (e.g. 30s) encoder cost + downstream tokens
    audio_gflops_per_window: float
    audio_tokens_per_window: int


def _qwen_omni_profile() -> EncoderProfile:
    """Derived from Qwen2.5-Omni-7B's encoders (documented, approximate).

    Vision (SigLIP-so400m, patch14 @ 384): num_patches=(384/14)^2~=729,
    dim=1152, depth=27, mlp_ratio=4304/1152~=3.74. Qwen merges 2x2 patch groups
    -> ~196 LLM tokens/frame. Audio (Whisper-large-v3 encoder): 30s window ->
    1500 frames, dim=1280, depth=32, then pooled to ~ (30s * ~25 tok/s)=~750 LLM
    tokens/window.
    """
    vision = transformer_encoder_gflops(num_tokens=729, dim=1152, depth=27, mlp_ratio=3.74)
    audio = transformer_encoder_gflops(num_tokens=1500, dim=1280, depth=32, mlp_ratio=4.0)
    return EncoderProfile(
        name="qwen2.5-omni-7b",
        vision_gflops_per_frame=vision,
        vision_tokens_per_frame=196,
        audio_gflops_per_window=audio,
        audio_tokens_per_window=750,
    )


PROFILES = {"qwen2.5-omni-7b": _qwen_omni_profile()}


class ConditionCost(BaseModel):
    condition: str
    frames: int
    audio_windows: int
    vision_gflops: float
    audio_gflops: float
    system_gflops: float
    downstream_tokens: int
    system_gflops_saving_pct: float | None = None
    token_saving_pct: float | None = None


class EfficiencyReport(BaseModel):
    profile: str
    baseline: str
    conditions: list[ConditionCost]

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))


def _cost(condition: str, frames: int, windows: int, profile: EncoderProfile) -> ConditionCost:
    vision_g = frames * profile.vision_gflops_per_frame
    audio_g = windows * profile.audio_gflops_per_window
    tokens = frames * profile.vision_tokens_per_frame + windows * profile.audio_tokens_per_window
    return ConditionCost(
        condition=condition, frames=frames, audio_windows=windows,
        vision_gflops=round(vision_g, 2), audio_gflops=round(audio_g, 2),
        system_gflops=round(vision_g + audio_g, 2), downstream_tokens=tokens,
    )


def build_report(
    conditions: dict[str, tuple[int, int]],
    profile: EncoderProfile,
    baseline: str,
) -> EfficiencyReport:
    if baseline not in conditions:
        raise ValueError(f"baseline '{baseline}' not among conditions")
    costs = {name: _cost(name, f, w, profile) for name, (f, w) in conditions.items()}
    base = costs[baseline]
    results: list[ConditionCost] = []
    for name, cost in costs.items():
        if base.system_gflops > 0:
            cost.system_gflops_saving_pct = round(
                (1 - cost.system_gflops / base.system_gflops) * 100, 1
            )
        if base.downstream_tokens > 0:
            cost.token_saving_pct = round(
                (1 - cost.downstream_tokens / base.downstream_tokens) * 100, 1
            )
        results.append(cost)
    return EfficiencyReport(profile=profile.name, baseline=baseline, conditions=results)


def render_markdown(report: EfficiencyReport) -> str:
    lines = [
        "# Gist system-efficiency — analytic (architecture-derived FLOPs, real item counts)",
        "",
        f"- Encoder profile: {report.profile}; baseline: {report.baseline}",
        "- Frames/windows are real encode counts. GFLOPs are **derived** from the encoder",
        "  configs, not profiled — do not quote them as measured.",
        "",
        "| Condition | Frames | Audio windows | System GFLOPs | Downstream tokens | GFLOPs saving |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for c in report.conditions:
        saving = "—" if c.system_gflops_saving_pct is None else f"{c.system_gflops_saving_pct:.1f}%"
        lines.append(
            f"| {c.condition} | {c.frames} | {c.audio_windows} | {c.system_gflops:,.0f} "
            f"| {c.downstream_tokens:,} | {saving} |"
        )
    return "\n".join(lines) + "\n"


def _conditions_from_vision_report(path: Path) -> dict[str, tuple[int, int]]:
    """Pull per-condition frame counts from a vision-benchmark report.

    Audio windows are 0 in that (frames-only) setting; use CLI args to add audio.
    """
    data = json.loads(path.read_text())
    out: dict[str, tuple[int, int]] = {}
    for cond, summ in data.get("summaries", {}).items():
        out[cond] = (int(round(summ.get("avg_frames", 0))), 0)
    return out


# --------------------------------------------------------------------------
# Measured: wall clock and peak GPU memory from a real run.
# --------------------------------------------------------------------------


class MeasuredCondition(BaseModel):
    """Per-condition aggregate of profiled rows.

    ``n`` is every row seen; ``n_used`` is how many survived the warmup drop and
    actually fed the timing statistics. They differ, so both are reported.

    ``n_oom`` is reported beside the timings rather than folded into them. A
    condition that runs out of memory has not produced a slow measurement, it
    has produced a different result — it did not fit — and averaging a failed
    forward pass into a latency would hide exactly the finding that matters for
    a pre-encoder claim.
    """

    condition: str
    n: int
    n_used: int
    n_oom: int
    wall_seconds_median: float | None = None
    wall_seconds_mean: float | None = None
    peak_reserved_gb_max: float | None = None
    activation_gb_median: float | None = None
    activation_gb_max: float | None = None
    speedup_vs_baseline: float | None = None
    activation_saving_pct: float | None = None


class MeasuredReport(BaseModel):
    device: str
    baseline: str
    stage: str | None = None
    warmup_dropped: int = 0
    conditions: list[MeasuredCondition]

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def build_measured_report(
    stages: Sequence[StageMeasurement],
    *,
    baseline: str,
    device: DeviceInfo | None = None,
    stage: str | None = None,
    drop_warmup: int = 1,
) -> MeasuredReport:
    """Aggregate profiled rows by condition.

    ``drop_warmup`` discards the first N successful rows of each condition. The
    first call on a fresh process pays for lazy CUDA context creation and kernel
    autotuning, which on a short run is large enough to dominate the median and
    to flatter whichever condition happens to run second.
    """
    rows = [row for row in stages if stage is None or row.stage == stage]
    grouped: dict[str, list[StageMeasurement]] = {}
    for row in rows:
        grouped.setdefault(str(row.extra.get("condition", row.stage)), []).append(row)
    if baseline not in grouped:
        raise ValueError(f"baseline '{baseline}' not among conditions {sorted(grouped)}")

    stats: dict[str, MeasuredCondition] = {}
    for name, group in grouped.items():
        ok = [row for row in group if row.ok][drop_warmup:]
        walls = [row.wall_seconds for row in ok]
        activations = [row.activation_gb for row in ok if row.activation_gb is not None]
        # Reserved memory spans every row, warmup and OOM included: warmup does not
        # inflate the footprint the way it inflates the clock, and the whole point
        # of the high-water mark is to include the case that did not fit.
        reserved = [row.peak_reserved_gb for row in group if row.peak_reserved_gb is not None]
        stats[name] = MeasuredCondition(
            condition=name,
            n=len(group),
            n_used=len(ok),
            n_oom=sum(1 for row in group if row.oom),
            wall_seconds_median=_median(walls),
            wall_seconds_mean=round(sum(walls) / len(walls), 4) if walls else None,
            peak_reserved_gb_max=max(reserved) if reserved else None,
            activation_gb_median=_median(activations),
            activation_gb_max=max(activations) if activations else None,
        )

    base = stats[baseline]
    for cond in stats.values():
        # Guard the divisor explicitly. Testing truthiness would silently drop a
        # legitimate median of 0.0 and report no comparison at all.
        if base.wall_seconds_median is not None and cond.wall_seconds_median:
            cond.speedup_vs_baseline = round(
                base.wall_seconds_median / cond.wall_seconds_median, 2
            )
        if base.activation_gb_median and cond.activation_gb_median is not None:
            cond.activation_saving_pct = round(
                (1 - cond.activation_gb_median / base.activation_gb_median) * 100, 1
            )

    described = device.describe() if device else "unknown device"
    return MeasuredReport(
        device=described,
        baseline=baseline,
        stage=stage,
        warmup_dropped=drop_warmup,
        conditions=sorted(stats.values(), key=lambda c: c.condition),
    )


def render_measured_markdown(report: MeasuredReport) -> str:
    lines = [
        "# Gist system-efficiency — measured (wall clock and peak GPU memory)",
        "",
        f"- Device: {report.device}",
        f"- Baseline: {report.baseline}"
        + (f"; stage: {report.stage}" if report.stage else ""),
        f"- First {report.warmup_dropped} successful call(s) per condition dropped as warmup.",
        "- `activation GB` is peak allocated minus memory already resident, so the",
        "  model weights (identical under every condition) do not mask the effect.",
        "- `peak reserved GB` is the allocator high-water mark, which is what decides",
        "  whether the card OOMs.",
        "",
        "| Condition | n | OOM | Median s | Speedup | Activation GB (median) "
        "| Peak reserved GB | Activation saving |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    def fmt(value: float | None, spec: str = ",.2f") -> str:
        return "—" if value is None else format(value, spec)

    for c in report.conditions:
        speedup = "—" if c.speedup_vs_baseline is None else f"{c.speedup_vs_baseline:.2f}x"
        saving = "—" if c.activation_saving_pct is None else f"{c.activation_saving_pct:.1f}%"
        lines.append(
            f"| {c.condition} | {c.n} | {c.n_oom} | {fmt(c.wall_seconds_median)} "
            f"| {speedup} | {fmt(c.activation_gb_median, ',.3f')} "
            f"| {fmt(c.peak_reserved_gb_max, ',.2f')} | {saving} |"
        )
    oomed = [c.condition for c in report.conditions if c.n_oom]
    if oomed:
        lines += [
            "",
            "**Out of memory.** " + ", ".join(oomed) + " exhausted the device on at least",
            "one case. For a pre-encoder claim that is a result, not a run failure: the",
            "condition did not fit on hardware where Gist did.",
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "System efficiency. Without --measurements: analytic FLOPs from real item "
            "counts. With --measurements: also wall clock and peak GPU memory from a run."
        )
    )
    parser.add_argument("--profile", default="qwen2.5-omni-7b", choices=list(PROFILES))
    parser.add_argument("--baseline-frames", type=int, default=64)
    parser.add_argument("--baseline-audio-windows", type=int, default=120)
    parser.add_argument("--uniform-frames", type=int, default=8)
    parser.add_argument("--uniform-audio-windows", type=int, default=8)
    parser.add_argument("--gist-frames", type=int, default=8)
    parser.add_argument("--gist-audio-windows", type=int, default=4)
    parser.add_argument(
        "--from-vision-report",
        type=Path,
        help="Pull per-condition frame counts from a gist-benchmark-videomme-vision JSON.",
    )
    parser.add_argument("--json", type=Path, dest="json_output")
    parser.add_argument("--markdown", type=Path, dest="markdown_output")
    parser.add_argument(
        "--measurements",
        type=Path,
        help="Profiling JSONL from gist.eval.profiling (as written by the pod runner).",
    )
    parser.add_argument(
        "--measured-baseline",
        default="full",
        help="Condition in the measurement log to compare the others against.",
    )
    parser.add_argument(
        "--measured-stage",
        default="answer",
        help="Only aggregate rows from this stage; empty string aggregates all.",
    )
    parser.add_argument(
        "--drop-warmup",
        type=int,
        default=1,
        help="Successful calls to discard per condition before timing (CUDA warmup).",
    )
    parser.add_argument(
        "--measured-json", type=Path, help="Where to write the measured report JSON."
    )
    parser.add_argument(
        "--measured-markdown",
        type=Path,
        help="Where to write the measured report markdown.",
    )
    args = parser.parse_args(argv)

    profile = PROFILES[args.profile]
    if args.from_vision_report:
        vision = _conditions_from_vision_report(args.from_vision_report)
        # dense uniform is the baseline; add audio windows from CLI to each.
        conditions = {
            "dense": (vision.get("dense", (args.baseline_frames, 0))[0], args.baseline_audio_windows),
            "uniform-K": (vision.get("uniform", (args.uniform_frames, 0))[0], args.uniform_audio_windows),
            "gist-K": (vision.get("gist", (args.gist_frames, 0))[0], args.gist_audio_windows),
        }
        baseline = "dense"
    else:
        conditions = {
            "full baseline": (args.baseline_frames, args.baseline_audio_windows),
            "uniform-K": (args.uniform_frames, args.uniform_audio_windows),
            "gist-K": (args.gist_frames, args.gist_audio_windows),
        }
        baseline = "full baseline"
    report = build_report(conditions, profile, baseline=baseline)

    if args.json_output:
        report.write_json(args.json_output)
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(report))

    print(f"[analytic] profile={report.profile} baseline={report.baseline}")
    for c in report.conditions:
        saving = "" if c.system_gflops_saving_pct is None else f" saving={c.system_gflops_saving_pct:.1f}%"
        print(
            f"{c.condition}: frames={c.frames} windows={c.audio_windows} "
            f"system_gflops={c.system_gflops:,.0f} tokens={c.downstream_tokens:,}{saving}"
        )

    if args.measurements:
        device, stages = load_measurements(args.measurements)
        measured = build_measured_report(
            stages,
            baseline=args.measured_baseline,
            device=device,
            stage=args.measured_stage or None,
            drop_warmup=args.drop_warmup,
        )
        if args.measured_json:
            measured.write_json(args.measured_json)
        if args.measured_markdown:
            args.measured_markdown.parent.mkdir(parents=True, exist_ok=True)
            args.measured_markdown.write_text(render_measured_markdown(measured))
        print(f"\n[measured] device={measured.device} baseline={measured.baseline}")
        for c in measured.conditions:
            speed = (
                "" if c.speedup_vs_baseline is None
                else f" speedup={c.speedup_vs_baseline:.2f}x"
            )
            act = (
                "" if c.activation_gb_median is None
                else f" activation={c.activation_gb_median:.3f}GB"
            )
            oom = f" OOM={c.n_oom}" if c.n_oom else ""
            median = "—" if c.wall_seconds_median is None else f"{c.wall_seconds_median:.2f}s"
            print(f"{c.condition}: n={c.n} median={median}{speed}{act}{oom}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
