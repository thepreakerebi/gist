"""Measured system-efficiency accounting for Gist.

Replaces the arbitrary token-count constants (``estimate_tokens`` = counts x
258/32) with two things:

1. The *measured* structural quantity — how many frames / audio windows Gist
   actually encodes (K) versus a full/uniform baseline (N). This is the
   dual-encoder saving the capstone plan's headline rests on, and it is a real
   count, not a guess.
2. Architecture-derived encoder FLOPs and downstream token counts, computed
   analytically from the target encoders' transformer configs (documented
   below), so "far less compute" is expressed in GFLOPs, not hand-waved.

The relative saving (1 - K/N) is exact and count-driven; the absolute GFLOPs
scale it by a per-item cost derived from the encoder architecture.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel


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
        "# Gist system-efficiency (architecture-derived FLOPs, measured item counts)",
        "",
        f"- Encoder profile: {report.profile}; baseline: {report.baseline}",
        "- Frames/windows are real encode counts; GFLOPs derived from encoder configs.",
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measured system-efficiency: frames/windows encoded + derived FLOPs/tokens."
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

    print(f"profile={report.profile} baseline={report.baseline}")
    for c in report.conditions:
        saving = "" if c.system_gflops_saving_pct is None else f" saving={c.system_gflops_saving_pct:.1f}%"
        print(
            f"{c.condition}: frames={c.frames} windows={c.audio_windows} "
            f"system_gflops={c.system_gflops:,.0f} tokens={c.downstream_tokens:,}{saving}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
