"""Summarise the stepped k6 runs in raw/ into load.md, against budgets set before the runs.

Each file raw/<mode>-<vus>.json is one k6 summary at a constant number of virtual
users. For every level this reports throughput, latency percentiles per endpoint
and the error rate, and marks the level against the budget. The breaking point is
the lowest level at which either budget is exceeded.

Budgets were written into this file and committed before any load was applied
(see this file's git history), so they are not fitted to the results:

  query    P95 ≤ 5 s for the whole event stream of one question: stored-evidence
           scoring, selection, cutting the clips with ffmpeg and the extractive
           answer. Looser than the 3 s selection budget in
           efficiency-measured/latency.md because it includes the clip cutting.
  browse   P95 ≤ 300 ms for listing the library and opening a video, the figure
           the system testing workshop gives for a plain API read.
  errors   at most 1% of iterations may fail any check. A query counts as failed
           when its stream does not end with a `done` event, even with status 200.

    python results/load-test/analyze.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"

BUDGET_MS = {"query": 5000.0, "list": 300.0, "detail": 300.0}
ERROR_BUDGET = 0.01
ENDPOINTS = {"ask": ["query"], "varied": ["query"], "browse": ["list", "detail"]}


def level(path: Path) -> tuple[str, int]:
    match = re.fullmatch(r"(ask|varied|browse)-(\d+)\.json", path.name)
    if not match:
        raise ValueError(f"unexpected file {path.name}")
    return match.group(1), int(match.group(2))


def summarise(path: Path) -> dict:
    mode, vus = level(path)
    metrics = json.loads(path.read_text())["metrics"]
    checks = metrics.get("checks", {"passes": 0, "fails": 0})
    total_checks = checks["passes"] + checks["fails"]
    row = {
        "mode": mode,
        "vus": vus,
        "iterations": metrics.get("iterations", {}).get("count", 0),
        "req_per_s": metrics.get("http_reqs", {}).get("rate", 0.0),
        "error_rate": checks["fails"] / total_checks if total_checks else 1.0,
        "endpoints": {},
    }
    for name in ENDPOINTS[mode]:
        trend = metrics.get(f"http_req_duration{{name:{name}}}", {})
        if trend.get("count"):
            row["endpoints"][name] = trend
    over = [name for name, trend in row["endpoints"].items() if trend["p(95)"] > BUDGET_MS[name]]
    row["over_latency"] = over
    row["over_errors"] = row["error_rate"] > ERROR_BUDGET
    row["within"] = not over and not row["over_errors"]
    return row


def users(n: int) -> str:
    return f"{n} virtual user{'' if n == 1 else 's'}"


def fmt_s(ms: float) -> str:
    return f"{ms / 1000:.2f}"


def render(rows: list[dict], heading: str = "##") -> list[str]:
    out: list[str] = []
    for mode in ("ask", "varied", "browse"):
        group = sorted((r for r in rows if r["mode"] == mode), key=lambda r: r["vus"])
        if not group:
            continue
        names = ENDPOINTS[mode]
        title = {
            "ask": "Asking three repeated questions (clip cache warm)",
            "varied": "Asking 40 different questions (clip cache emptied per level)",
            "browse": "Browsing the library",
        }[mode]
        budgets = ", ".join(f"{n} P95 ≤ {fmt_s(BUDGET_MS[n])} s" for n in names)
        out += [f"{heading} {title}", "", f"Budget: {budgets}; errors ≤ {ERROR_BUDGET:.0%}.", ""]
        header = "| Virtual users | Iterations | Requests/s |"
        rule = "| --: | --: | --: |"
        for name in names:
            header += f" {name} P50 s | {name} P95 s | {name} P99 s |"
            rule += " --: | --: | --: |"
        header += " Errors | Within budget |"
        rule += " --: | :-- |"
        out += [header, rule]
        for r in group:
            line = f"| {r['vus']} | {r['iterations']} | {r['req_per_s']:.2f} |"
            for name in names:
                t = r["endpoints"].get(name)
                if t:
                    line += f" {fmt_s(t['med'])} | {fmt_s(t['p(95)'])} | {fmt_s(t['p(99)'])} |"
                else:
                    line += " — | — | — |"
            verdict = "yes" if r["within"] else "**no**"
            reasons = r["over_latency"] + (["errors"] if r["over_errors"] else [])
            if reasons:
                verdict += f" ({', '.join(reasons)})"
            line += f" {r['error_rate']:.1%} | {verdict} |"
            out.append(line)
        broken = next((r for r in group if not r["within"]), None)
        last_ok = max((r["vus"] for r in group if r["within"]), default=None)
        out.append("")
        if broken is None:
            out.append(f"**Breaking point:** not reached up to {users(group[-1]['vus'])}.")
        else:
            held = f"held up to {last_ok}" if last_ok is not None else "held at no tested level"
            out.append(
                f"**Breaking point:** {users(broken['vus'])}, the first level over "
                f"budget; the service {held}."
            )
        out.append("")
    return out


def main() -> None:
    after = [summarise(path) for path in sorted(RAW.glob("*.json"))]
    before = [summarise(path) for path in sorted((RAW / "before-fix").glob("*.json"))]
    lines = [
        "# Load test",
        "",
        "Generated by `analyze.py` from the k6 summaries in `raw/`. Setup, hardware and",
        "the limits of this test are in `README.md`.",
        "",
        "## Current code",
        "",
        *render(after, "###"),
    ]
    if before:
        lines += [
            "## Before the encoder fix (commit df563f2)",
            "",
            "The first run found every question reloading CLIP and CLAP from disk. These",
            "rows are that run, kept in `raw/before-fix/` as the record of the defect.",
            "",
            *render(before, "###"),
        ]
    text = "\n".join(lines)
    (HERE / "load.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
