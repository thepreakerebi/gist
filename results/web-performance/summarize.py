"""Summarise the Lighthouse reports and the network-condition runs into summary.md.

Inputs, all committed beside this file:

  lighthouse/<page>-<form>.report.json          Lighthouse 12.8.2, default
                                                (simulated) throttling
  lighthouse/<page>-mobile-devtools-<n>.report.json
                                                the same mobile profile applied
                                                for real through DevTools, 3 runs
  network.json                                  web/scripts/network-scenarios.mjs

Lighthouse's default mobile run does not slow the page down; it measures an
unthrottled load and simulates what a slow-4G phone would see. The DevTools runs
apply the throttling and measure, so where the two disagree the DevTools figure
is the observation and the simulated one is an estimate.

Budgets are Google's "good" thresholds for Core Web Vitals, which the system
testing workshop quotes: LCP ≤ 2.5 s, CLS ≤ 0.1.

    python results/web-performance/summarize.py
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
LH = HERE / "lighthouse"
LCP_BUDGET_MS = 2500.0
CLS_BUDGET = 0.1
STEPS = {
    "library_listed_s": "Library listed",
    "video_ready_s": "Video page usable",
    "evidence_shown_s": "Ask → evidence shown",
    "clip_playable_s": "Open evidence → clip playable",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def pct(score: float | None) -> str:
    return "—" if score is None else f"{round(score * 100)}"


def lighthouse_table() -> list[str]:
    lines = [
        "## Lighthouse (simulated throttling)",
        "",
        "| Page | Form factor | Performance | Accessibility | Best practices | SEO "
        "| FCP s | LCP s | TBT ms | CLS | LCP ≤ 2.5 s |",
        "| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: | :-- |",
    ]
    for page in ("library", "video"):
        for form in ("mobile", "desktop"):
            path = LH / f"{page}-{form}.report.json"
            if not path.exists():
                continue
            report = load(path)
            cats, audits = report["categories"], report["audits"]
            lcp = audits["largest-contentful-paint"]["numericValue"]
            cls = audits["cumulative-layout-shift"]["numericValue"]
            ok = lcp <= LCP_BUDGET_MS and cls <= CLS_BUDGET
            cells = [
                page,
                form,
                pct(cats["performance"]["score"]),
                pct(cats["accessibility"]["score"]),
                pct(cats["best-practices"]["score"]),
                pct(cats["seo"]["score"]),
                f"{audits['first-contentful-paint']['numericValue'] / 1000:.2f}",
                f"{lcp / 1000:.2f}",
                f"{audits['total-blocking-time']['numericValue']:.0f}",
                f"{cls:.3f}",
                "pass" if ok else "**fail** (estimate)",
            ]
            lines.append("| " + " | ".join(cells) + " |")
    return lines


def devtools_table() -> list[str]:
    lines = [
        "",
        "## Lighthouse mobile, throttling applied (DevTools), 3 runs per page",
        "",
        "| Page | LCP s per run | Median LCP s | Performance per run | LCP ≤ 2.5 s |",
        "| :-- | :-- | --: | :-- | :-- |",
    ]
    for page in ("library", "video"):
        runs = sorted(LH.glob(f"{page}-mobile-devtools-*.report.json"))
        if not runs:
            continue
        lcps, scores = [], []
        for path in runs:
            report = load(path)
            lcps.append(report["audits"]["largest-contentful-paint"]["numericValue"])
            scores.append(pct(report["categories"]["performance"]["score"]))
        median = statistics.median(lcps)
        verdict = "pass" if median <= LCP_BUDGET_MS else "**fail**"
        per_run = ", ".join(f"{x / 1000:.2f}" for x in lcps)
        lines.append(
            f"| {page} | {per_run} | {median / 1000:.2f} | {', '.join(scores)} | {verdict} |"
        )
    return lines


def network_table() -> list[str]:
    path = HERE / "network.json"
    if not path.exists():
        return []
    results = load(path)["results"]
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        groups[row["profile"]].append(row)
    lines = [
        "",
        "## User flow under network conditions (median of 3 runs, seconds)",
        "",
        "| Profile | Completed | " + " | ".join(STEPS.values()) + " |",
        "| :-- | :-- | " + " | ".join("--:" for _ in STEPS) + " |",
    ]
    for profile, rows in groups.items():
        ok = [r for r in rows if r["ok"]]
        cells = [profile, f"{len(ok)}/{len(rows)}"]
        for key in STEPS:
            cells.append(f"{statistics.median(r[key] for r in ok):.2f}" if ok else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def main() -> None:
    lines = [
        "# Web performance",
        "",
        "Generated by `summarize.py`. Setup, findings and limits are in `README.md`.",
        "",
        *lighthouse_table(),
        *devtools_table(),
        *network_table(),
        "",
    ]
    text = "\n".join(lines)
    (HERE / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
