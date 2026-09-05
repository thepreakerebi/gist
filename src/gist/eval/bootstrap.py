"""Bootstrap confidence intervals for the evaluation results.

Section 3.9 of the proposal commits to reporting "accuracy differences with
bootstrap confidence intervals rather than point estimates, because the honest
reading of preliminary work is that a five-versus-six difference on eighteen
questions is within noise". This is that.

Two things make the analysis here the right shape:

*Paired resampling.* Every condition is evaluated on the same cases, so the
comparison is paired. Resampling cases (not outcomes independently per
condition) preserves the fact that a hard case is hard for everyone, which is
exactly the correlation an unpaired interval would throw away — and throwing it
away inflates the interval on the difference, making a real effect look like
noise.

*Non-inferiority, not superiority.* The pre-registered criterion is that Gist's
accuracy sits within one standard error of the dense baseline. That is a claim
about the difference being *small*, so it is tested against the lower bound of
the interval on the difference, not by asking whether zero is excluded.

Percentile intervals are used rather than BCa: at n=18 to n=39 the bias
correction is estimated from the same handful of points it is meant to correct,
and the added machinery buys precision the sample size cannot support.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 0


@dataclass(frozen=True, slots=True)
class Interval:
    point: float
    low: float
    high: float

    def as_percent(self) -> str:
        return f"{self.point:.1%} [{self.low:.1%}, {self.high:.1%}]"

    def as_points(self) -> str:
        """Render a difference in percentage points, signed."""
        return f"{self.point:+.1%} [{self.low:+.1%}, {self.high:+.1%}]"


@dataclass(frozen=True, slots=True)
class Comparison:
    name: str
    baseline: str
    n: int
    a: Interval
    b: Interval
    difference: Interval
    standard_error: float
    agreement: float

    @property
    def non_inferior(self) -> bool:
        """True when the difference's lower bound clears -1 SE of the baseline.

        The proposal's criterion: accuracy within one standard error of dense
        full-context. A difference whose plausible range does not extend below
        that margin satisfies it.
        """

        return self.difference.low >= -self.standard_error


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return math.nan
    position = q * (len(sorted_values) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[int(position)]
    weight = position - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


def proportion_interval(
    outcomes: Sequence[bool],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> Interval:
    """Percentile bootstrap interval for a single pass rate."""

    n = len(outcomes)
    if n == 0:
        return Interval(math.nan, math.nan, math.nan)

    values = [1.0 if outcome else 0.0 for outcome in outcomes]
    rng = random.Random(seed)
    draws = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)
    )
    return Interval(
        point=sum(values) / n,
        low=_percentile(draws, alpha / 2),
        high=_percentile(draws, 1 - alpha / 2),
    )


def paired_difference(
    a: Sequence[bool],
    b: Sequence[bool],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> Interval:
    """Percentile bootstrap interval for (a - b), resampling cases together."""

    if len(a) != len(b):
        raise ValueError("paired comparison needs equal-length outcome vectors")
    n = len(a)
    if n == 0:
        return Interval(math.nan, math.nan, math.nan)

    deltas = [(1.0 if x else 0.0) - (1.0 if y else 0.0) for x, y in zip(a, b, strict=True)]
    rng = random.Random(seed)
    draws = sorted(
        sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)
    )
    return Interval(
        point=sum(deltas) / n,
        low=_percentile(draws, alpha / 2),
        high=_percentile(draws, 1 - alpha / 2),
    )


def standard_error(outcomes: Sequence[bool]) -> float:
    """Binomial standard error of a pass rate — the proposal's margin unit."""

    n = len(outcomes)
    if n == 0:
        return math.nan
    p = sum(1 for outcome in outcomes if outcome) / n
    return math.sqrt(p * (1 - p) / n)


def compare(
    name: str,
    baseline_name: str,
    a: Sequence[bool],
    b: Sequence[bool],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Comparison:
    return Comparison(
        name=name,
        baseline=baseline_name,
        n=len(a),
        a=proportion_interval(a, resamples=resamples, seed=seed),
        b=proportion_interval(b, resamples=resamples, seed=seed),
        difference=paired_difference(a, b, resamples=resamples, seed=seed),
        standard_error=standard_error(b),
        # How often the two conditions agree case-by-case. A high number with a
        # near-zero difference means the conditions are behaving alike, not that
        # they are trading wins and losses that happen to cancel.
        agreement=(
            sum(1 for x, y in zip(a, b, strict=True) if x == y) / len(a) if a else math.nan
        ),
    )


# ----------------------------------------------------------------- inputs ----


def outcomes_from_ablation(
    report: Path,
    *,
    only_cases: set[str] | None = None,
) -> dict[str, list[bool]]:
    """Read per-case pass/fail per mode from an ablation report."""

    payload = json.loads(report.read_text())
    results = payload["results"]
    if only_cases is not None:
        results = [row for row in results if row["case_id"] in only_cases]

    modes = list(results[0]["outcomes"]) if results else []
    return {
        mode: [bool(row["outcomes"][mode]["passed"]) for row in results] for mode in modes
    }


def outcomes_from_predictions(
    path: Path,
    *,
    gold_field: str = "gold",
    conditions: Sequence[str] = ("full", "gist"),
) -> dict[str, list[bool]]:
    """Read correctness from a wide benchmark JSONL.

    One row per question carrying the gold answer alongside each condition's
    prediction — the shape the RunPod 7B runs emit. Rows stay in file order, so
    the conditions remain aligned question-by-question and the comparison is
    paired.
    """

    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {
        condition: [
            str(row.get(condition, "")).strip().upper()
            == str(row.get(gold_field, "")).strip().upper()
            for row in rows
        ]
        for condition in conditions
    }


def outcomes_from_jsonl(path: Path, field: str) -> dict[str, list[bool]]:
    """Read per-question correctness from a benchmark JSONL.

    Rows are grouped by their ``condition`` and aligned on ``question_id`` so the
    comparison stays paired even when the file interleaves conditions.
    """

    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_condition: dict[str, dict[str, bool]] = {}
    for row in rows:
        condition = str(row.get("condition") or row.get("mode") or "unknown")
        key = str(row.get("question_id") or row.get("id") or len(by_condition.get(condition, {})))
        by_condition.setdefault(condition, {})[key] = bool(row.get(field))

    shared = set.intersection(*(set(v) for v in by_condition.values())) if by_condition else set()
    order = sorted(shared)
    return {
        condition: [values[key] for key in order] for condition, values in by_condition.items()
    }


# ------------------------------------------------------------------ report ---


def render(
    comparisons: list[Comparison],
    *,
    title: str,
    non_inferiority: bool = False,
) -> str:
    """Render the comparison table.

    ``non_inferiority`` is opt-in because the pre-registered criterion applies
    to exactly one comparison — Gist against the dense full-context baseline.
    Printing it beside ablation arms that are *supposed* to be worse invites
    reading a meaningless "no" as a finding.
    """

    lines = [f"# {title}", ""]
    lines.append(
        "Percentile bootstrap, 10,000 resamples, cases resampled together so the "
        "comparison stays paired. Intervals are 95%."
    )
    lines.append("")
    head = "| Condition | Pass rate [95% CI] | vs baseline (pp) | Agreement |"
    rule = "| :--- | :--- | :--- | ---: |"
    if non_inferiority:
        head += " Non-inferior |"
        rule += " :--- |"
    lines += [head, rule]

    for c in comparisons:
        row = (
            f"| {c.name} | {c.a.as_percent()} | {c.difference.as_points()} "
            f"| {c.agreement:.0%} |"
        )
        if non_inferiority:
            row += f" {'yes' if c.non_inferior else 'no'} |"
        lines.append(row)

    if comparisons:
        first = comparisons[0]
        lines += ["", f"Baseline: **{first.baseline}**, {first.b.as_percent()}, n={first.n}."]
        if non_inferiority:
            lines[-1] += (
                f" Non-inferiority margin is one standard error of the baseline "
                f"(±{first.standard_error:.1%}), as pre-registered."
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", type=Path, help="Ablation report JSON.")
    parser.add_argument("--jsonl", type=Path, help="Benchmark JSONL of per-question rows.")
    parser.add_argument("--field", default="correct", help="Boolean field in --jsonl rows.")
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Wide JSONL: one row per question with a gold answer and per-condition picks.",
    )
    parser.add_argument("--gold-field", default="gold")
    parser.add_argument(
        "--conditions",
        default="full,gist",
        help="Comma-separated prediction fields to compare, for --predictions.",
    )
    parser.add_argument(
        "--non-inferiority",
        action="store_true",
        help="Report the pre-registered non-inferiority verdict. Only meaningful "
        "when the baseline is the dense full-context condition.",
    )
    parser.add_argument("--baseline", required=True, help="Condition to compare against.")
    parser.add_argument("--split", type=Path, help="Restrict to a frozen split manifest.")
    parser.add_argument("--use", choices=["dev", "held_out"], default="dev")
    parser.add_argument("--title", default="Bootstrap confidence intervals")
    parser.add_argument("--markdown", type=Path, dest="markdown_output")
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    only: set[str] | None = None
    if args.split is not None:
        manifest = json.loads(args.split.read_text())
        only = set(manifest[args.use]["case_ids"])

    if args.ablation:
        outcomes = outcomes_from_ablation(args.ablation, only_cases=only)
    elif args.predictions:
        outcomes = outcomes_from_predictions(
            args.predictions,
            gold_field=args.gold_field,
            conditions=tuple(c.strip() for c in args.conditions.split(",") if c.strip()),
        )
    elif args.jsonl:
        outcomes = outcomes_from_jsonl(args.jsonl, args.field)
    else:
        raise SystemExit("pass --ablation, --predictions or --jsonl")

    if args.baseline not in outcomes:
        raise SystemExit(f"baseline {args.baseline!r} not in {sorted(outcomes)}")

    baseline = outcomes[args.baseline]
    comparisons = [
        compare(name, args.baseline, values, baseline, resamples=args.resamples, seed=args.seed)
        for name, values in outcomes.items()
        if name != args.baseline
    ]
    comparisons.sort(key=lambda c: -c.a.point)

    report = render(comparisons, title=args.title, non_inferiority=args.non_inferiority)
    print(report)
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
