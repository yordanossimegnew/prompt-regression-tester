"""Small, dependency-free statistics for deciding whether a change is a real regression.

LLM outputs vary run to run, so each case is generated several times and differences are
only called regressions when they are unlikely to be noise:

* fisher_less       one-sided Fisher exact test: is the candidate's pass rate lower?
* sign_test_less    one-sided binomial sign test on pairwise wins vs losses (ties dropped)
* bootstrap_diff    bootstrap CI for the mean per-case difference (e.g. judge scores)
"""

from __future__ import annotations

import random
from math import comb


def fisher_less(pass_a: int, n_a: int, pass_b: int, n_b: int) -> float:
    """P-value that B (candidate) passes less often than A (baseline), one-sided Fisher exact.

    2x2 table [[pass_a, fail_a], [pass_b, fail_b]]; sums the hypergeometric probability of
    every table with as few or fewer candidate passes given the margins.
    """
    total_pass, total = pass_a + pass_b, n_a + n_b
    if total == 0 or n_b == 0:
        return 1.0
    denom = comb(total, n_b)
    lo = max(0, total_pass - n_a)
    return min(1.0, sum(comb(total_pass, k) * comb(total - total_pass, n_b - k) for k in range(lo, pass_b + 1)) / denom)


def fisher_greater(pass_a: int, n_a: int, pass_b: int, n_b: int) -> float:
    """P-value that B passes MORE often than A (for reporting improvements)."""
    return fisher_less(n_a - pass_a, n_a, n_b - pass_b, n_b)


def sign_test_less(wins: int, losses: int) -> float:
    """P-value that the candidate loses more than it wins (two outcomes, p=0.5, ties excluded)."""
    n = wins + losses
    if n == 0:
        return 1.0
    return sum(comb(n, k) for k in range(0, wins + 1)) / 2 ** n


def bootstrap_diff(diffs: list[float], iterations: int = 4000, confidence: float = 0.95, seed: int = 0) -> dict:
    """Mean of paired differences with a percentile bootstrap confidence interval."""
    if not diffs:
        return {"mean": 0.0, "low": 0.0, "high": 0.0, "n": 0}
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(iterations))
    tail = (1 - confidence) / 2
    return {"mean": round(sum(diffs) / n, 4), "low": round(means[int(tail * iterations)], 4),
            "high": round(means[min(iterations - 1, int((1 - tail) * iterations))], 4), "n": n}
