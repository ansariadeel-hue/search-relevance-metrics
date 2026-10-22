"""Compare two runs: per-query wins/losses and a paired randomization test."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from .evaluate import Evaluation

EPS = 1e-9


def paired_randomization_test(
    a: Sequence[float], b: Sequence[float], iterations: int = 10_000, seed: int = 42
) -> float:
    """Two-sided paired randomization (permutation) test on the mean difference.

    Under the null hypothesis the two systems are exchangeable for each query,
    so the sign of each per-query difference ``b_i - a_i`` can be flipped.

    * If ``2**n <= iterations`` all ``2**n`` sign assignments are enumerated
      and the exact p-value is returned (the share of assignments whose
      absolute mean difference is at least the observed one).
    * Otherwise ``iterations`` random assignments are drawn with a
      ``random.Random(seed)`` generator and the p-value is
      ``(c + 1) / (iterations + 1)`` (the +1 counts the observed assignment
      and keeps the estimate valid). A fixed seed makes results reproducible.
    """
    if len(a) != len(b):
        raise ValueError("score lists must be paired (same length)")
    if not a:
        raise ValueError("need at least one paired score")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    diffs = [y - x for x, y in zip(a, b)]
    n = len(diffs)
    observed = abs(sum(diffs)) / n
    if observed <= EPS:
        return 1.0
    if 2**n <= iterations:
        count = 0
        for mask in range(2**n):
            total = sum(-d if mask >> i & 1 else d for i, d in enumerate(diffs))
            if abs(total) / n >= observed - EPS:
                count += 1
        return count / 2**n
    rng = random.Random(seed)
    count = 0
    for _ in range(iterations):
        total = 0.0
        for d in diffs:
            total += d if rng.random() < 0.5 else -d
        if abs(total) / n >= observed - EPS:
            count += 1
    return (count + 1) / (iterations + 1)


@dataclass
class MetricComparison:
    metric: str
    mean_a: float
    mean_b: float
    wins: int  # queries where B beats A
    losses: int
    ties: int
    p_value: float

    @property
    def delta(self) -> float:
        return self.mean_b - self.mean_a

    @property
    def relative(self) -> float | None:
        return None if self.mean_a == 0 else self.delta / self.mean_a

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "delta": self.delta,
            "relative": self.relative,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "p_value": self.p_value,
        }


def win_loss_tie(a: Sequence[float], b: Sequence[float]) -> tuple[int, int, int]:
    wins = sum(1 for x, y in zip(a, b) if y - x > EPS)
    losses = sum(1 for x, y in zip(a, b) if x - y > EPS)
    return wins, losses, len(a) - wins - losses


def compare_metric(
    ev_a: Evaluation, ev_b: Evaluation, metric: str, queries: list[str], iterations: int, seed: int
) -> MetricComparison:
    a = [ev_a.per_query[q][metric] for q in queries]
    b = [ev_b.per_query[q][metric] for q in queries]
    w, l, t = win_loss_tie(a, b)
    p = paired_randomization_test(a, b, iterations, seed)
    return MetricComparison(metric, sum(a) / len(a), sum(b) / len(b), w, l, t, p)


def compare(
    ev_a: Evaluation,
    ev_b: Evaluation,
    primary: str = "nDCG@10",
    iterations: int = 10_000,
    seed: int = 42,
) -> dict:
    """Full comparison of run B against baseline run A (both evaluated on the same qrels)."""
    if set(ev_a.per_query) != set(ev_b.per_query):
        raise ValueError("evaluations cover different query sets")
    if primary not in ev_a.metric_names:
        raise ValueError(f"unknown metric {primary!r}; choose from {', '.join(ev_a.metric_names)}")
    queries = sorted(ev_a.per_query)
    overall = [compare_metric(ev_a, ev_b, m, queries, iterations, seed) for m in ev_a.metric_names]

    by_locale = {}
    groups: dict[str, list[str]] = {}
    for q in queries:
        groups.setdefault(ev_a.locales[q], []).append(q)
    for loc, qs in sorted(groups.items()):
        by_locale[loc] = {"n": len(qs), **compare_metric(ev_a, ev_b, primary, qs, iterations, seed).to_dict()}

    per_query = []
    for q in queries:
        a, b = ev_a.per_query[q][primary], ev_b.per_query[q][primary]
        outcome = "win" if b - a > EPS else "loss" if a - b > EPS else "tie"
        per_query.append({"query_id": q, "locale": ev_a.locales[q], "a": a, "b": b, "delta": b - a, "outcome": outcome})
    per_query.sort(key=lambda r: (-r["delta"], r["query_id"]))

    return {
        "primary_metric": primary,
        "n_queries": len(queries),
        "test": {"name": "paired randomization (sign-flip), two-sided", "iterations": iterations, "seed": seed},
        "overall": [m.to_dict() for m in overall],
        "by_locale": by_locale,
        "per_query": per_query,
    }
