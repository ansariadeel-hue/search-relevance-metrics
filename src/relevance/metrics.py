"""Ranking metrics over a single query.

``ranking`` is the list of retrieved doc ids, best first. ``judgments`` maps
doc id -> graded relevance (0-4). Unjudged documents count as grade 0.
Binary metrics treat a document as relevant when ``grade >= threshold``.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

GAINS = ("exponential", "linear")


def _relevant(judgments: Mapping[str, int], threshold: int) -> set[str]:
    return {d for d, g in judgments.items() if g >= threshold}


def precision_at_k(ranking: Sequence[str], judgments: Mapping[str, int], k: int, threshold: int = 1) -> float:
    """Relevant documents in the top k divided by k (missing ranks count as non-relevant)."""
    if k <= 0:
        raise ValueError("k must be positive")
    rel = _relevant(judgments, threshold)
    return sum(1 for d in ranking[:k] if d in rel) / k


def recall_at_k(ranking: Sequence[str], judgments: Mapping[str, int], k: int, threshold: int = 1) -> float:
    """Relevant documents in the top k divided by all judged-relevant documents."""
    if k <= 0:
        raise ValueError("k must be positive")
    rel = _relevant(judgments, threshold)
    if not rel:
        return 0.0
    return sum(1 for d in ranking[:k] if d in rel) / len(rel)


def reciprocal_rank(ranking: Sequence[str], judgments: Mapping[str, int], threshold: int = 1) -> float:
    """1 / rank of the first relevant document (0 if none is retrieved)."""
    rel = _relevant(judgments, threshold)
    for i, d in enumerate(ranking, start=1):
        if d in rel:
            return 1.0 / i
    return 0.0


def average_precision(ranking: Sequence[str], judgments: Mapping[str, int], threshold: int = 1) -> float:
    """Average precision over the full ranking, normalised by all relevant documents."""
    rel = _relevant(judgments, threshold)
    if not rel:
        return 0.0
    hits = 0
    total = 0.0
    for i, d in enumerate(ranking, start=1):
        if d in rel:
            hits += 1
            total += hits / i
    return total / len(rel)


def gain(grade: int, kind: str = "exponential") -> float:
    """Gain of a grade: ``2**g - 1`` (exponential) or ``g`` (linear)."""
    if kind == "exponential":
        return 2.0 ** grade - 1.0
    if kind == "linear":
        return float(grade)
    raise ValueError(f"unknown gain {kind!r}; expected one of {GAINS}")


def dcg_from_grades(grades: Sequence[int], k: int | None = None, kind: str = "exponential") -> float:
    """DCG = sum_i gain(g_i) / log2(i + 1) over the first k grades (rank i from 1)."""
    if k is not None:
        grades = grades[:k]
    return sum(gain(g, kind) / math.log2(i + 1) for i, g in enumerate(grades, start=1))


def dcg_at_k(ranking: Sequence[str], judgments: Mapping[str, int], k: int, kind: str = "exponential") -> float:
    return dcg_from_grades([judgments.get(d, 0) for d in ranking[:k]], k, kind)


def ndcg_at_k(ranking: Sequence[str], judgments: Mapping[str, int], k: int, kind: str = "exponential") -> float:
    """DCG@k divided by the DCG@k of the ideal ordering of *all* judged documents."""
    if k <= 0:
        raise ValueError("k must be positive")
    ideal = dcg_from_grades(sorted(judgments.values(), reverse=True), k, kind)
    if ideal == 0:
        return 0.0
    return dcg_at_k(ranking, judgments, k, kind) / ideal
