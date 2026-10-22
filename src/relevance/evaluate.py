"""Evaluate a run against qrels: per-query scores, overall and per-locale means."""

from __future__ import annotations

from dataclasses import dataclass, field

from . import metrics
from .io import Qrels, Run, infer_locale

DEFAULT_KS = (5, 10)
DEFAULT_THRESHOLD = 2  # on the 0-4 scale, "good" (2) and above counts as relevant


@dataclass
class Settings:
    ks: tuple[int, ...] = DEFAULT_KS
    threshold: int = DEFAULT_THRESHOLD
    gain: str = "exponential"

    def metric_names(self) -> list[str]:
        names = [f"P@{k}" for k in self.ks]
        names += [f"R@{k}" for k in self.ks]
        names += ["MRR", "MAP"]
        names += [f"nDCG@{k}" for k in self.ks]
        return names


@dataclass
class Evaluation:
    settings: Settings
    per_query: dict[str, dict[str, float]]
    locales: dict[str, str]  # query -> locale
    skipped_no_relevant: list[str] = field(default_factory=list)
    missing_from_run: list[str] = field(default_factory=list)

    @property
    def metric_names(self) -> list[str]:
        return self.settings.metric_names()

    def mean(self, queries: list[str] | None = None) -> dict[str, float]:
        qs = list(self.per_query) if queries is None else queries
        if not qs:
            return {m: 0.0 for m in self.metric_names}
        return {m: sum(self.per_query[q][m] for q in qs) / len(qs) for m in self.metric_names}

    def by_locale(self) -> dict[str, dict]:
        groups: dict[str, list[str]] = {}
        for q in self.per_query:
            groups.setdefault(self.locales[q], []).append(q)
        return {loc: {"n": len(qs), "mean": self.mean(qs)} for loc, qs in sorted(groups.items())}

    def to_dict(self) -> dict:
        return {
            "settings": {
                "k": list(self.settings.ks),
                "relevance_threshold": self.settings.threshold,
                "gain": self.settings.gain,
            },
            "n_queries": len(self.per_query),
            "mean": self.mean(),
            "by_locale": self.by_locale(),
            "per_query": {q: {"locale": self.locales[q], **s} for q, s in self.per_query.items()},
            "skipped_no_relevant": self.skipped_no_relevant,
            "missing_from_run": self.missing_from_run,
        }


def score_query(ranking: list[str], judgments: dict[str, int], s: Settings) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in s.ks:
        out[f"P@{k}"] = metrics.precision_at_k(ranking, judgments, k, s.threshold)
    for k in s.ks:
        out[f"R@{k}"] = metrics.recall_at_k(ranking, judgments, k, s.threshold)
    out["MRR"] = metrics.reciprocal_rank(ranking, judgments, s.threshold)
    out["MAP"] = metrics.average_precision(ranking, judgments, s.threshold)
    for k in s.ks:
        out[f"nDCG@{k}"] = metrics.ndcg_at_k(ranking, judgments, k, s.gain)
    return out


def evaluate(
    qrels: Qrels,
    run: Run,
    settings: Settings | None = None,
    queries: dict[str, dict[str, str]] | None = None,
) -> Evaluation:
    """Score every judged query.

    Queries are taken from the qrels (queries only present in the run are
    ignored, as in trec_eval). Queries without any document at or above the
    relevance threshold are skipped because recall, MAP and nDCG are undefined
    for them. Judged queries missing from the run score zero on every metric.
    """
    s = settings or Settings()
    if s.gain not in metrics.GAINS:
        raise ValueError(f"unknown gain {s.gain!r}")
    if not s.ks or any(k <= 0 for k in s.ks):
        raise ValueError("cut-offs k must be positive")
    per_query: dict[str, dict[str, float]] = {}
    locales: dict[str, str] = {}
    skipped, missing = [], []
    for qid in sorted(qrels):
        judgments = qrels[qid]
        if not any(g >= s.threshold for g in judgments.values()):
            skipped.append(qid)
            continue
        if qid not in run:
            missing.append(qid)
        per_query[qid] = score_query(run.get(qid, []), judgments, s)
        locales[qid] = infer_locale(qid, queries)
    if not per_query:
        raise ValueError("no query in the qrels has a relevant document; nothing to evaluate")
    return Evaluation(s, per_query, locales, skipped, missing)
