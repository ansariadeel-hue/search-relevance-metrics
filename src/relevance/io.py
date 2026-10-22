"""Readers for qrels, run and query files (simple TSV or TREC-style)."""

from __future__ import annotations

import re
from pathlib import Path

MIN_GRADE, MAX_GRADE = 0, 4
LOCALE_PREFIX = re.compile(r"^([a-z]{2,3}-[A-Z]{2})[-_:]")

Qrels = dict[str, dict[str, int]]  # query -> doc -> grade
Run = dict[str, list[str]]  # query -> ranked doc ids (best first)


class FormatError(ValueError):
    """Raised when an input file cannot be parsed."""


def _rows(path: Path):
    with Path(path).open(encoding="utf-8-sig") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.rstrip("\r\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split("\t") if "\t" in line else line.split()
            yield lineno, [f.strip() for f in fields]


def _int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def load_qrels(path: str | Path) -> Qrels:
    """Read graded judgments.

    Accepted layouts (tab- or whitespace-separated, ``#`` comments allowed, an
    optional header line is skipped):

    * ``query_id  doc_id  grade``
    * ``query_id  0  doc_id  grade`` (TREC qrels)

    Grades must be integers on the 0-4 scale.
    """
    path = Path(path)
    qrels: Qrels = {}
    first = True
    for lineno, f in _rows(path):
        if len(f) == 3:
            qid, doc, grade_s = f
        elif len(f) == 4:
            qid, _, doc, grade_s = f
        else:
            raise FormatError(f"{path}:{lineno}: expected 3 or 4 fields, got {len(f)}")
        grade = _int(grade_s)
        if grade is None:
            if first:  # header line
                first = False
                continue
            raise FormatError(f"{path}:{lineno}: grade {grade_s!r} is not an integer")
        first = False
        if not MIN_GRADE <= grade <= MAX_GRADE:
            raise FormatError(f"{path}:{lineno}: grade {grade} outside {MIN_GRADE}-{MAX_GRADE}")
        docs = qrels.setdefault(qid, {})
        if doc in docs and docs[doc] != grade:
            raise FormatError(f"{path}:{lineno}: conflicting grades for {qid}/{doc}")
        docs[doc] = grade
    if not qrels:
        raise FormatError(f"{path}: no judgments found")
    return qrels


def load_run(path: str | Path) -> Run:
    """Read a ranked run.

    Accepted layouts:

    * ``query_id  doc_id  rank`` - ordered by ascending rank
    * ``query_id  Q0  doc_id  rank  score  tag`` (TREC run) - ordered by
      descending score, ties broken by rank

    If a document appears twice for the same query only its best position is kept.
    """
    path = Path(path)
    entries: dict[str, list[tuple[float, int, str]]] = {}
    first = True
    for lineno, f in _rows(path):
        if len(f) == 3:
            qid, doc, rank_s = f
            rank = _int(rank_s)
            if rank is None:
                if first:
                    first = False
                    continue
                raise FormatError(f"{path}:{lineno}: rank {rank_s!r} is not an integer")
            key: tuple[float, int, str] = (float(rank), rank, doc)
        elif len(f) == 6:
            qid, _, doc, rank_s, score_s, _ = f
            try:
                score = float(score_s)
            except ValueError:
                if first:
                    first = False
                    continue
                raise FormatError(f"{path}:{lineno}: score {score_s!r} is not a number") from None
            rank = _int(rank_s) or 0
            key = (-score, rank, doc)
        else:
            raise FormatError(f"{path}:{lineno}: expected 3 or 6 fields, got {len(f)}")
        first = False
        entries.setdefault(qid, []).append(key)
    if not entries:
        raise FormatError(f"{path}: no results found")
    run: Run = {}
    for qid, rows in entries.items():
        seen: set[str] = set()
        ranked = []
        for _, _, doc in sorted(rows):
            if doc not in seen:
                seen.add(doc)
                ranked.append(doc)
        run[qid] = ranked
    return run


def load_queries(path: str | Path) -> dict[str, dict[str, str]]:
    """Read ``query_id  locale  query_text`` (header optional, UTF-8)."""
    path = Path(path)
    out: dict[str, dict[str, str]] = {}
    for lineno, f in _rows(path):
        if len(f) < 2:
            raise FormatError(f"{path}:{lineno}: expected query_id, locale[, text]")
        qid, locale = f[0], f[1]
        if (qid, locale) == ("query_id", "locale"):
            continue
        out[qid] = {"locale": locale, "text": "\t".join(f[2:]) if len(f) > 2 else ""}
    return out


def infer_locale(query_id: str, queries: dict[str, dict[str, str]] | None = None) -> str:
    """Locale from the queries file, else from an id prefix like ``hi-IN-003``."""
    if queries and query_id in queries:
        return queries[query_id]["locale"]
    m = LOCALE_PREFIX.match(query_id)
    return m.group(1) if m else "unknown"
