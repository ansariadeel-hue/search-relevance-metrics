"""Plain-text rendering of evaluations and comparisons."""

from __future__ import annotations

from .evaluate import Evaluation


def _heading(title: str) -> list[str]:
    return ["", title, "-" * len(title)]


def _table(headers: list[str], rows: list[list[str]], left: int = 1) -> list[str]:
    """Left-align the first ``left`` columns, right-align the rest.

    A trailing ``text`` column (free-form query text, possibly Devanagari) is
    printed as-is, without padding.
    """
    free_text = headers[-1] == "text"
    widths = [max([len(h)] + [len(r[i]) for r in rows]) for i, h in enumerate(headers)]

    def fmt(cells: list[str]) -> str:
        out = []
        for i, c in enumerate(cells):
            if free_text and i == len(cells) - 1:
                out.append(c)
            elif i < left:
                out.append(f"{c:<{widths[i]}}")
            else:
                out.append(f"{c:>{widths[i]}}")
        return "  ".join(out).rstrip()

    return [fmt(headers)] + [fmt(r) for r in rows]


def _gain_text(gain: str) -> str:
    return "exponential (2^g - 1)" if gain == "exponential" else "linear (g)"


def format_evaluation(
    ev: Evaluation,
    qrels_path: str | None = None,
    run_path: str | None = None,
    queries: dict[str, dict[str, str]] | None = None,
    per_query: bool = True,
) -> str:
    s = ev.settings
    names = ev.metric_names
    lines = ["Search relevance evaluation", "=" * 27]
    if qrels_path or run_path:
        lines.append(f"Qrels: {qrels_path}   Run: {run_path}")
    lines.append(
        f"Queries: {len(ev.per_query)}   Relevant: grade >= {s.threshold} (0-4 scale)   "
        f"nDCG gain: {_gain_text(s.gain)}"
    )
    if ev.skipped_no_relevant:
        lines.append(f"Skipped (no relevant judgments): {', '.join(ev.skipped_no_relevant)}")
    if ev.missing_from_run:
        lines.append(f"Judged but not in run (scored 0): {', '.join(ev.missing_from_run)}")

    lines += _heading("Overall (mean over queries)")
    mean = ev.mean()
    lines += _table(["", *names], [["all", *(f"{mean[m]:.4f}" for m in names)]])

    lines += _heading("By locale")
    rows = [
        [loc, str(g["n"]), *(f"{g['mean'][m]:.4f}" for m in names)] for loc, g in ev.by_locale().items()
    ]
    lines += _table(["locale", "n", *names], rows)

    if per_query:
        lines += _heading("Per query")
        rows = []
        for q, sc in ev.per_query.items():
            row = [q, ev.locales[q], *(f"{sc[m]:.4f}" for m in names)]
            if queries:
                row.append(queries.get(q, {}).get("text", ""))
            rows.append(row)
        headers = ["query", "locale", *names] + (["text"] if queries else [])
        lines += _table(headers, rows, left=2)
    return "\n".join(lines)


def _p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _sig(p: float) -> str:
    return "**" if p < 0.01 else "*" if p < 0.05 else ""


def format_comparison(
    result: dict, run_a: str, run_b: str, queries: dict[str, dict[str, str]] | None = None, top: int = 0
) -> str:
    primary = result["primary_metric"]
    t = result["test"]
    lines = ["Run comparison", "=" * 14]
    lines.append(f"A (baseline): {run_a}")
    lines.append(f"B (candidate): {run_b}")
    lines.append(
        f"Queries: {result['n_queries']}   Test: {t['name']}, {t['iterations']} iterations, seed {t['seed']}"
    )
    lines.append("(p-values are exact by full enumeration whenever 2^n <= iterations)")

    lines += _heading("Overall (B - A; W/L/T = queries where B wins/loses/ties)")
    rows = []
    for m in result["overall"]:
        rel = "" if m["relative"] is None else f"{m['relative'] * 100:+.1f}%"
        rows.append([
            m["metric"], f"{m['mean_a']:.4f}", f"{m['mean_b']:.4f}", f"{m['delta']:+.4f}", rel,
            f"{m['wins']}/{m['losses']}/{m['ties']}", _p(m["p_value"]) + _sig(m["p_value"]),
        ])
    lines += _table(["metric", "A", "B", "delta", "rel", "W/L/T", "p"], rows)
    lines.append("(* p < 0.05, ** p < 0.01)")

    lines += _heading(f"By locale ({primary})")
    rows = []
    for loc, m in result["by_locale"].items():
        rows.append([
            loc, str(m["n"]), f"{m['mean_a']:.4f}", f"{m['mean_b']:.4f}", f"{m['delta']:+.4f}",
            f"{m['wins']}/{m['losses']}/{m['ties']}", _p(m["p_value"]) + _sig(m["p_value"]),
        ])
    lines += _table(["locale", "n", "A", "B", "delta", "W/L/T", "p"], rows)

    pq = result["per_query"]
    if top and len(pq) > 2 * top:  # show only the largest gains and drops
        pq = pq[:top] + pq[-top:]
        lines += _heading(f"Per query ({primary}): top {top} and bottom {top} by delta")
    else:
        lines += _heading(f"Per query ({primary}, sorted by delta)")
    rows = []
    for r in pq:
        row = [r["query_id"], r["locale"], f"{r['a']:.4f}", f"{r['b']:.4f}", f"{r['delta']:+.4f}", r["outcome"]]
        if queries:
            row.append(queries.get(r["query_id"], {}).get("text", ""))
        rows.append(row)
    headers = ["query", "locale", "A", "B", "delta", "result"] + (["text"] if queries else [])
    lines += _table(headers, rows, left=2)
    return "\n".join(lines)
