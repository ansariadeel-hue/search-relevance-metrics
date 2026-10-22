"""Command-line interface.

    relevance eval    qrels.tsv run.tsv
    relevance compare qrels.tsv runA.tsv runB.tsv
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .compare import compare
from .evaluate import DEFAULT_KS, DEFAULT_THRESHOLD, Settings, evaluate
from .io import FormatError, load_qrels, load_queries, load_run
from .report import format_comparison, format_evaluation


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("-k", type=int, nargs="+", default=list(DEFAULT_KS),
                   help="rank cut-offs for P@k, R@k and nDCG@k (default: 5 10)")
    p.add_argument("--rel-threshold", type=int, default=DEFAULT_THRESHOLD, choices=range(1, 5),
                   metavar="{1..4}",
                   help=f"minimum grade counted as relevant for P/R/MRR/MAP (default: {DEFAULT_THRESHOLD})")
    p.add_argument("--gain", choices=["exponential", "linear"], default="exponential",
                   help="nDCG gain: 2^g - 1 (default) or g")
    p.add_argument("--queries", help="optional TSV: query_id, locale, query_text")
    p.add_argument("--json", action="store_true", help="emit JSON instead of text")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="relevance", description="Offline search-relevance evaluation from graded (0-4) judgments."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("eval", help="score one run against qrels")
    ev.add_argument("qrels")
    ev.add_argument("run")
    ev.add_argument("--no-per-query", action="store_true", help="hide the per-query table")
    _common(ev)

    cmp_ = sub.add_parser("compare", help="compare run B against baseline run A")
    cmp_.add_argument("qrels")
    cmp_.add_argument("run_a")
    cmp_.add_argument("run_b")
    cmp_.add_argument("--metric", default=None,
                      help="primary metric for per-query wins/losses (default: nDCG@<largest k>)")
    cmp_.add_argument("--iterations", type=int, default=10_000, help="permutations (default: 10000)")
    cmp_.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    cmp_.add_argument("--top", type=int, default=0,
                      help="show only the N biggest wins and N biggest losses per query (default: all)")
    _common(cmp_)
    return parser


def _configure_stdout() -> None:
    # Query text may be Hindi/Marathi; make sure Windows consoles and pipes get UTF-8.
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            pass


def main(argv: list[str] | None = None) -> int:
    _configure_stdout()
    args = build_parser().parse_args(argv)
    settings = Settings(ks=tuple(sorted(set(args.k))), threshold=args.rel_threshold, gain=args.gain)
    try:
        qrels = load_qrels(args.qrels)
        queries = load_queries(args.queries) if args.queries else None
        if args.command == "eval":
            ev = evaluate(qrels, load_run(args.run), settings, queries)
            if args.json:
                print(json.dumps(ev.to_dict(), indent=2, ensure_ascii=False))
            else:
                print(format_evaluation(ev, args.qrels, args.run, queries, per_query=not args.no_per_query))
            return 0

        ev_a = evaluate(qrels, load_run(args.run_a), settings, queries)
        ev_b = evaluate(qrels, load_run(args.run_b), settings, queries)
        primary = args.metric or f"nDCG@{max(settings.ks)}"
        result = compare(ev_a, ev_b, primary=primary, iterations=args.iterations, seed=args.seed)
        if args.json:
            print(json.dumps({"run_a": args.run_a, "run_b": args.run_b, **result}, indent=2, ensure_ascii=False))
        else:
            print(format_comparison(result, args.run_a, args.run_b, queries, top=args.top))
        return 0
    except (FormatError, ValueError, OSError) as exc:
        print(f"relevance: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
