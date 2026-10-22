# search-relevance-metrics

**Offline search-ranking evaluation from graded human relevance judgments: P@k, R@k, MRR, MAP, NDCG, per-locale breakdowns and paired significance tests.**

[![CI](https://github.com/ansariadeel-hue/search-relevance-metrics/actions/workflows/ci.yml/badge.svg)](https://github.com/ansariadeel-hue/search-relevance-metrics/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

## Why

Search-quality rating work produces graded judgments: a rater reads a query
and a result and assigns a grade on a scale such as 0 (bad / off-topic) to
4 (perfect / exactly what the user wanted). Those judgments only become
useful once they are turned into ranking metrics and used to answer
concrete questions:

- *Is the candidate ranker actually better than the baseline, or is the
  difference noise?* That needs per-query scores and a paired significance test,
  not just two averages.
- *Where* is it better? In multilingual markets an overall gain can hide a
  regression in one language, so results should be broken down by locale
  (e.g. `en-IN`, `hi-IN`, `mr-IN`).
- *Which queries* moved? Wins and losses per query are what you take back to
  the raters and engineers.

`relevance` does this from two plain TSV files, with only the Python standard
library, and prints Hindi/Marathi query text correctly (UTF-8 throughout).

## Features

- **Precision@k, Recall@k, MRR, MAP** with a configurable relevance threshold on the 0-4 scale (default: grade >= 2)
- **DCG / NDCG@k** with **exponential** (`2^g - 1`, default) or **linear** (`g`) gain; ideal DCG uses all judged documents for the query
- Several cut-offs at once (`-k 3 5 10`)
- **Per-query** and **per-locale** breakdowns; locales come from a queries file or from ids like `hi-IN-03`
- **Run comparison**: mean of each run, delta, relative change, per-query **wins / losses / ties**
- **Paired randomization (sign-flip permutation) test**, two-sided, with a fixed seed; exact enumeration when the query count is small
- Simple TSV or TREC-style qrels/run files; unjudged documents count as grade 0
- `--json` output for dashboards and CI gates
- Bundled **English, Hindi and Marathi** example queries

## Install

```bash
pip install git+https://github.com/ansariadeel-hue/search-relevance-metrics
```

Requires Python 3.10+. No third-party runtime dependencies.

## Quickstart

The bundled example data is **synthetic**: 15 queries (5 each in `en-IN`, `hi-IN`
and `mr-IN`), 12 graded documents per query, and two simulated runs, a
`baseline` and a `candidate`, generated to illustrate the tool, not results from
any real search engine.

### Evaluate one run

```bash
relevance eval examples/qrels.tsv examples/run_baseline.tsv
```

```text
Search relevance evaluation
===========================
Qrels: examples/qrels.tsv   Run: examples/run_baseline.tsv
Queries: 15   Relevant: grade >= 2 (0-4 scale)   nDCG gain: exponential (2^g - 1)

Overall (mean over queries)
---------------------------
        P@5    P@10     R@5    R@10     MRR     MAP  nDCG@5  nDCG@10
all  0.6400  0.4800  0.5805  0.8762  0.8333  0.6575  0.6202   0.7095

By locale
---------
locale  n     P@5    P@10     R@5    R@10     MRR     MAP  nDCG@5  nDCG@10
en-IN   5  0.7600  0.5200  0.6910  0.9381  1.0000  0.8310  0.7790   0.8193
hi-IN   5  0.5200  0.4600  0.4333  0.8190  0.7000  0.5302  0.4663   0.6107
mr-IN   5  0.6400  0.4600  0.6171  0.8714  0.8000  0.6114  0.6153   0.6984

Per query
---------
query     locale     P@5    P@10     R@5    R@10     MRR     MAP  nDCG@5  nDCG@10
en-IN-01  en-IN   0.8000  0.5000  0.6667  0.8333  1.0000  0.7306  0.8876   0.8652
en-IN-02  en-IN   0.8000  0.5000  0.8000  1.0000  1.0000  0.9667  0.8303   0.8610
en-IN-03  en-IN   0.8000  0.6000  0.6667  1.0000  1.0000  0.9107  0.8531   0.8652
en-IN-04  en-IN   0.8000  0.6000  0.5714  0.8571  1.0000  0.6722  0.8278   0.8232
en-IN-05  en-IN   0.6000  0.4000  0.7500  1.0000  1.0000  0.8750  0.4961   0.6819
hi-IN-01  hi-IN   0.8000  0.8000  0.5000  1.0000  1.0000  0.8755  0.4894   0.7192
hi-IN-02  hi-IN   0.6000  0.5000  0.4286  0.7143  1.0000  0.5423  0.4577   0.5399
hi-IN-03  hi-IN   0.0000  0.2000  0.0000  0.6667  0.1667  0.1508  0.1568   0.4416
hi-IN-04  hi-IN   0.8000  0.5000  0.5714  0.7143  1.0000  0.6619  0.7342   0.7879
hi-IN-05  hi-IN   0.4000  0.3000  0.6667  1.0000  0.3333  0.4206  0.4935   0.5648
mr-IN-01  mr-IN   0.8000  0.6000  0.5714  0.8571  0.5000  0.5726  0.6737   0.6926
mr-IN-02  mr-IN   0.4000  0.4000  0.5000  1.0000  1.0000  0.5583  0.8252   0.8926
mr-IN-03  mr-IN   0.2000  0.1000  0.5000  0.5000  0.5000  0.2500  0.3741   0.4734
mr-IN-04  mr-IN   0.8000  0.5000  0.8000  1.0000  1.0000  0.7433  0.5825   0.7043
mr-IN-05  mr-IN   1.0000  0.7000  0.7143  1.0000  1.0000  0.9325  0.6212   0.7290
```

### Compare two runs

```bash
relevance compare examples/qrels.tsv examples/run_baseline.tsv examples/run_candidate.tsv \
    --queries examples/queries.tsv --top 3
```

```text
Run comparison
==============
A (baseline): examples/run_baseline.tsv
B (candidate): examples/run_candidate.tsv
Queries: 15   Test: paired randomization (sign-flip), two-sided, 10000 iterations, seed 42
(p-values are exact by full enumeration whenever 2^n <= iterations)

Overall (B - A; W/L/T = queries where B wins/loses/ties)
--------------------------------------------------------
metric        A       B    delta     rel   W/L/T        p
P@5      0.6400  0.7733  +0.1333  +20.8%   8/1/6   0.031*
P@10     0.4800  0.5133  +0.0333   +6.9%   7/3/5    0.272
R@5      0.5805  0.7245  +0.1440  +24.8%   8/1/6    0.072
R@10     0.8762  0.9504  +0.0742   +8.5%   7/3/5    0.202
MRR      0.8333  1.0000  +0.1667  +20.0%  4/0/11    0.120
MAP      0.6575  0.8532  +0.1957  +29.8%  12/3/0  0.006**
nDCG@5   0.6202  0.8376  +0.2174  +35.1%  13/2/0  0.001**
nDCG@10  0.7095  0.8701  +0.1607  +22.6%  13/2/0  0.002**
(* p < 0.05, ** p < 0.01)

By locale (nDCG@10)
-------------------
locale  n       A       B    delta  W/L/T      p
en-IN   5  0.8193  0.8968  +0.0775  4/1/0  0.312
hi-IN   5  0.6107  0.8734  +0.2627  5/0/0  0.062
mr-IN   5  0.6984  0.8403  +0.1419  4/1/0  0.188

Per query (nDCG@10): top 3 and bottom 3 by delta
------------------------------------------------
query     locale       A       B    delta  result  text
hi-IN-03  hi-IN   0.4416  0.9101  +0.4685     win  पैन कार्ड ऑनलाइन आवेदन कैसे करें
hi-IN-02  hi-IN   0.5399  0.9592  +0.4193     win  15000 के अंदर सबसे अच्छा स्मार्टफोन
mr-IN-03  mr-IN   0.4734  0.8033  +0.3299     win  आधार कार्ड पत्ता बदल कसा करायचा
en-IN-03  en-IN   0.8652  0.8679  +0.0027     win  how to apply for pan card online
en-IN-01  en-IN   0.8652  0.8073  -0.0579    loss  cheap flights delhi to mumbai
mr-IN-02  mr-IN   0.8926  0.7768  -0.1157    loss  आजच्या मराठी बातम्या
```

Reading it: the candidate wins 13 of 15 queries on nDCG@10 and the gain is
significant overall (p = 0.002), with the largest gain on `hi-IN`. Note that
with only 5 queries per locale the smallest attainable two-sided p-value is
2/2^5 = 0.0625, so per-locale results need more queries before they can
reach significance on their own.

### Options

| option | meaning |
| --- | --- |
| `-k 5 10` | cut-offs for P@k, R@k and nDCG@k |
| `--rel-threshold 2` | minimum grade counted as relevant for P/R/MRR/MAP |
| `--gain {exponential,linear}` | nDCG gain function |
| `--queries queries.tsv` | locale + query text per query id |
| `--metric nDCG@10` | (`compare`) primary metric for per-query wins/losses and locale table |
| `--iterations 10000 --seed 42` | (`compare`) permutation-test settings |
| `--top N` | (`compare`) only show the N largest and N smallest per-query deltas |
| `--no-per-query` | (`eval`) hide the per-query table |
| `--json` | machine-readable output |

### Python API

```python
from relevance import load_qrels, load_run, evaluate, compare, ndcg_at_k

qrels = load_qrels("examples/qrels.tsv")
base = evaluate(qrels, load_run("examples/run_baseline.tsv"))
cand = evaluate(qrels, load_run("examples/run_candidate.tsv"))
base.mean()["nDCG@10"]                      # 0.7095...
compare(base, cand)["overall"]              # per-metric deltas, W/L/T, p-values
ndcg_at_k(["d1", "d2"], {"d1": 2, "d2": 4}, k=2, kind="linear")
```

## Data formats

All files are UTF-8, tab-separated (whitespace also accepted), `#` comments and a header line are allowed.

**Qrels** (`query_id  doc_id  grade`), grades are integers 0-4; the TREC layout `query_id 0 doc_id grade` is also accepted:

```text
query_id	doc_id	grade
en-IN-01	d0101	0
en-IN-01	d0102	1
```

**Run** (`query_id  doc_id  rank`), ordered by ascending rank; the TREC layout
`query_id Q0 doc_id rank score tag` is also accepted and is ordered by descending score:

```text
query_id	doc_id	rank
en-IN-01	d0104	1
en-IN-01	d0112	2
```

**Queries** (optional, `query_id  locale  query_text`):

```text
query_id	locale	query_text
hi-IN-03	hi-IN	पैन कार्ड ऑनलाइन आवेदन कैसे करें
mr-IN-04	mr-IN	उकडीचे मोदक रेसिपी
```

Without a queries file the locale is taken from a query-id prefix such as `mr-IN-04`, otherwise `unknown`.

## Metric definitions

With `rel(d) = grade(d) >= threshold`, R = number of judged-relevant documents for the query:

- **P@k** = relevant in top k / k (a ranking shorter than k is padded with non-relevant)
- **R@k** = relevant in top k / R
- **MRR** = mean of 1 / rank of the first relevant document (0 if none retrieved)
- **MAP** = mean over queries of (sum of precision at each relevant rank) / R
- **DCG@k** = sum over i <= k of gain(g_i) / log2(i + 1); **nDCG@k** = DCG@k / DCG@k of the ideal ordering of all judged documents

As in `trec_eval`, only queries that appear in the qrels are scored; queries with no
document at or above the threshold are skipped (and listed), and judged queries absent
from the run score 0. Unjudged retrieved documents count as grade 0.

**Significance**: for per-query differences `d_i = B_i - A_i`, the test flips the sign of
each `d_i` at random and counts how often `|mean|` is at least the observed `|mean|`.
When `2^n <= iterations` all sign assignments are enumerated (exact p); otherwise a
seeded Monte Carlo estimate `(c + 1) / (iterations + 1)` is used.

## Tests

```bash
python -m pytest -q
```

33 tests, including the worked DCG example from Wikipedia's *Discounted cumulative gain*
article (grades 3,2,3,0,1,2: DCG@6 = 6.861, IDCG@6 = 8.740, nDCG@6 = 0.785), hand-computed AP,
exponential-gain DCG, exact permutation p-values, file-format parsing and the CLI end to end.

## Project structure

```text
search-relevance-metrics/
├── src/relevance/
│   ├── io.py         # qrels / run / queries readers, locale inference
│   ├── metrics.py    # P@k, R@k, RR, AP, DCG, nDCG
│   ├── evaluate.py   # per-query, overall and per-locale evaluation
│   ├── compare.py    # wins/losses, paired randomization test
│   ├── report.py     # text rendering
│   └── cli.py        # `relevance eval`, `relevance compare`
├── tests/
├── examples/
│   ├── queries.tsv        # 15 queries: en-IN, hi-IN, mr-IN
│   ├── qrels.tsv          # graded judgments (0-4)
│   ├── run_baseline.tsv   # simulated baseline ranking
│   └── run_candidate.tsv  # simulated candidate ranking
├── pyproject.toml
└── LICENSE
```

## License

[MIT](LICENSE) © 2026 Ansari Adeel Ahmad
