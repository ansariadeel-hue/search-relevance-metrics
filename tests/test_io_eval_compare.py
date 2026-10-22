import json
from pathlib import Path

import pytest

from relevance import (
    FormatError,
    Settings,
    compare,
    evaluate,
    infer_locale,
    load_qrels,
    load_queries,
    load_run,
    paired_randomization_test,
    win_loss_tie,
)
from relevance.cli import main

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


# --- file formats -------------------------------------------------------------
def test_load_qrels_simple_with_header(tmp_path):
    p = write(tmp_path, "q.tsv", "query_id\tdoc_id\tgrade\nq1\td1\t3\nq1\td2\t0\n# comment\nq2\td9\t4\n")
    assert load_qrels(p) == {"q1": {"d1": 3, "d2": 0}, "q2": {"d9": 4}}


def test_load_qrels_trec_format(tmp_path):
    p = write(tmp_path, "q.txt", "q1 0 d1 2\nq1 0 d2 1\n")
    assert load_qrels(p) == {"q1": {"d1": 2, "d2": 1}}


def test_load_qrels_rejects_out_of_scale_grade(tmp_path):
    p = write(tmp_path, "q.tsv", "q1\td1\t5\n")
    with pytest.raises(FormatError, match="outside 0-4"):
        load_qrels(p)


def test_load_run_simple_orders_by_rank(tmp_path):
    p = write(tmp_path, "r.tsv", "query_id\tdoc_id\trank\nq1\tb\t2\nq1\ta\t1\nq1\tc\t3\nq1\ta\t4\n")
    assert load_run(p) == {"q1": ["a", "b", "c"]}  # duplicate 'a' keeps best rank


def test_load_run_trec_orders_by_score(tmp_path):
    p = write(tmp_path, "r.txt", "q1 Q0 a 1 0.2 sys\nq1 Q0 b 2 0.9 sys\nq1 Q0 c 3 0.5 sys\n")
    assert load_run(p) == {"q1": ["b", "c", "a"]}


def test_load_queries_and_locale_inference():
    queries = load_queries(EXAMPLES / "queries.tsv")
    assert queries["hi-IN-03"]["locale"] == "hi-IN"
    assert "पैन कार्ड" in queries["hi-IN-03"]["text"]
    assert infer_locale("mr-IN-02") == "mr-IN"
    assert infer_locale("q17") == "unknown"
    assert infer_locale("q17", {"q17": {"locale": "ta-IN", "text": ""}}) == "ta-IN"


# --- evaluation -------------------------------------------------------------
def test_evaluate_skips_unanswerable_and_zeroes_missing():
    qrels = {"q1": {"a": 3, "b": 0}, "q2": {"x": 1}, "q3": {"z": 4}}
    run = {"q1": ["a", "b"], "q9": ["nope"]}
    ev = evaluate(qrels, run, Settings(ks=(1,), threshold=2))
    assert ev.skipped_no_relevant == ["q2"]
    assert ev.missing_from_run == ["q3"]
    assert ev.per_query["q1"]["P@1"] == 1.0
    assert all(v == 0 for v in ev.per_query["q3"].values())
    assert ev.mean()["MRR"] == pytest.approx(0.5)


def test_evaluate_locale_breakdown():
    qrels = {"en-IN-1": {"a": 4}, "hi-IN-1": {"a": 4}, "hi-IN-2": {"a": 4}}
    run = {"en-IN-1": ["a"], "hi-IN-1": ["a"], "hi-IN-2": ["b", "a"]}
    groups = evaluate(qrels, run).by_locale()
    assert groups["en-IN"]["n"] == 1 and groups["hi-IN"]["n"] == 2
    assert groups["hi-IN"]["mean"]["MRR"] == pytest.approx(0.75)


def test_evaluate_example_data_has_three_locales():
    ev = evaluate(load_qrels(EXAMPLES / "qrels.tsv"), load_run(EXAMPLES / "run_baseline.tsv"))
    assert set(ev.by_locale()) == {"en-IN", "hi-IN", "mr-IN"}
    for scores in ev.per_query.values():
        assert all(0.0 <= v <= 1.0 for v in scores.values())


# --- significance testing ---------------------------------------------------
def test_permutation_identical_runs():
    assert paired_randomization_test([0.1, 0.5, 0.9], [0.1, 0.5, 0.9]) == 1.0


def test_permutation_exact_enumeration():
    # 5 queries, all improved by the same amount: only the all-+ and all-- sign
    # assignments are as extreme -> p = 2 / 2^5
    assert paired_randomization_test([0.0] * 5, [0.1] * 5) == pytest.approx(2 / 32)


def test_permutation_monte_carlo_is_seeded_and_detects_real_gain():
    a = [0.40 + 0.01 * i for i in range(30)]
    b = [x + 0.05 + (0.02 if i % 3 else -0.03) for i, x in enumerate(a)]
    p1 = paired_randomization_test(a, b, iterations=2000, seed=7)
    p2 = paired_randomization_test(a, b, iterations=2000, seed=7)
    assert p1 == p2
    assert p1 < 0.01


def test_permutation_noise_not_significant():
    a = [0.5, 0.6, 0.4, 0.7, 0.5, 0.6, 0.4, 0.7, 0.5, 0.6, 0.4, 0.7, 0.5, 0.6, 0.4]
    b = [0.6, 0.5, 0.5, 0.6, 0.4, 0.7, 0.5, 0.6, 0.5, 0.6, 0.3, 0.8, 0.6, 0.5, 0.4]
    assert paired_randomization_test(a, b, iterations=5000) > 0.5


def test_permutation_validates_input():
    with pytest.raises(ValueError):
        paired_randomization_test([1.0], [1.0, 2.0])


def test_win_loss_tie():
    assert win_loss_tie([0.1, 0.5, 0.3], [0.2, 0.4, 0.3]) == (1, 1, 1)


def test_compare_structure_on_examples():
    qrels = load_qrels(EXAMPLES / "qrels.tsv")
    ev_a = evaluate(qrels, load_run(EXAMPLES / "run_baseline.tsv"))
    ev_b = evaluate(qrels, load_run(EXAMPLES / "run_candidate.tsv"))
    result = compare(ev_a, ev_b, primary="nDCG@10", iterations=2000)
    assert set(result["by_locale"]) == {"en-IN", "hi-IN", "mr-IN"}
    row = next(m for m in result["overall"] if m["metric"] == "nDCG@10")
    assert row["wins"] + row["losses"] + row["ties"] == result["n_queries"]
    deltas = [r["delta"] for r in result["per_query"]]
    assert deltas == sorted(deltas, reverse=True)
    with pytest.raises(ValueError):
        compare(ev_a, ev_b, primary="nDCG@3")


# --- CLI --------------------------------------------------------------------
def test_cli_eval_text(capsys):
    assert main(["eval", str(EXAMPLES / "qrels.tsv"), str(EXAMPLES / "run_baseline.tsv")]) == 0
    out = capsys.readouterr().out
    assert "By locale" in out and "mr-IN" in out and "nDCG@10" in out


def test_cli_eval_json_linear_gain(capsys):
    args = ["eval", str(EXAMPLES / "qrels.tsv"), str(EXAMPLES / "run_baseline.tsv"), "--gain", "linear", "-k", "3", "--json"]
    assert main(args) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["settings"]["gain"] == "linear"
    assert "nDCG@3" in data["mean"]


def test_cli_compare_text_with_queries(capsys):
    args = [
        "compare", str(EXAMPLES / "qrels.tsv"), str(EXAMPLES / "run_baseline.tsv"),
        str(EXAMPLES / "run_candidate.tsv"), "--queries", str(EXAMPLES / "queries.tsv"),
    ]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "Run comparison" in out and "W/L/T" in out
    assert "उकडीचे मोदक" in out


def test_cli_compare_is_reproducible(capsys):
    args = ["compare", str(EXAMPLES / "qrels.tsv"), str(EXAMPLES / "run_baseline.tsv"),
            str(EXAMPLES / "run_candidate.tsv"), "--json", "--seed", "1"]
    main(args)
    first = capsys.readouterr().out
    main(args)
    assert capsys.readouterr().out == first


def test_cli_error(tmp_path, capsys):
    assert main(["eval", str(tmp_path / "missing.tsv"), str(tmp_path / "run.tsv")]) == 2
    assert "relevance: error" in capsys.readouterr().err
