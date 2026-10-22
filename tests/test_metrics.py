import math

import pytest

from relevance import (
    average_precision,
    dcg_at_k,
    dcg_from_grades,
    gain,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RANKING = ["d1", "d2", "d3", "d4", "d5"]
JUDGED = {"d1": 3, "d2": 0, "d3": 2, "d4": 1, "d6": 4}  # d5 unjudged, d6 not retrieved


def test_precision_at_k():
    # relevant (grade >= 2): d1, d3, d6
    assert precision_at_k(RANKING, JUDGED, 1, threshold=2) == 1.0
    assert precision_at_k(RANKING, JUDGED, 3, threshold=2) == pytest.approx(2 / 3)
    # ranking shorter than k: missing positions count as non-relevant
    assert precision_at_k(RANKING, JUDGED, 10, threshold=2) == pytest.approx(0.2)


def test_threshold_changes_binary_relevance():
    assert precision_at_k(RANKING, JUDGED, 5, threshold=1) == pytest.approx(3 / 5)
    assert precision_at_k(RANKING, JUDGED, 5, threshold=3) == pytest.approx(1 / 5)


def test_recall_at_k():
    assert recall_at_k(RANKING, JUDGED, 3, threshold=2) == pytest.approx(2 / 3)
    assert recall_at_k(RANKING, JUDGED, 1, threshold=2) == pytest.approx(1 / 3)
    assert recall_at_k(RANKING, {"x": 0}, 5, threshold=2) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(RANKING, JUDGED, threshold=2) == 1.0
    assert reciprocal_rank(RANKING, JUDGED, threshold=4) == 0.0  # d6 never retrieved
    assert reciprocal_rank(["a", "b", "c"], {"c": 2}, threshold=2) == pytest.approx(1 / 3)


def test_average_precision_hand_computed():
    # relevant at ranks 1 and 3, third relevant doc (d6) never retrieved:
    # AP = (1/1 + 2/3) / 3
    assert average_precision(RANKING, JUDGED, threshold=2) == pytest.approx((1 + 2 / 3) / 3)


def test_average_precision_perfect_and_empty():
    assert average_precision(["a", "b"], {"a": 4, "b": 3}, threshold=2) == 1.0
    assert average_precision(["a"], {"a": 1}, threshold=2) == 0.0


def test_gain_functions():
    assert gain(0) == 0 and gain(4) == 15
    assert gain(3, "linear") == 3
    with pytest.raises(ValueError):
        gain(1, "cubic")


def test_dcg_wikipedia_example_linear_gain():
    # Wikipedia "Discounted cumulative gain": grades 3,2,3,0,1,2 -> DCG6 = 6.861
    grades = [3, 2, 3, 0, 1, 2]
    assert dcg_from_grades(grades, 6, "linear") == pytest.approx(6.861, abs=5e-4)
    # ideal ordering of all judged docs 3,3,3,2,2,2,1,0 -> IDCG6 = 8.740, nDCG6 = 0.785
    assert dcg_from_grades([3, 3, 3, 2, 2, 2, 1, 0], 6, "linear") == pytest.approx(8.740, abs=5e-4)
    ranking = [f"D{i}" for i in range(1, 7)]
    judged = {f"D{i}": g for i, g in enumerate(grades, start=1)}
    judged.update({"D7": 3, "D8": 2})  # judged but not retrieved
    assert ndcg_at_k(ranking, judged, 6, "linear") == pytest.approx(0.785, abs=5e-4)


def test_dcg_exponential_gain_hand_computed():
    assert dcg_from_grades([3, 2], kind="exponential") == pytest.approx(7 + 3 / math.log2(3))
    assert dcg_at_k(["a", "b", "x"], {"a": 3, "b": 2}, 3) == pytest.approx(7 + 3 / math.log2(3))


def test_ndcg_bounds():
    judged = {"a": 4, "b": 2, "c": 0}
    assert ndcg_at_k(["a", "b", "c"], judged, 3) == pytest.approx(1.0)
    assert ndcg_at_k(["c", "x", "y"], judged, 3) == 0.0
    assert ndcg_at_k(["a"], {"a": 0}, 3) == 0.0  # no gain available at all
    worse = ndcg_at_k(["b", "a", "c"], judged, 3)
    assert 0 < worse < 1


def test_exponential_gain_rewards_top_grade_more_than_linear():
    judged = {"perfect": 4, "fair": 1, "good": 2}
    ranking = ["fair", "good", "perfect"]  # perfect doc buried at rank 3
    assert ndcg_at_k(ranking, judged, 3, "exponential") < ndcg_at_k(ranking, judged, 3, "linear")


def test_invalid_k():
    with pytest.raises(ValueError):
        precision_at_k(RANKING, JUDGED, 0)
    with pytest.raises(ValueError):
        ndcg_at_k(RANKING, JUDGED, 0)
