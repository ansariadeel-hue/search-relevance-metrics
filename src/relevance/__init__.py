"""relevance - offline search-ranking evaluation from graded human judgments."""

from .compare import compare, paired_randomization_test, win_loss_tie
from .evaluate import Evaluation, Settings, evaluate
from .io import FormatError, infer_locale, load_qrels, load_queries, load_run
from .metrics import (
    average_precision,
    dcg_at_k,
    dcg_from_grades,
    gain,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

__version__ = "0.1.0"

__all__ = [
    "compare",
    "paired_randomization_test",
    "win_loss_tie",
    "Evaluation",
    "Settings",
    "evaluate",
    "FormatError",
    "infer_locale",
    "load_qrels",
    "load_queries",
    "load_run",
    "average_precision",
    "dcg_at_k",
    "dcg_from_grades",
    "gain",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "__version__",
]
