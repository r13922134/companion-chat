from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import precision_score, recall_score


RESULT_RE = re.compile(
    r"(?:Evaluation\s*)?result\s*[:=]\s*([0-3])(?!\d)(?!\s*[-\u2013]\s*[0-3])|"
    r"overall\s+score\s*(?:is|=|:)\s*([0-3])(?!\d)(?!\s*[-\u2013]\s*[0-3])|"
    r"score\s*(?:is|=|:)\s*([0-3])(?!\d)(?!\s*[-\u2013]\s*[0-3])",
    re.IGNORECASE,
)
LEADING_SCORE_WORD_RE = re.compile(
    r"^\s*([0-3])\s+(?:score|result)\b",
    re.IGNORECASE,
)
REPEATED_SINGLE_SCORE_RE = re.compile(
    r"^\s*([0-3])(?:\s+\1)+\s*[.\u3002]?\s*$",
    re.IGNORECASE,
)
LEADING_RESULT_RE = re.compile(
    r"^\s*(?:score\s*[:=]\s*)?([0-3])(?:\s*[\.).,:;-]|\s*$)",
    re.IGNORECASE,
)


def parse_evaluation_result(text: str) -> int | None:
    text = text or ""
    match = (
        RESULT_RE.search(text)
        or LEADING_SCORE_WORD_RE.search(text)
        or REPEATED_SINGLE_SCORE_RE.search(text)
        or LEADING_RESULT_RE.search(text)
    )
    if not match:
        return None
    return int(next(group for group in match.groups() if group is not None))


def concordance_correlation_coefficient(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return 0.0
    mean_true = y_true.mean()
    mean_pred = y_pred.mean()
    var_true = y_true.var()
    var_pred = y_pred.var()
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    denom = var_true + var_pred + (mean_true - mean_pred) ** 2
    if denom == 0:
        return 1.0
    return float((2.0 * covariance) / denom)


@dataclass(frozen=True)
class MetricResult:
    ccc: float
    precision: float
    recall: float
    f1: float
    rmse: float
    mae: float

    def as_dict(self) -> dict[str, float]:
        return {
            "CCC": self.ccc,
            "Precision": self.precision,
            "Recall": self.recall,
            "F1": self.f1,
            "RMSE": self.rmse,
            "MAE": self.mae,
        }


def compute_metrics(
    y_true_total: np.ndarray,
    y_pred_total: np.ndarray,
    depression_threshold: float = 10.0,
) -> MetricResult:
    y_true_total = np.asarray(y_true_total, dtype=np.float64)
    y_pred_total = np.asarray(y_pred_total, dtype=np.float64)
    y_true_bin = (y_true_total >= depression_threshold).astype(int)
    y_pred_bin = (y_pred_total >= depression_threshold).astype(int)
    return MetricResult(
        ccc=concordance_correlation_coefficient(y_true_total, y_pred_total),
        precision=float(
            precision_score(y_true_bin, y_pred_bin, zero_division=0)
        ),
        recall=float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
        f1=float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
        rmse=float(mean_squared_error(y_true_total, y_pred_total) ** 0.5),
        mae=float(mean_absolute_error(y_true_total, y_pred_total)),
    )
