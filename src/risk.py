from __future__ import annotations
import numpy as np
import math


def expected_cost(y_true, y_pred, cost_fp: float, cost_fn: float) -> float:
    """
    Compute total expected cost given predictions.
    """
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    return float(fp * cost_fp + fn * cost_fn)


def sweep_thresholds(y_true, probs, cost_fp: float, cost_fn: float):
    """
    Sweep thresholds from 0.01 to 0.99 and compute expected cost.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    results = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        cost = expected_cost(y_true, preds, cost_fp, cost_fn)
        results.append((float(t), float(cost)))

    return results


def precision_at_k(y_true, probs, k_frac: float = 0.01) -> float:
    """
    Precision among top k% highest-risk transactions.
    """
    k = int(len(probs) * k_frac)
    idx = np.argsort(probs)[::-1][:k]
    return float(y_true.iloc[idx].mean())


def recall_at_k(y_true, probs, k_frac: float = 0.01) -> float:
    """
    Recall captured within top k% highest-risk transactions.
    """
    k = int(len(probs) * k_frac)
    idx = np.argsort(probs)[::-1][:k]
    return float(y_true.iloc[idx].sum() / y_true.sum())


def flag_top_k(probs, k_frac: float = 0.02):
    """
    Flag top k% highest probability transactions for review.
    """
    k = int(len(probs) * k_frac)
    idx = np.argsort(probs)[::-1][:k]

    flags = np.zeros(len(probs))
    flags[idx] = 1
    return flags


def to_risk_score(prob: float) -> int:
    """
    Convert probability to risk score [0–100].
    """
    return int(math.floor(prob * 100))


def bucket(score: int) -> str:
    """
    Categorise risk score into operational bucket.
    """
    if score >= 80:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"