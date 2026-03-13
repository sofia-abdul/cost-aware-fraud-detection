from __future__ import annotations
import math
import numpy as np


def expected_cost(y_true, y_pred, cost_fp: float, cost_fn: float) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    return float(fp * cost_fp + fn * cost_fn)


def _resolve_k(probs, k_frac: float) -> int:
    if not (0 < k_frac <= 1):
        raise ValueError(f"k_frac must be in (0, 1], got {k_frac}")
    return max(1, int(len(probs) * k_frac))


def sweep_thresholds(y_true, probs, cost_fp: float, cost_fn: float):
    thresholds = np.linspace(0.01, 0.99, 99)
    results = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        cost = expected_cost(y_true, preds, cost_fp, cost_fn)
        results.append((float(t), float(cost)))

    return results


def precision_at_k(y_true, probs, k_frac: float = 0.01) -> float:
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)

    k = _resolve_k(probs, k_frac)
    idx = np.argsort(probs)[::-1][:k]
    return float(y_true[idx].mean())


def recall_at_k(y_true, probs, k_frac: float = 0.01) -> float:
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)

    positives = y_true.sum()
    if positives == 0:
        return 0.0

    k = _resolve_k(probs, k_frac)
    idx = np.argsort(probs)[::-1][:k]
    return float(y_true[idx].sum() / positives)


def flag_top_k(probs, k_frac: float = 0.02):
    probs = np.asarray(probs)
    k = _resolve_k(probs, k_frac)
    idx = np.argsort(probs)[::-1][:k]

    flags = np.zeros(len(probs), dtype=int)
    flags[idx] = 1
    return flags


def to_risk_score(prob: float) -> int:
    return int(math.floor(prob * 100))


def bucket(score: int) -> str:
    if score >= 80:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"