from __future__ import annotations
from sklearn.metrics import roc_auc_score, average_precision_score


def compute_core_metrics(y_true, probs):
    return {
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
    }