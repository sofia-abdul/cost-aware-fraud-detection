import numpy as np
import pandas as pd

from src.risk import (
    expected_cost,
    precision_at_k,
    recall_at_k,
    flag_top_k,
)


def test_expected_cost_simple_case():
    # One false positive and one missed fraud
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0])

    cost = expected_cost(y_true, y_pred, cost_fp=5, cost_fn=100)

    # 1 FP (5) + 1 FN (100)
    assert cost == 105


def test_flag_top_k_marks_highest_probs():
    probs = np.array([0.9, 0.8, 0.1, 0.05])

    flags = flag_top_k(probs, k_frac=0.5)

    # Should flag the two highest probabilities
    assert len(flags) == len(probs)
    assert flags.sum() == 2
    assert flags[0] == 1
    assert flags[1] == 1


def test_precision_at_k_basic():
    y_true = pd.Series([1, 0, 1, 0])
    probs = np.array([0.9, 0.8, 0.2, 0.1])

    # Top 50% → first two values → [1, 0]
    precision = precision_at_k(y_true, probs, k_frac=0.5)

    assert precision == 0.5


def test_recall_at_k_basic():
    y_true = pd.Series([1, 0, 1, 0])
    probs = np.array([0.9, 0.8, 0.2, 0.1])

    # We capture 1 of the 2 frauds
    recall = recall_at_k(y_true, probs, k_frac=0.5)

    assert recall == 0.5