from __future__ import annotations

import json
import platform
import datetime as dt
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

from src.features import FeatureEngineer
from src.config import (
    RAW_PATH,
    RUNS_DIR,
    LATEST_MODEL_PATH,
    LATEST_METRICS_PATH,
    COST_FALSE_POSITIVE_REVIEW,
    COST_MISSED_FRAUD,
    RANDOM_STATE,
)
from src.risk import (
    sweep_thresholds,
    precision_at_k,
    recall_at_k,
    flag_top_k,
    expected_cost,
)
from src.evaluation import compute_core_metrics


def main():
    run_id = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")

    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)

    if "Class" not in df.columns:
        raise ValueError("Dataset must contain a 'Class' target column.")

    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    models = {
        "logistic": Pipeline([
            ("features", FeatureEngineer()),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        "rf_calibrated": Pipeline([
            ("features", FeatureEngineer()),
            ("model", CalibratedClassifierCV(
                estimator=RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
                method="sigmoid",
                cv=3,
            )),
        ]),
    }

    best = None

    for name, pipe in models.items():
        print(f"Training {name}...")
        pipe.fit(X_train, y_train)

        probs = pipe.predict_proba(X_test)[:, 1]

        core_metrics = compute_core_metrics(y_test, probs)

        sweep = sweep_thresholds(
            y_test.values,
            probs,
            COST_FALSE_POSITIVE_REVIEW,
            COST_MISSED_FRAUD,
        )
        threshold, min_cost = min(sweep, key=lambda x: x[1])

        topk_flags = flag_top_k(probs, k_frac=0.02)
        topk_cost = expected_cost(
            y_test.values,
            topk_flags,
            COST_FALSE_POSITIVE_REVIEW,
            COST_MISSED_FRAUD,
        )

        candidate = {
            "name": name,
            "pipeline": pipe,
            "threshold": threshold,
            "expected_cost": min_cost,
            "topk_cost_2pct": topk_cost,
            "precision_at_1pct": precision_at_k(y_test, probs, 0.01),
            "recall_at_5pct": recall_at_k(y_test, probs, 0.05),
            **core_metrics,
            "baseline_stats": {
                "mean_probability": float(probs.mean()),
                "fraud_rate_test": float(y_test.mean()),
            },
        }

        if best is None or candidate["expected_cost"] < best["expected_cost"]:
            best = candidate

    if best is None:
        raise RuntimeError("No model candidate was trained successfully.")

    run_model_path = RUNS_DIR / f"{run_id}_model.joblib"
    run_metrics_path = RUNS_DIR / f"{run_id}_metrics.json"

    joblib.dump(best["pipeline"], run_model_path)

    metrics = {
        "run_id": run_id,
        "model": best["name"],
        "threshold": best["threshold"],
        "performance": {
            "roc_auc": best["roc_auc"],
            "pr_auc": best["pr_auc"],
            "precision_at_1pct": best["precision_at_1pct"],
            "recall_at_5pct": best["recall_at_5pct"],
        },
        "cost_analysis": {
            "threshold_cost": best["expected_cost"],
            "topk_cost_2pct": best["topk_cost_2pct"],
        },
        "baseline_stats": best["baseline_stats"],
        "costs": {
            "false_positive_review": COST_FALSE_POSITIVE_REVIEW,
            "missed_fraud": COST_MISSED_FRAUD,
        },
        "env": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }

    with open(run_metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(best["pipeline"], LATEST_MODEL_PATH)
    with open(LATEST_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete.")


if __name__ == "__main__":
    main()