from __future__ import annotations
import pandas as pd


def prepare_features(df: pd.DataFrame, model) -> tuple[pd.DataFrame, list[str]]:
    X = df.drop(columns=["Class"], errors="ignore")

    feature_step = model.named_steps["features"]
    expected_cols = list(feature_step.raw_feature_names_in_)

    missing = sorted(set(expected_cols) - set(X.columns))
    extra = sorted(set(X.columns) - set(expected_cols))

    if missing:
        raise ValueError(
            "Input schema does not match training schema.\n"
            f"Missing columns: {missing}\n"
            f"Extra columns: {extra}"
        )

    if extra:
        X = X.drop(columns=extra)

    X = X[expected_cols]
    return X, expected_cols