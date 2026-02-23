from __future__ import annotations
import pandas as pd

def prepare_features(df: pd.DataFrame, model) -> tuple[pd.DataFrame, list[str]]:
    """
    Drops target column if present, validates schema vs training, and aligns column order.

    Returns:
        X: dataframe ready for model.predict_proba
        expected_cols: list of expected feature columns
    """
    X = df.drop(columns=["Class"], errors="ignore")

    expected_cols = list(model.named_steps["features"].feature_names_in_)

    if set(X.columns) != set(expected_cols):
        missing = sorted(set(expected_cols) - set(X.columns))
        extra = sorted(set(X.columns) - set(expected_cols))
        msg = (
            "Input schema does not match training schema.\n"
            f"Missing columns: {missing}\n"
            f"Extra columns: {extra}"
        )
        raise ValueError(msg)

    X = X[expected_cols]
    return X, expected_cols
