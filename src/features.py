from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Adds engineered features and removes redundant raw fields.
    Ensures schema validation for production safety.
    """

    REQUIRED_COLS = {"Time", "Amount"}

    def fit(self, X, y=None):
        if not self.REQUIRED_COLS.issubset(set(X.columns)):
            raise ValueError("Missing required columns for feature engineering.")
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X):
        X = X.copy()

        # Feature engineering
        X["Hour"] = ((X["Time"] // 3600) % 24).astype(int)
        X["AmountLog"] = np.log1p(X["Amount"])
        X["IsNight"] = X["Hour"].between(0, 5).astype(int)

        # Drop redundant raw fields
        X = X.drop(columns=["Time", "Amount"])

        return X