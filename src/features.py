from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    REQUIRED_COLS = {"Time", "Amount"}

    def fit(self, X, y=None):
        if not hasattr(X, "columns"):
            raise TypeError("Input must be a pandas DataFrame with named columns.")

        if not self.REQUIRED_COLS.issubset(set(X.columns)):
            missing = sorted(self.REQUIRED_COLS - set(X.columns))
            raise ValueError(f"Missing required columns for feature engineering: {missing}")

        self.raw_feature_names_in_ = list(X.columns)

        transformed = X.copy()
        transformed["Hour"] = ((transformed["Time"] // 3600) % 24).astype(int)
        transformed["AmountLog"] = np.log1p(transformed["Amount"])
        transformed["IsNight"] = transformed["Hour"].between(0, 5).astype(int)
        transformed = transformed.drop(columns=["Time", "Amount"])

        self.feature_names_out_ = list(transformed.columns)
        return self

    def transform(self, X):
        if not hasattr(X, "columns"):
            raise TypeError("Input must be a pandas DataFrame with named columns.")

        missing = sorted(set(self.raw_feature_names_in_) - set(X.columns))
        extra = sorted(set(X.columns) - set(self.raw_feature_names_in_))

        if missing or extra:
            raise ValueError(
                "Input schema does not match training schema.\n"
                f"Missing columns: {missing}\n"
                f"Extra columns: {extra}"
            )

        X = X[self.raw_feature_names_in_].copy()
        X["Hour"] = ((X["Time"] // 3600) % 24).astype(int)
        X["AmountLog"] = np.log1p(X["Amount"])
        X["IsNight"] = X["Hour"].between(0, 5).astype(int)
        X = X.drop(columns=["Time", "Amount"])

        return X