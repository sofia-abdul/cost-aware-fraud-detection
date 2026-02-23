import pandas as pd
from src.features import FeatureEngineer


def test_feature_engineering_adds_expected_columns():
    df = pd.DataFrame({
        "Time": [0, 3600],
        "Amount": [100, 200],
        "V1": [0.1, 0.2],
    })

    fe = FeatureEngineer()
    fe.fit(df)

    transformed = fe.transform(df)

    # New engineered features should exist
    assert "Hour" in transformed.columns
    assert "AmountLog" in transformed.columns
    assert "IsNight" in transformed.columns

    # Raw fields should be removed after transformation
    assert "Time" not in transformed.columns
    assert "Amount" not in transformed.columns