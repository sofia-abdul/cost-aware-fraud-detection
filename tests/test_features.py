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

    assert "Hour" in transformed.columns
    assert "AmountLog" in transformed.columns
    assert "IsNight" in transformed.columns


def test_feature_engineering_removes_raw_columns():
    df = pd.DataFrame({
        "Time": [0],
        "Amount": [100],
        "V1": [0.1],
    })

    fe = FeatureEngineer()
    fe.fit(df)

    transformed = fe.transform(df)

    assert "Time" not in transformed.columns
    assert "Amount" not in transformed.columns