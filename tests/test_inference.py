import pandas as pd
from unittest.mock import MagicMock

from src.inference import prepare_features


def test_prepare_features_aligns_columns():
    df = pd.DataFrame({
        "Time": [0],
        "Amount": [100],
        "V1": [0.1],
        "V2": [0.2],
    })

    mock_model = MagicMock()
    mock_model.named_steps = {
        "features": MagicMock(raw_feature_names_in_=["Time", "Amount", "V1", "V2"])
    }

    X, expected = prepare_features(df, mock_model)

    assert list(X.columns) == expected


def test_prepare_features_detects_missing_columns():
    df = pd.DataFrame({
        "Time": [0],
        "Amount": [100],
    })

    mock_model = MagicMock()
    mock_model.named_steps = {
        "features": MagicMock(raw_feature_names_in_=["Time", "Amount", "V1"])
    }

    try:
        prepare_features(df, mock_model)
        assert False
    except ValueError:
        assert True