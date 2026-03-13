from __future__ import annotations

import joblib
import pandas as pd
from fastapi import FastAPI

from src.config import LATEST_MODEL_PATH
from src.inference import prepare_features
from src.risk import bucket, to_risk_score

app = FastAPI(title="Fraud Detection API")


def load_model():
    return joblib.load(LATEST_MODEL_PATH)


model = load_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: list[dict]):
    df = pd.DataFrame(payload)

    X, _ = prepare_features(df, model)

    probs = model.predict_proba(X)[:, 1]

    results = []

    for p in probs:
        score = to_risk_score(p)

        results.append(
            {
                "fraud_probability": float(p),
                "risk_score": score,
                "risk_bucket": bucket(score),
            }
        )

    return {"predictions": results}