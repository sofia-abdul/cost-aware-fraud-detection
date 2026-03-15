from __future__ import annotations

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import LATEST_MODEL_PATH
from src.inference import prepare_features
from src.risk import bucket, to_risk_score

app = FastAPI(
    title="Fraud Detection API",
    description="Fraud scoring service using a trained scikit-learn model",
    version="1.0.0",
)


class TransactionRecord(BaseModel):
    Time: float = Field(..., description="Transaction time")
    Amount: float = Field(..., description="Transaction amount")

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


def load_model():
    if not LATEST_MODEL_PATH.exists():
        raise RuntimeError("Model artifact not found. Run training first.")
    return joblib.load(LATEST_MODEL_PATH)


model = load_model()


@app.get("/")
def root():
    return {
        "message": "Fraud Detection API is running",
        "health": "/health",
        "docs": "/docs",
        "predict": "/predict",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: list[TransactionRecord]):
    try:
        df = pd.DataFrame([record.model_dump() for record in payload])
        X, _ = prepare_features(df, model)
        probs = model.predict_proba(X)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    predictions = []

    for p in probs:
        score = to_risk_score(p)
        predictions.append(
            {
                "fraud_probability": float(p),
                "risk_score": score,
                "risk_bucket": bucket(score),
            }
        )

    return {"predictions": predictions}