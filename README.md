# Fraud Risk Review Console

A cost-aware fraud detection system demonstrating the end-to-end lifecycle of a machine learning model, from training and evaluation to operational review workflows and API serving.

This project focuses on **operational realism** rather than leaderboard metrics.

It combines:

- cost-optimised model training
- schema-safe inference
- a fraud review dashboard
- monitoring of model behaviour
- a REST inference API
- automated tests

The goal is to simulate how a fraud detection system might function in a real operational environment.

---

## Problem Context

Fraud detection systems operate under **asymmetric costs**:

| Event | Impact |
|------|------|
| False positive | Manual review cost |
| False negative | Direct financial loss |

Traditional metrics like accuracy or F1 do not capture this trade-off.

Instead, this system selects decision thresholds by **minimising expected operational cost**:

```
Expected Cost =
(False Positives × review cost)
+ (False Negatives × missed fraud cost)
```

Cost assumptions are configurable in:

```
src/config.py
```

---

## Dataset

This project uses the public **Credit Card Fraud Detection dataset**:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Dataset characteristics:

| Property | Value |
|------|------|
| Transactions | 284,807 |
| Fraud cases | 492 |
| Fraud rate | 0.17% |

Features include:

- `Time`
- `Amount`
- `V1–V28` (PCA-transformed transaction features)

The PCA transformation means the original raw features are not available, limiting interpretability but preserving privacy.

---

## Model Performance

The current best model is a **cost-optimised logistic regression**.

Evaluation on the test set:

| Metric | Value |
|------|------|
| ROC-AUC | **0.97** |
| PR-AUC | **0.71** |
| Precision @ 1% | **15.3%** |
| Recall @ 5% | **91.8%** |

These results indicate that fraud is highly concentrated in the top-ranked transactions.

---

## Operational Cost Optimisation

Decision thresholds are selected by **minimising expected fraud cost**.

| Metric | Value |
|------|------|
| Optimal threshold | **0.98** |
| Expected cost (threshold policy) | **2805** |
| Expected cost (top-2% review policy) | **7050** |

Cost assumptions:

| Event | Cost |
|------|------|
| False positive review | $5 |
| Missed fraud | $200 |

---

## System Architecture

```
Training Pipeline
        |
        v
Feature Engineering
        |
        v
Model Training
        |
        v
Threshold Optimisation
        |
        v
Model Artifacts
        |
        +-----------------------+
        |                       |
        v                       v
Streamlit Review Console     FastAPI Inference API
        |                       |
        v                       v
Review Decisions           Real-time Predictions
        |
        v
Monitoring Logs
```

---

## Training Pipeline

Training is implemented in:

```
src/train.py
```

Steps include:

1. Load dataset  
2. Train/test split  
3. Feature engineering  
4. Model training  
5. Probability scoring  
6. Threshold sweep using expected cost  
7. Model selection  
8. Artifact persistence  

Artifacts are saved to:

```
artifacts/
```

---

## Feature Engineering

Implemented in:

```
src/features.py
```

Additional features:

- transaction hour
- log-transformed amount
- night transaction indicator

Feature engineering is embedded inside the **scikit-learn pipeline** to prevent training-serving skew.

---

## Streamlit Fraud Review Console

Run the dashboard:

```
streamlit run app/streamlit_app.py
```

The dashboard simulates a fraud operations workflow:

- probability distribution visualisation
- risk bucket segmentation
- threshold adjustment
- review queue prioritisation
- transaction inspection
- reviewer decision logging
- prediction drift monitoring

Reviewer actions are stored in:

```
data/decisions/review_log.csv
```

---

## Monitoring

Batch scoring metrics are logged to track system behaviour.

Logged metrics include:

- number of transactions scored
- number flagged for review
- review rate
- average fraud probability
- p95 probability

Logs are stored in:

```
data/monitoring/
```

---

## FastAPI Inference API

The trained model can be served through a REST API.

Run:

```
uvicorn api.main:app --reload
```

Available endpoints:

| Endpoint | Purpose |
|------|------|
| `/health` | Service health check |
| `/predict` | Fraud prediction |
| `/docs` | Interactive Swagger UI |

---

## Testing

Run tests with:

```
pytest
```

Test coverage includes:

- feature engineering
- risk utilities
- inference schema validation
- API health endpoint

---

## Project Structure

```
fraud-risk/
│
├── api/                # FastAPI inference service
├── app/                # Streamlit dashboard
├── src/                # training and inference code
├── tests/              # automated tests
│
├── artifacts/          # trained models and metrics
├── data/               # datasets and logs
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Setup

Install dependencies:

```
pip install -r requirements.txt
```

---

## Train the Model

```
python -m src.train
```

This produces:

```
artifacts/model.joblib
artifacts/metrics.json
```

---

## Run Dashboard

```
streamlit run app/streamlit_app.py
```

---

## Run API

```
uvicorn api.main:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

## Limitations

- PCA features limit interpretability.
- The dataset lacks original transaction features.
- The system focuses on **batch review workflows** rather than streaming fraud detection.
- Pickled scikit-learn models are version sensitive.

---

## Future Improvements

Potential extensions include:

- SHAP feature attribution
- automated drift detection
- streaming inference pipeline
- database-backed review decisions
- CI/CD pipelines for training and deployment

---

