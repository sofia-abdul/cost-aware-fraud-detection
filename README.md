# Cost-Aware Fraud Ranking System

A fraud detection system framed as a cost-sensitive ranking problem under operational review constraints.

This repository focuses on practical ML engineering considerations: asymmetric business cost, limited review capacity, calibrated probabilities, and prediction drift monitoring.

---

## Problem Framing

Fraud detection in production is not a pure classification task.

Key constraints:

- Severe class imbalance (~0.17% fraud rate)
- Asymmetric cost of false positives vs missed fraud
- Limited manual review capacity
- Need for probability calibration
- Risk of distribution shift over time

This system treats fraud detection as a ranking problem where decisions are made based on cost and operational limits.

---

## Dataset

This project uses the publicly available **Credit Card Fraud Detection** dataset:

Andrea Dal Pozzolo et al., Université Libre de Bruxelles (ULB)  
Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  

Dataset characteristics:

- 284,807 transactions  
- 492 fraud cases (~0.172%)  
- PCA-transformed features (V1–V28)  
- Highly imbalanced  

Place the dataset at:

data/raw/creditcard.csv

The dataset is not committed to this repository.

---

## Approach

### Models

- Logistic Regression (class-weighted baseline)
- Calibrated Random Forest

### Evaluation

- PR-AUC and ROC-AUC
- Cost-sensitive threshold optimisation
- Top-k review simulation
- Precision@K and Recall@K
- Baseline prediction logging for drift comparison

### Decision Policies

Two decision strategies are supported:

1. **Threshold-based**

   Minimises expected cost:

   cost = FP_cost × FP + FN_cost × FN

2. **Top-k review**

   Flags the highest-risk k% of transactions to simulate fixed review capacity.

---

## Drift Monitoring

During training, baseline prediction statistics are logged.

At inference time, the Streamlit console compares:

- Baseline mean predicted probability
- Current batch mean predicted probability
- Delta between baseline and current batch

Significant deviations are surfaced as drift warnings.

---

## Repository Structure

```
.
├── app/                # Streamlit review console
├── src/
│   ├── features.py     # Feature engineering
│   ├── risk.py         # Cost and ranking logic
│   ├── evaluation.py   # Metrics and analysis utilities
│   ├── inference.py    # Model loading and schema validation
│   └── config.py       # Central configuration
├── tests/              # Unit tests
├── train.py            # Training entrypoint
├── notebooks/          # Exploratory analysis
└── reports/            # Figures / screenshots
```

The project separates feature engineering, business logic, evaluation logic, and UI concerns.

---

## Training

Run:

```
python train.py
```

Training performs:

- Stratified train/test split
- Model comparison
- Cost-sensitive threshold sweep
- Ranking metric evaluation
- Baseline statistic logging

Artifacts are written to:

```
artifacts/runs/
```

---

## Review Console

Launch:

```
python -m streamlit run app/streamlit_app.py
```

The console supports:

- Threshold vs Top-k mode
- Batch CSV upload
- Cost estimation (if labels present)
- Risk bucket distribution
- Transaction inspection
- Threshold sensitivity analysis
- Prediction drift panel

---

## Testing

Core logic is unit tested:

- Cost computation
- Ranking policy behaviour
- Feature engineering contracts

Run:

```
pytest
```

---

## Scope

This repository prioritises production-oriented ML design over model novelty.

Focus areas:

- Cost-aware decision logic
- Ranking under capacity constraints
- Calibration
- Basic lifecycle monitoring
- Modular code organisation