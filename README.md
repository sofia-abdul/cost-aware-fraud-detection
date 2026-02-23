
# Fraud Risk Review Console

A cost-optimized fraud scoring pipeline and batch review dashboard built with scikit-learn and Streamlit.

This project focuses on operational realism rather than leaderboard metrics. It demonstrates:

* Business-aligned threshold selection (expected-cost minimization)
* Schema enforcement at inference time
* Versioned model artifacts with reproducibility metadata
* Batch review workflow with decision capture
* Automated tests
* Optional containerized deployment

---

## Problem Framing

Fraud detection systems operate under asymmetric costs:

* False positives create operational expense through manual review.
* False negatives represent direct financial loss.

Instead of optimizing accuracy or F1, this system selects a decision threshold that minimizes:

```
Expected Cost =
  (False Positives × review cost)
+ (False Negatives × missed fraud cost)
```

Cost assumptions are configurable in `src/config.py`.

---

## Data Contract

The model expects the Credit Card dataset schema:

### Required Columns

* `Time`
* `V1`–`V28`
* `Amount`

### Optional

* `Class` (used only for evaluation and cost estimation)

`V1–V28` are PCA-transformed components included in the dataset. They cannot be derived from raw transaction logs without the original preprocessing pipeline. This is a deliberate constraint of the dataset; schema validation is enforced at inference time to prevent silent scoring errors.

---

## Architecture

### Training Pipeline

```
raw dataset
  → FeatureEngineer (sklearn transformer)
  → model (Logistic Regression / Random Forest / calibrated variants)
  → threshold sweep (cost-based)
  → versioned artifacts
```

### Inference Pipeline

```
input batch
  → schema validation
  → feature alignment
  → probability scoring
  → threshold decision
  → review dashboard
```

---

## Design Rationale

This project is structured around practical failure modes seen in ML systems.

### Why cost-based thresholding (not F1)

Fraud datasets are extremely imbalanced. A threshold optimized for generic metrics (accuracy/F1) often produces an operationally unusable review load or misses too much fraud. Selecting the threshold by expected cost explicitly encodes the business tradeoff between:

* Review workload (false positives)
* Missed fraud loss (false negatives)

This makes the decision boundary interpretable in terms of operational impact.

### Why the pipeline embeds feature engineering

Feature engineering is included inside the scikit-learn pipeline rather than performed as a separate preprocessing step. This reduces training/serving skew and makes the artifact self-contained: the same transformation is applied during training and inference.

### Why strict schema validation exists

Schema mismatch is a common source of silent model failures (extra columns, missing columns, wrong ordering). The inference layer:

* Drops the label column (`Class`) when present
* Validates the feature set against training-time expectations
* Aligns column ordering before scoring

The goal is to fail fast with a clear error rather than produce incorrect outputs.

### Why probability calibration is relevant

For threshold-based decisions, calibrated probabilities are valuable: they make “0.2 means ~20% risk” closer to true. Calibration is included as an optional model candidate to improve decision quality and make threshold tuning more meaningful.

---

## Dashboard Capabilities

The dashboard simulates a fraud operations workflow:

* Risk distribution visualization for the scored batch
* Adjustable decision threshold with live KPI updates
* Review queue filtering (bucket, probability range, flagged-only)
* Human decision capture (Approve / Reject / Escalate)
* Export of scored batches and review decisions
* Threshold sensitivity analysis and calibration table when labels are available

The interface is designed as a batch review tool rather than a model demo.

---

## Reproducibility

Dependencies are pinned in `requirements.txt`.

Each training run stores:

* Model type
* Selected threshold
* Expected cost
* PR-AUC
* Python and scikit-learn versions
* Unique run identifier

Artifacts are saved under:

```
artifacts/runs/<run_id>/
```

The latest model is copied to:

```
artifacts/model.joblib
artifacts/metrics.json
```

---

## Project Structure

```
fraud-risk/
│
├── app/                 # Streamlit dashboard
├── src/                 # Core training and inference logic
├── tests/               # Unit tests
├── data/                # Dataset and decision logs
├── artifacts/           # Versioned model artifacts
├── requirements.txt
├── Dockerfile
├── pytest.ini
└── README.md
```

---

## Setup

Install dependencies:

```
pip install -r requirements.txt
```

---

## Train

```
python -m src.train
```

---

## Run Dashboard

```
python -m streamlit run app/streamlit_app.py
```

The dashboard loads the default dataset automatically. Uploading a CSV allows scoring a new batch with the same schema.

---

## Run Tests

```
pytest
```

Tests validate:

* Schema enforcement and alignment
* Target column handling
* Risk bucketing boundaries
* Failure cases for malformed input

---

## Docker

Build:

```
docker build -t fraud-risk .
```

Run:

```
docker run -p 8501:8501 fraud-risk
```

Open:

```
http://localhost:8501
```

---

## Operational Notes

This project models batch scoring and review. In a production setting, additional components would typically be required:

* Data ingestion and schema contracts upstream
* Monitoring for drift and review load over time
* Alerting on distribution shifts or sudden threshold instability
* Decision persistence in a database with auditability
* A real-time scoring API or streaming pipeline (if required by the use case)

---

## Limitations

* The model is tied to a specific dataset schema.
* Pickled scikit-learn artifacts are version-sensitive.
* No streaming inference pipeline is implemented.
* No automated drift detection is included.
* Interpretability is limited due to PCA-transformed input features.

This project focuses on batch scoring and operational review simulation.

---

## Potential Extensions

* Probability calibration curves and reliability plots
* Batch-level drift monitoring and alerting
* Database-backed decision storage
* Real-time inference API
* Feature store abstraction

---
