from __future__ import annotations

import json
import pandas as pd
import numpy as np
import streamlit as st
import joblib

from src.config import (
    LATEST_MODEL_PATH,
    LATEST_METRICS_PATH,
    RAW_PATH,
)
from src.risk import (
    to_risk_score,
    bucket,
    expected_cost,
    flag_top_k,
)
from src.inference import prepare_features

st.set_page_config(page_title="Fraud Review Console", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(LATEST_MODEL_PATH)


@st.cache_data
def load_metrics():
    with open(LATEST_METRICS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_default_df():
    return pd.read_csv(RAW_PATH)


model = load_model()
metrics = load_metrics()

st.title("Fraud Review Console")
st.caption("Cost-aware fraud scoring with ranking-based review simulation.")

st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload CSV", type="csv")
df = pd.read_csv(uploaded) if uploaded else load_default_df()

st.sidebar.header("Decision Policy")
mode = st.sidebar.radio("Mode", ["Threshold", "Top-K Review"])

threshold = None
review_capacity = None

if mode == "Threshold":
    threshold = st.sidebar.slider(
        "Threshold",
        0.01,
        0.99,
        float(metrics.get("threshold", 0.5)),
        0.01,
    )
else:
    review_capacity = (
        st.sidebar.slider("Review Capacity (%)", 1, 20, 2, 1) / 100.0
    )

try:
    X, _ = prepare_features(df, model)
except Exception as e:
    st.error("Input schema mismatch.")
    st.exception(e)
    st.stop()

probs = model.predict_proba(X)[:, 1]

scored = df.copy()
scored["fraud_probability"] = probs
scored["risk_score"] = scored["fraud_probability"].apply(to_risk_score)
scored["risk_bucket"] = scored["risk_score"].apply(bucket)

has_labels = "Class" in scored.columns
if has_labels:
    scored["Class"] = scored["Class"].astype(int)

if mode == "Threshold":
    scored["flagged"] = (scored["fraud_probability"] >= threshold).astype(int)
else:
    flags = flag_top_k(probs, review_capacity)
    scored["flagged"] = flags

n_total = len(scored)
n_flagged = int(scored["flagged"].sum())
flag_rate = n_flagged / n_total if n_total else 0.0

cost_fp = metrics["costs"]["false_positive_review"]
cost_fn = metrics["costs"]["missed_fraud"]

est_cost = None
if has_labels:
    est_cost = expected_cost(
        scored["Class"].values,
        scored["flagged"].values,
        cost_fp,
        cost_fn,
    )

c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions", f"{n_total:,}")
c2.metric("Flagged", f"{n_flagged:,}", f"{flag_rate*100:.2f}%")
c3.metric("Mode", mode)
c4.metric("Estimated Cost", f"{est_cost:,.0f}" if est_cost else "—")

tabs = st.tabs(["Overview", "Review Queue", "Inspector", "Model"])

with tabs[0]:
    st.subheader("Probability Distribution")
    st.bar_chart(
        scored["fraud_probability"]
        .round(2)
        .value_counts()
        .sort_index()
    )

    st.subheader("Risk Buckets")
    st.bar_chart(
        scored["risk_bucket"]
        .value_counts()
        .reindex(["High", "Medium", "Low"])
        .fillna(0)
    )

    st.subheader("Prediction Drift")

    baseline = metrics.get("baseline_stats", {})
    baseline_mean = baseline.get("mean_probability")
    current_mean = float(scored["fraud_probability"].mean())

    drift = None
    if baseline_mean is not None:
        drift = current_mean - baseline_mean

    d1, d2, d3 = st.columns(3)
    d1.metric("Baseline Mean", f"{baseline_mean:.4f}")
    d2.metric("Current Mean", f"{current_mean:.4f}")
    d3.metric("Delta", f"{drift:+.4f}" if drift else "—")

    if drift and abs(drift) > 0.02:
        st.warning("Significant prediction drift detected.")

with tabs[1]:
    st.subheader("Review Queue")

    view = (
        scored.sort_values("fraud_probability", ascending=False)
        .reset_index(drop=False)
        .rename(columns={"index": "_row_id"})
    )

    cols = [
        "_row_id",
        "fraud_probability",
        "risk_score",
        "risk_bucket",
        "flagged",
    ]

    if has_labels:
        cols.append("Class")

    st.dataframe(view[cols], use_container_width=True, height=500)

with tabs[2]:
    st.subheader("Transaction Inspector")

    idx = st.number_input(
        "Row index",
        0,
        max(len(scored) - 1, 0),
        0,
        1,
    )

    row = scored.iloc[int(idx)]

    summary = {
        "fraud_probability": float(row["fraud_probability"]),
        "risk_score": int(row["risk_score"]),
        "risk_bucket": row["risk_bucket"],
        "flagged": int(row["flagged"]),
    }

    if has_labels:
        summary["Class"] = int(row["Class"])

    st.json(summary)

with tabs[3]:
    st.subheader("Threshold Sensitivity")

    if has_labels:
        thresholds = np.linspace(0.01, 0.99, 99)
        rows = []

        for t in thresholds:
            preds = (probs >= t).astype(int)
            cost = expected_cost(
                scored["Class"].values,
                preds,
                cost_fp,
                cost_fn,
            )
            rows.append({"threshold": t, "cost": cost})

        curve = pd.DataFrame(rows).set_index("threshold")
        st.line_chart(curve)
    else:
        st.info("Upload data with 'Class' column to enable evaluation.")