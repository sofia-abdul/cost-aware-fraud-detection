from __future__ import annotations

import datetime as dt
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    DECISION_LOG_PATH,
    LATEST_METRICS_PATH,
    LATEST_MODEL_PATH,
    RAW_PATH,
    SCORE_LOG_PATH,
)
from src.inference import prepare_features
from src.monitoring import append_batch_log
from src.risk import (
    bucket,
    expected_cost,
    flag_top_k,
    to_risk_score,
)

st.set_page_config(page_title="Fraud Review Console", layout="wide")


@st.cache_resource
def load_model():
    if not LATEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {LATEST_MODEL_PATH}. "
            "Run training first to generate artifacts."
        )
    return joblib.load(LATEST_MODEL_PATH)


@st.cache_data
def load_metrics():
    if not LATEST_METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Metrics artifact not found at {LATEST_METRICS_PATH}. "
            "Run training first to generate artifacts."
        )
    with open(LATEST_METRICS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_default_df():
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Default dataset not found at {RAW_PATH}. "
            "Place the dataset in the configured raw data directory."
        )
    return pd.read_csv(RAW_PATH)


def append_review_decision(row_id: int, action: str, row: pd.Series) -> None:
    DECISION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "row_id": int(row_id),
        "action": action,
        "fraud_probability": float(row["fraud_probability"]),
        "risk_score": int(row["risk_score"]),
        "risk_bucket": row["risk_bucket"],
        "flagged": int(row["flagged"]),
    }

    if "Amount" in row.index:
        record["Amount"] = float(row["Amount"])
    if "Time" in row.index:
        record["Time"] = float(row["Time"])
    if "Class" in row.index:
        record["Class"] = int(row["Class"])

    out = pd.DataFrame([record])

    if DECISION_LOG_PATH.exists():
        out.to_csv(DECISION_LOG_PATH, mode="a", header=False, index=False)
    else:
        out.to_csv(DECISION_LOG_PATH, index=False)


def main():
    st.title("Fraud Review Console")
    st.caption("Cost-aware fraud scoring with ranking-based review simulation.")

    try:
        model = load_model()
        metrics = load_metrics()
        df = load_default_df()
    except Exception as e:
        st.error("Failed to initialize the application.")
        st.exception(e)
        st.stop()

    st.sidebar.header("Data")
    st.sidebar.caption("Using the built-in dataset configured for this project.")
    st.sidebar.text(f"Dataset: {RAW_PATH.name}")

    st.sidebar.header("Decision Policy")
    mode = st.sidebar.radio("Mode", ["Threshold", "Top-K Review"])

    threshold = None
    review_capacity = None

    if mode == "Threshold":
        threshold = st.sidebar.slider(
            "Threshold",
            min_value=0.01,
            max_value=0.99,
            value=float(metrics.get("threshold", 0.5)),
            step=0.01,
        )
    else:
        review_capacity = (
            st.sidebar.slider("Review Capacity (%)", 1, 20, 2, 1) / 100.0
        )

    try:
        X, _ = prepare_features(df, model)
    except Exception as e:
        st.error("Input schema mismatch between dataset and trained model.")
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
        scored["flagged"] = flag_top_k(probs, review_capacity)

    n_total = len(scored)
    n_flagged = int(scored["flagged"].sum())
    flag_rate = (n_flagged / n_total) if n_total else 0.0

    avg_prob = float(scored["fraud_probability"].mean())
    p95_prob = float(scored["fraud_probability"].quantile(0.95))

    append_batch_log(
        SCORE_LOG_PATH,
        metrics.get("run_id", "unknown"),
        n_total,
        n_flagged,
        threshold if threshold is not None else 0.0,
        avg_prob,
        p95_prob,
    )

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
    c2.metric("Flagged", f"{n_flagged:,}", f"{flag_rate * 100:.2f}%")
    c3.metric("Mode", mode)
    c4.metric(
        "Estimated Cost",
        f"{est_cost:,.0f}" if est_cost is not None else "—",
    )

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
        d1.metric(
            "Baseline Mean",
            f"{baseline_mean:.4f}" if baseline_mean is not None else "—",
        )
        d2.metric("Current Mean", f"{current_mean:.4f}")
        d3.metric(
            "Delta",
            f"{drift:+.4f}" if drift is not None else "—",
        )

        if drift is not None and abs(drift) > 0.02:
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

        if "Amount" in view.columns:
            cols.append("Amount")
        if "Time" in view.columns:
            cols.append("Time")
        if has_labels:
            cols.append("Class")

        st.dataframe(view[cols], use_container_width=True, height=500)

    with tabs[2]:
        st.subheader("Transaction Inspector")

        idx = st.number_input(
            "Row index",
            min_value=0,
            max_value=max(len(scored) - 1, 0),
            value=0,
            step=1,
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

        action = st.selectbox(
            "Reviewer action",
            ["Approve", "Reject", "Escalate"],
        )

        if st.button("Save review decision"):
            try:
                append_review_decision(int(idx), action, row)
                st.success(f"Saved decision: {action}")
            except Exception as e:
                st.error("Failed to save review decision.")
                st.exception(e)

        with st.expander("View full row data"):
            st.dataframe(pd.DataFrame([row]), use_container_width=True)

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
                rows.append({"threshold": float(t), "cost": float(cost)})

            curve = pd.DataFrame(rows).set_index("threshold")
            st.line_chart(curve)
        else:
            st.info("This view requires labels in the dataset to evaluate cost.")

    st.caption(
        "Built-in demo mode uses the configured default dataset and saved model artifacts."
    )


if __name__ == "__main__":
    main()