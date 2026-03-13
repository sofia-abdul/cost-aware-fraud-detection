from __future__ import annotations

from pathlib import Path
import datetime as dt
import pandas as pd


def append_batch_log(
    out_path: Path,
    run_id: str,
    n_total: int,
    n_flagged: int,
    threshold: float,
    avg_prob: float,
    p95_prob: float,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp_utc": dt.datetime.utcnow().isoformat(timespec="seconds"),
        "run_id": run_id,
        "n_total": n_total,
        "n_flagged": n_flagged,
        "flag_rate": (n_flagged / n_total) if n_total else 0.0,
        "threshold": threshold,
        "avg_prob": avg_prob,
        "p95_prob": p95_prob,
    }
    df = pd.DataFrame([row])
    if out_path.exists():
        df.to_csv(out_path, mode="a", header=False, index=False)
    else:
        df.to_csv(out_path, index=False)
