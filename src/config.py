from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
RAW_PATH = DATA_DIR / "raw" / "creditcard.csv"

ARTIFACT_DIR = BASE_DIR / "artifacts"
RUNS_DIR = ARTIFACT_DIR / "runs"

LATEST_MODEL_PATH = ARTIFACT_DIR / "model.joblib"
LATEST_METRICS_PATH = ARTIFACT_DIR / "metrics.json"

DECISIONS_DIR = DATA_DIR / "decisions"
DECISION_LOG_PATH = DECISIONS_DIR / "review_log.csv"

MONITORING_DIR = DATA_DIR / "monitoring"
SCORE_LOG_PATH = MONITORING_DIR / "score_log.csv"

COST_FALSE_POSITIVE_REVIEW = 5.0
COST_MISSED_FRAUD = 200.0

RANDOM_STATE = 42