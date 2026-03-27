"""Project-wide constants: paths, model hyperparameters, random seed, and temporal split."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Paths (adjust to your Lending Club data location)
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
# Default filename from Kaggle Lending Club archive
RAW_FILENAME: str = "accepted_2007_to_2018Q4.csv.gz"
RAW_DATA_PATH: Path = DATA_DIR / RAW_FILENAME

# Reproducibility
RANDOM_SEED: int = 42

# Optional row cap (None = use all rows after cleaning); notebook used 100_000
SAMPLE_N: Optional[int] = 100_000

# Drop columns with more than this fraction of missing values
MISSING_COL_THRESHOLD: float = 0.8

# Temporal split (train: issue_d < SPLIT_DATE, test: >= SPLIT_DATE) — matches notebook
SPLIT_DATE: str = "2017-01-01"

# Text fields combined for NLP (only those present in the frame are used)
TEXT_COLUMNS: List[str] = ["emp_title", "title", "desc", "purpose"]
MIN_TEXT_LENGTH: int = 10

# Columns excluded from numerical features (ids, target, obvious leakage proxies)
EXCLUDE_NUMERICAL_COLS: List[str] = [
    "id",
    "member_id",
    "target",
    "policy_code",
    "recoveries",
    "collection_recovery_fee",
    "total_rec_late_fee",
]

# Model defaults (overridden by Optuna)
MODEL_PARAMS: Dict[str, Any] = {
    "random_state": RANDOM_SEED,
    "eval_metric": "auc",
    "objective": "binary:logistic",
}

# Feature / NLP (notebook: max_features=500, min_df=5, max_df=0.8, ngram (1,2))
BERT_MODEL_NAME: str = "bert-base-uncased"
BERT_MAX_LENGTH: int = 128
TFIDF_MAX_FEATURES: int = 500
TFIDF_MIN_DF: int = 5
TFIDF_MAX_DF: float = 0.8

# Optuna
OPTUNA_N_TRIALS: int = 50

# MLflow / DagsHub — set via environment or edit before running
MLFLOW_TRACKING_URI: str = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "https://dagshub.com/seif-gaaloul123/credit-default-prediction.mlflow",
)
MLFLOW_EXPERIMENT_NAME: str = os.environ.get("MLFLOW_EXPERIMENT_NAME", "credit-default")
