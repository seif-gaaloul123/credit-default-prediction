"""Log parameters, metrics, and XGBoost models to MLflow (DagsHub URI from config/env)."""

from typing import Any, Dict, Optional

import mlflow

import config


def log_run(
    params: Dict[str, Any],
    metrics: Dict[str, float],
    model: Any,
    artifact_path: Optional[str] = None,
    run_name: Optional[str] = None,
    log_model: bool = True,
) -> None:
    """Log params/metrics and optionally the XGBoost estimator under an MLflow run."""
    name = artifact_path or "model"
    with mlflow.start_run(run_name=run_name):
        for k, v in params.items():
            if v is None or isinstance(v, (bool, int, float, str)):
                mlflow.log_param(str(k), v)
            else:
                mlflow.log_param(str(k), str(v))
        for k, v in metrics.items():
            if k == "threshold":
                mlflow.log_param("optimal_threshold", float(v))
            else:
                mlflow.log_metric(k, float(v))
        if log_model and model is not None:
            mlflow.xgboost.log_model(model, name)


def setup_tracking() -> None:
    """Point MLflow at the tracking URI and ensure the experiment exists."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
