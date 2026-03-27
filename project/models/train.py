"""Train credit default models with Optuna tuning (notebook train_bert_tfidf logic)."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score

import config


class TrainModel:
    """XGBoost + Optuna CV search, then refit on full train; predict and evaluate like the notebook."""

    def __init__(
        self,
        x_train: np.ndarray,
        x_eval: np.ndarray,
        y_train: np.ndarray,
        y_eval: np.ndarray,
    ) -> None:
        """Store train/eval matrices and labels for tuning and evaluation."""
        self.x_train = x_train
        self.x_eval = x_eval
        self.y_train = y_train
        self.y_eval = y_eval
        self.model: Optional[xgb.XGBClassifier] = None

    def tune_train_model(self, n_trials: Optional[int] = None) -> xgb.XGBClassifier:
        """Optuna 3-fold CV ROC-AUC on training data, then fit best XGBoost on full train."""
        n_trials = n_trials if n_trials is not None else config.OPTUNA_N_TRIALS
        neg = (self.y_train == 0).sum()
        pos = (self.y_train == 1).sum()
        scale_pos_weight = float(neg / pos) if pos > 0 else 1.0

        def objective(trial: optuna.Trial) -> float:
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
                "scale_pos_weight": scale_pos_weight,
                "random_state": config.RANDOM_SEED,
                "eval_metric": "auc",
                "objective": "binary:logistic",
            }
            clf = xgb.XGBClassifier(**params)
            scores = cross_val_score(
                clf,
                self.x_train,
                self.y_train,
                cv=3,
                scoring="roc_auc",
            )
            return float(scores.mean())

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = dict(study.best_params)
        best_params["scale_pos_weight"] = scale_pos_weight
        best_params["random_state"] = config.RANDOM_SEED
        best_params["eval_metric"] = "auc"
        best_params["objective"] = "binary:logistic"

        self.model = xgb.XGBClassifier(**best_params)
        self.model.fit(self.x_train, self.y_train)
        return self.model

    def predict(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Return positive-class probabilities; defaults to held-out eval matrix."""
        if self.model is None:
            raise RuntimeError("Call tune_train_model() first.")
        X = self.x_eval if X is None else X
        return self.model.predict_proba(X)[:, 1]

    def evaluate(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """Threshold from precision*recall on PR curve; return metrics, proba, and binary preds."""
        X = self.x_eval if X is None else X
        y = self.y_eval if y is None else y
        pred_proba = self.predict(X)
        precision, recall, thresholds = precision_recall_curve(y, pred_proba)
        scores = precision[:-1] * recall[:-1]
        threshold = float(thresholds[np.argmax(scores)])
        pred = (pred_proba >= threshold).astype(int)
        metrics = {
            "threshold": threshold,
            "average_precision_score": float(average_precision_score(y, pred_proba)),
            "recall_score": float(recall_score(y, pred)),
            "roc_auc_score": float(roc_auc_score(y, pred_proba)),
        }
        return metrics, pred_proba, pred
