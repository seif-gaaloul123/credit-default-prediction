"""End-to-end pipeline: load → clean → split → features → train TF-IDF & BERT → MLflow → optional SHAP."""

import argparse
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import config
from data.process import clean_data, load_data, temporal_split
from features.build_features import (
    BertEmbedding,
    TfidfFeatures,
    combine_numerical_and_text_features,
    scale_numerical,
)
from models.explain import XAI
from models.train import TrainModel
from tracking.mlflow_logger import log_run, setup_tracking


def run_pipeline(
    data_path: Optional[str] = None,
    skip_bert: bool = False,
    skip_mlflow: bool = False,
    skip_shap: bool = False,
    n_trials: Optional[int] = None,
) -> None:
    """Run full credit-default workflow aligned with the Colab notebook."""
    path = data_path or str(config.RAW_DATA_PATH)
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Data not found at {path}. Place {config.RAW_FILENAME} under data/raw/ "
            "or pass --data-path to your accepted_2007_to_2018Q4.csv.gz file."
        )

    df = load_data(path)
    df_clean, numerical_cols = clean_data(df)
    train, test = temporal_split(df_clean)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(config.PROCESSED_DIR / "data_clean.csv", index=False)

    X_num_train = train[numerical_cols].values
    X_num_test = test[numerical_cols].values
    y_train = train["target"].values
    y_test = test["target"].values
    X_text_train = train["combined_text"]
    X_text_test = test["combined_text"]

    X_num_train_s, X_num_test_s, scaler = scale_numerical(
        train[numerical_cols], test[numerical_cols]
    )

    # TF-IDF branch
    tfidf = TfidfFeatures()
    tfidf.fit(X_text_train)
    x_train_tfidf_txt = tfidf.transform(X_text_train)
    x_test_tfidf_txt = tfidf.transform(X_text_test)
    x_train_tfidf, x_test_tfidf = combine_numerical_and_text_features(
        X_num_train_s, X_num_test_s, x_train_tfidf_txt, x_test_tfidf_txt
    )

    # BERT branch (heavy)
    if not skip_bert:
        bert = BertEmbedding()
        bert.fit(X_text_train)
        x_train_bert = bert.transform(X_text_train)
        x_test_bert = bert.transform(X_text_test)
        X_train_bert_combined, X_test_bert_combined = combine_numerical_and_text_features(
            X_num_train_s, X_num_test_s, x_train_bert, x_test_bert
        )
    else:
        X_train_bert_combined = X_test_bert_combined = None

    artifacts_dir = config.PROCESSED_DIR
    np.save(artifacts_dir / "x_train_tfidf.npy", x_train_tfidf)
    np.save(artifacts_dir / "x_test_tfidf.npy", x_test_tfidf)
    np.save(artifacts_dir / "y_train.npy", y_train)
    np.save(artifacts_dir / "y_test.npy", y_test)
    if not skip_bert:
        np.save(artifacts_dir / "X_train_bert_combined.npy", X_train_bert_combined)
        np.save(artifacts_dir / "X_test_bert_combined.npy", X_test_bert_combined)

    joblib.dump(scaler, artifacts_dir / "scaler.pkl")
    joblib.dump(tfidf.vectorizer, artifacts_dir / "tfidf.pkl")

    if not skip_mlflow:
        setup_tracking()

    # Train + log TF-IDF model
    trainer_tfidf = TrainModel(x_train_tfidf, x_test_tfidf, y_train, y_test)
    trainer_tfidf.tune_train_model(n_trials=n_trials)
    tfidf_metrics, _, _ = trainer_tfidf.evaluate()
    print(f"metrics for tfidf model: {tfidf_metrics}")
    joblib.dump(trainer_tfidf.model, artifacts_dir / "xgb_model_tfidf.pkl")

    if not skip_mlflow:
        log_run(
            params=trainer_tfidf.model.get_params(),
            metrics={k: float(v) for k, v in tfidf_metrics.items() if k != "threshold"},
            model=trainer_tfidf.model,
            artifact_path="xgboost_tfidf_model",
            run_name="xgboost_tfidf",
        )

    # Train + log BERT-combined model
    if not skip_bert and X_train_bert_combined is not None:
        trainer_bert = TrainModel(X_train_bert_combined, X_test_bert_combined, y_train, y_test)
        trainer_bert.tune_train_model(n_trials=n_trials)
        bert_metrics, _, _ = trainer_bert.evaluate()
        print(f"metrics for bert model: {bert_metrics}")
        joblib.dump(trainer_bert.model, artifacts_dir / "xgb_model_bert.pkl")
        if not skip_mlflow:
            log_run(
                params=trainer_bert.model.get_params(),
                metrics={k: float(v) for k, v in bert_metrics.items() if k != "threshold"},
                model=trainer_bert.model,
                artifact_path="xgboost_bert_model",
                run_name="xgboost_bert",
            )

    # SHAP on TF-IDF model (notebook)
    if not skip_shap:
        all_feature_names = numerical_cols + list(tfidf.vectorizer.get_feature_names_out())
        explainable = XAI(trainer_tfidf.model, x_test_tfidf, all_feature_names)
        plots_dir = config.PROJECT_ROOT / "plots"
        plots_dir.mkdir(exist_ok=True)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        explainable.plot_summary()
        plt.tight_layout()
        plt.savefig(plots_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.subplot(1, 2, 1)
        explainable.plot_waterfall(instance_index=1)
        plt.tight_layout()
        plt.savefig(plots_dir / "shap_waterfall.png", dpi=150, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lending Club credit default pipeline")
    parser.add_argument("--data-path", type=str, default=None, help="Path to accepted_2007_to_2018Q4.csv.gz")
    parser.add_argument("--skip-bert", action="store_true", help="Skip BERT embeddings and second model")
    parser.add_argument("--skip-mlflow", action="store_true", help="Do not log to MLflow")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP plots")
    parser.add_argument("--n-trials", type=int, default=None, help="Optuna trials (default from config)")
    args = parser.parse_args()
    run_pipeline(
        data_path=args.data_path,
        skip_bert=args.skip_bert,
        skip_mlflow=args.skip_mlflow,
        skip_shap=args.skip_shap,
        n_trials=args.n_trials,
    )
