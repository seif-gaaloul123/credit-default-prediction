# Credit Default Prediction (Lending Club)

End-to-end machine learning pipeline for credit default prediction using the Lending Club dataset.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Layout

- `config.py` — paths, model settings, random seed, temporal split date
- `data/process.py` — load, clean, temporal split, text cleaning
- `features/build_features.py` — BERT embeddings, TF-IDF, numerical scaling
- `models/train.py` — training with optional tuning, prediction, evaluation
- `models/explain.py` — SHAP-based explanations
- `tracking/mlflow_logger.py` — MLflow experiment logging
- `main.py` — full pipeline orchestration
