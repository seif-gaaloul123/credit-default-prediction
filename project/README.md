# Credit Default Prediction

Predicts the probability of loan default on the Lending Club dataset (2007-2018). 



## Results

| Model | ROC-AUC | Recall | Avg Precision |
|-------|---------|--------|---------------|
| XGBoost + TF-IDF | 0.713 | 0.854 | 0.383 |
| XGBoost + BERT | 0.717 | 0.846 | 0.387 |

BERT gains 0.4% AUC but loses 0.8% recall with significantly higher compute cost. For short formulaic financial text, TF-IDF is sufficient.

Evaluated on a temporal split (train: 2007-2016, test: 2017+) rather than random split to simulate real production conditions.

## Experiment Tracking
https://dagshub.com/seif-gaaloul123/credit-default-prediction/experiments

## Setup
```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Stack
- XGBoost with Optuna hyperparameter tuning (50 trials, 3-fold CV)
- TF-IDF (500 features) and BERT embeddings for text
- SHAP TreeExplainer for prediction explainability
- MLflow + DagsHub for experiment tracking

## Project Structure
```
├── app/                  # Streamlit demo
├── artifacts/            # Saved models and preprocessors
├── models/               # Training and explainability
├── data/                 # Preprocessing pipeline
├── features/             # TF-IDF and BERT feature extraction
├── tracking/             # MLflow logging
└── notebooks/            # EDA and modeling notebooks
```

## Dataset
Lending Club loan data 2007-2018, available on [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club).
