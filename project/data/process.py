"""Load, clean, and temporally split Lending Club data."""

import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import config


def _clean_text_scalar(text: object) -> str:
    """Lowercase, strip URLs/non-letters, and collapse whitespace for one text value."""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text(series: pd.Series) -> pd.Series:
    """Normalize text fields element-wise for NLP features."""
    return series.map(_clean_text_scalar)


def load_data(path: Optional[str] = None, sample_n: Optional[int] = None) -> pd.DataFrame:
    """Load Lending Club CSV (gzip) from disk; optionally subsample rows."""
    csv_path = path or str(config.RAW_DATA_PATH)
    df = pd.read_csv(csv_path, compression="gzip", low_memory=False)
    n = sample_n if sample_n is not None else config.SAMPLE_N
    if n is not None and len(df) > n:
        df = df.sample(n=n, random_state=config.RANDOM_SEED)
    return df


def select_numerical_columns(df: pd.DataFrame) -> List[str]:
    """Pick numeric columns with enough non-null coverage, excluding id/target/leakage proxies."""
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numerical_cols = [c for c in numerical_cols if c not in config.EXCLUDE_NUMERICAL_COLS]
    numerical_cols = [
        col
        for col in numerical_cols
        if df[col].notna().sum() / len(df) > 0.5
    ]
    return numerical_cols


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Drop sparse columns, filter loan outcomes, build combined text, impute/clip numerics."""
    th = config.MISSING_COL_THRESHOLD
    missing_values = df.isna().sum() / len(df)
    drop_cols = missing_values[missing_values > th].index
    df_clean = df.drop(columns=drop_cols).copy()

    df_clean = df_clean[df_clean["loan_status"].isin(["Fully Paid", "Charged Off"])].copy()
    df_clean["target"] = (df_clean["loan_status"] == "Charged Off").astype(int)

    existing_text_cols = [c for c in config.TEXT_COLUMNS if c in df_clean.columns]
    df_clean["combined_text"] = ""
    for col in existing_text_cols:
        df_clean["combined_text"] = df_clean["combined_text"] + df_clean[col].apply(_clean_text_scalar) + " "
    df_clean["combined_text"] = df_clean["combined_text"].str.strip()
    df_clean = df_clean[df_clean["combined_text"].str.len() >= config.MIN_TEXT_LENGTH].copy()

    numerical_cols = select_numerical_columns(df_clean)
    for col in numerical_cols:
        if df_clean[col].isna().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    for col in numerical_cols:
        lower = df_clean[col].quantile(0.01)
        upper = df_clean[col].quantile(0.99)
        df_clean[col] = df_clean[col].clip(lower, upper)

    return df_clean, numerical_cols


def temporal_split(
    df: pd.DataFrame,
    date_column: str = "issue_d",
    split_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into train (before split_date) and test (on or after split_date)."""
    split = pd.to_datetime(split_date or config.SPLIT_DATE)
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    train = df[df[date_column] < split]
    test = df[df[date_column] >= split]
    return train, test
