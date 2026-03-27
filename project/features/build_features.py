"""Build BERT embeddings, TF-IDF, and scaled numerical features."""

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from transformers import BertModel, BertTokenizer

import config


class BertEmbedding:
    """Compute CLS-token embeddings from text using pretrained BERT (notebook logic)."""

    def __init__(self, model_name: Optional[str] = None, max_length: Optional[int] = None) -> None:
        """Load tokenizer and BERT on CPU or CUDA."""
        name = model_name or config.BERT_MODEL_NAME
        self.max_length = max_length if max_length is not None else config.BERT_MAX_LENGTH
        self.tokenizer = BertTokenizer.from_pretrained(name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BertModel.from_pretrained(name).to(self.device)
        self.model.eval()

    def fit(self, texts: pd.Series) -> "BertEmbedding":
        """No-op fit for API symmetry; BERT is used in eval mode."""
        return self

    def embeddings(self, texts: pd.Series, batch_size: int = 32) -> torch.Tensor:
        """Return a tensor of shape (n_samples, hidden_size) for the text series."""
        embedding_tensors = []
        arr = texts if isinstance(texts, pd.Series) else pd.Series(texts)
        for i in range(0, len(arr), batch_size):
            batch = arr.iloc[i : i + batch_size].tolist()
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.model(**inputs).last_hidden_state[:, 0, :]
            embedding_tensors.append(out)
        return torch.cat(embedding_tensors, dim=0)

    def transform(self, texts: pd.Series) -> np.ndarray:
        """Return NumPy embedding matrix on CPU."""
        return self.embeddings(texts).cpu().numpy()


class TfidfFeatures:
    """Fit TF-IDF on training text and transform train/test (matches notebook params)."""

    def __init__(
        self,
        max_features: Optional[int] = None,
        min_df: Optional[int] = None,
        max_df: Optional[float] = None,
        ngram_range: Tuple[int, int] = (1, 2),
    ) -> None:
        """Configure TfidfVectorizer like the notebook."""
        self._vectorizer = TfidfVectorizer(
            max_features=max_features if max_features is not None else config.TFIDF_MAX_FEATURES,
            min_df=min_df if min_df is not None else config.TFIDF_MIN_DF,
            max_df=max_df if max_df is not None else config.TFIDF_MAX_DF,
            ngram_range=ngram_range,
        )

    def fit(self, texts: pd.Series) -> "TfidfFeatures":
        """Fit vocabulary on training text."""
        self._vectorizer.fit(texts)
        return self

    def transform(self, texts: pd.Series) -> np.ndarray:
        """Dense TF-IDF matrix for the given texts."""
        return self._vectorizer.transform(texts).toarray()

    @property
    def vectorizer(self) -> TfidfVectorizer:
        """Expose the fitted sklearn vectorizer (e.g. get_feature_names_out)."""
        return self._vectorizer


def scale_numerical(
    X_train: pd.DataFrame,
    X_val: Optional[pd.DataFrame] = None,
    scaler: Optional[StandardScaler] = None,
) -> Union[
    Tuple[np.ndarray, StandardScaler],
    Tuple[np.ndarray, np.ndarray, StandardScaler],
]:
    """Fit StandardScaler on train numerical columns; transform train and optional val/test."""
    scaler = scaler or StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if X_val is None:
        return X_train_scaled, scaler
    X_val_scaled = scaler.transform(X_val)
    return X_train_scaled, X_val_scaled, scaler


def combine_numerical_and_text_features(
    X_num_train: np.ndarray,
    X_num_test: np.ndarray,
    X_text_train: np.ndarray,
    X_text_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate scaled numerics with text-derived features (BERT or TF-IDF)."""
    X_tr = np.concatenate([X_num_train, X_text_train], axis=1)
    X_te = np.concatenate([X_num_test, X_text_test], axis=1)
    return X_tr, X_te
