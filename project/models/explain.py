"""SHAP TreeExplainer plots (waterfall + summary) matching the notebook xai class."""

from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import shap


class XAI:
    """SHAP explanations for tree models (XGBoost) on a fixed background matrix."""

    def __init__(
        self,
        model: Any,
        x_test: np.ndarray,
        feature_names: Optional[List[str]] = None,
        X_background: Optional[np.ndarray] = None,
    ) -> None:
        """Attach model, test design matrix, names; precompute SHAP values for summary plot."""
        _ = X_background
        self.model = model
        self.x_test = x_test
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(self.model)
        self.all_shap_values = self.explainer.shap_values(self.x_test)
        ev = np.asarray(self.explainer.expected_value).ravel()
        self.expected_value_for_waterfall = float(ev[1]) if ev.size > 1 else float(ev[0])

    def plot_waterfall(
        self,
        instance_index: int = 0,
        X_instance: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """Waterfall for one row (by index into x_test or explicit 1 x n_features vector)."""
        names = feature_names if feature_names is not None else self.feature_names
        if X_instance is not None:
            single = np.asarray(X_instance).reshape(1, -1)
            idx = 0
        else:
            single = self.x_test[instance_index].reshape(1, -1)
            idx = instance_index
        shap_vals_inst = self.explainer.shap_values(single)
        if isinstance(shap_vals_inst, list):
            shap_row = shap_vals_inst[1] if len(shap_vals_inst) > 1 else shap_vals_inst[0]
        else:
            shap_row = shap_vals_inst
        shap_1d = np.asarray(shap_row).ravel()
        explanation = shap.Explanation(
            values=shap_1d,
            base_values=self.expected_value_for_waterfall,
            data=single[0],
            feature_names=names,
        )
        shap.waterfall_plot(explanation, show=False)
        plt.title(f"SHAP Waterfall Plot for Instance {idx}")

    def plot_summary(
        self,
        X: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        ax: Any = None,
    ) -> None:
        """Global summary (class 1) using precomputed shap values on x_test."""
        _ = ax
        X = X if X is not None else self.x_test
        names = feature_names if feature_names is not None else self.feature_names
        shap.summary_plot(
            self.all_shap_values,
            X,
            feature_names=names,
            show=False,
            class_inds=1,
        )
        plt.title("SHAP Summary Plot (Class 1)")
