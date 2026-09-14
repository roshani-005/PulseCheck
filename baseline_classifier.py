"""Month-3 static classification baseline for PulseCheck."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from survival_model import FEATURES


def fit_logistic_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str | Path,
    features: Sequence[str] = FEATURES,
) -> tuple[Pipeline, dict, np.ndarray]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logit", LogisticRegression(max_iter=2500, class_weight="balanced", random_state=42)),
        ]
    )
    model.fit(train_df[list(features)], train_df["defaults_next_3m"])
    prob = model.predict_proba(test_df[list(features)])[:, 1]
    y = test_df["defaults_next_3m"].to_numpy()

    metrics = {
        "roc_auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else None,
        "average_precision": float(average_precision_score(y, prob)) if len(np.unique(y)) > 1 else None,
        "brier": float(brier_score_loss(y, prob)),
        "test_prevalence": float(np.mean(y)),
    }
    (output_dir / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if len(np.unique(y)) > 1:
        fpr, tpr, _ = roc_curve(y, prob)
        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.plot(fpr, tpr, label=f"Logistic AUC = {metrics['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.set(xlabel="False positive rate", ylabel="True positive rate", title="3-month default classifier ROC")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / "baseline_roc.png", dpi=160)
        plt.close(fig)

        frac_pos, mean_pred = calibration_curve(y, prob, n_bins=8, strategy="quantile")
        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.plot(mean_pred, frac_pos, marker="o")
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.set(xlabel="Mean predicted probability", ylabel="Observed default rate", title="Baseline calibration")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / "baseline_calibration.png", dpi=160)
        plt.close(fig)

    return model, metrics, prob
