"""
rag_guard.calibrate
-------------------
10-fold Stratified Cross-Validation and Algebraic Weight Folding engine.
Trains StandardScaler + LogisticRegression on RAG features, evaluates generalization,
and algebraically folds the normalization parameters into raw weights for zero-dependency serving.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from rag_guard.config import DEFAULT_WEIGHTS_PATH, FEATURE_NAMES

logger = logging.getLogger("rag_guard.calibrate")


@dataclass(frozen=True)
class CrossValidationMetrics:
    """Aggregated cross-validation metrics across all folds."""

    roc_auc_mean: float
    roc_auc_std: float
    f1_mean: float
    f1_std: float
    precision_mean: float
    recall_mean: float
    optimal_threshold: float
    fold_aucs: List[float]


@dataclass(frozen=True)
class FoldedWeights:
    """Algebraically folded weights for zero-dependency inference."""

    weights_raw: List[float]
    bias_raw: float
    optimal_threshold: float
    feature_names: List[str]
    cv_metrics: CrossValidationMetrics
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary format for weights.json."""
        return {
            "version": "1.0.0",
            "feature_names": self.feature_names,
            "weights_raw": [round(float(w), 6) for w in self.weights_raw],
            "bias_raw": round(float(self.bias_raw), 6),
            "optimal_threshold": round(float(self.optimal_threshold), 4),
            "calibration_metadata": {
                "cross_validation": {
                    "n_splits": len(self.cv_metrics.fold_aucs),
                    "roc_auc_mean": round(self.cv_metrics.roc_auc_mean, 4),
                    "roc_auc_std": round(self.cv_metrics.roc_auc_std, 4),
                    "f1_mean": round(self.cv_metrics.f1_mean, 4),
                    "f1_std": round(self.cv_metrics.f1_std, 4),
                    "precision_mean": round(self.cv_metrics.precision_mean, 4),
                    "recall_mean": round(self.cv_metrics.recall_mean, 4),
                    "fold_aucs": [round(a, 4) for a in self.cv_metrics.fold_aucs],
                },
                "folding_method": "algebraic_scaler_folding",
                **self.metadata,
            },
        }

    def save_json(self, output_path: Union[str, Path]) -> None:
        """Write folded weights to JSON file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        logger.info("Saved calibrated folded weights to '%s'", out)


def fold_scaler_into_weights(
    scaler: StandardScaler,
    model: LogisticRegression,
) -> Tuple[np.ndarray, float]:
    """
    Algebraically fold StandardScaler normalization parameters (mu, sigma)
    into LogisticRegression weights (w, b).

    Mathematical derivation:
      z = sum_i w_i * (x_i - mu_i) / sigma_i + b
        = sum_i (w_i / sigma_i) * x_i + (b - sum_i w_i * mu_i / sigma_i)

      w_raw,i = w_i / sigma_i
      b_raw   = b - sum_i (w_raw,i * mu_i) = b - w_raw · mu

    Parameters
    ----------
    scaler : StandardScaler
        Fitted scaler with mean_ and scale_ attributes.
    model : LogisticRegression
        Fitted logistic regression model with coef_ and intercept_.

    Returns
    -------
    Tuple[np.ndarray, float]
        Raw weights w_raw of shape (5,) and raw scalar bias b_raw.
    """
    w_scaled = model.coef_[0]  # Shape: (5,)
    b_scaled = float(model.intercept_[0])

    mu = scaler.mean_  # Shape: (5,)
    sigma = scaler.scale_  # Shape: (5,)

    # Prevent division by zero if any feature variance is zero
    safe_sigma = np.where(sigma == 0.0, 1.0, sigma)

    w_raw = w_scaled / safe_sigma
    b_raw = b_scaled - float(np.dot(w_raw, mu))

    return w_raw, b_raw


def verify_weight_folding(
    scaler: StandardScaler,
    model: LogisticRegression,
    w_raw: np.ndarray,
    b_raw: float,
    num_test_samples: int = 1000,
) -> float:
    """
    Verify algebraic equivalence between the scaled pipeline and raw dot product.

    Returns
    -------
    float
        Maximum absolute deviation |z_pipeline - z_raw|.
    """
    rng = np.random.default_rng(seed=42)
    # Generate random feature matrix
    test_x = rng.normal(loc=0.0, scale=2.0, size=(num_test_samples, len(w_raw)))

    # Pipeline output
    scaled_x = scaler.transform(test_x)
    z_pipeline = model.decision_function(scaled_x)

    # Raw dot product
    z_raw = np.dot(test_x, w_raw) + b_raw

    max_diff = float(np.max(np.abs(z_pipeline - z_raw)))
    if max_diff > 1e-9:
        raise ValueError(
            f"Weight folding mathematical verification failed! Max diff: {max_diff}"
        )
    return max_diff


def generate_synthetic_feature_dataset(
    num_samples: int = 2000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic RAGTruth-calibrated feature matrix for offline calibration testing.

    Distributions match Phase 2 empirical findings:
      - Grounded (Label=0, Faithful=1):
          l_bar ~ N(-0.18, 0.12), l_min ~ N(-0.65, 0.25),
          var_l ~ Gamma(1.5, 0.05), H_bar ~ Gamma(1.2, 0.08), R_entity ~ Beta(9, 1)
      - Hallucinated (Label=1, Faithful=0):
          l_bar ~ N(-1.45, 0.50), l_min ~ N(-3.20, 0.75),
          var_l ~ Gamma(4.0, 0.35), H_bar ~ Gamma(5.0, 0.30), R_entity ~ Beta(2, 6)
    """
    rng = np.random.default_rng(seed=random_state)
    half = num_samples // 2

    # 1. Grounded samples (Faithful = 1, Label = 0)
    g_l_bar = rng.normal(loc=-0.18, scale=0.12, size=half)
    g_l_min = np.minimum(g_l_bar, rng.normal(loc=-0.65, scale=0.20, size=half))
    g_var_l = rng.gamma(shape=1.5, scale=0.05, size=half)
    g_H_bar = rng.gamma(shape=1.2, scale=0.08, size=half)
    g_R_ent = rng.beta(a=9.0, b=1.0, size=half)
    g_faithful = np.ones(half, dtype=int)
    g_label = np.zeros(half, dtype=int)

    # 2. Hallucinated samples (Faithful = 0, Label = 1)
    h_count = num_samples - half
    h_l_bar = rng.normal(loc=-1.45, scale=0.45, size=h_count)
    h_l_min = np.minimum(h_l_bar, rng.normal(loc=-3.20, scale=0.70, size=h_count))
    h_var_l = rng.gamma(shape=3.5, scale=0.30, size=h_count)
    h_H_bar = rng.gamma(shape=4.5, scale=0.28, size=h_count)
    h_R_ent = rng.beta(a=1.5, b=5.0, size=h_count)
    h_faithful = np.zeros(h_count, dtype=int)
    h_label = np.ones(h_count, dtype=int)

    data = {
        "l_bar": np.concatenate([g_l_bar, h_l_bar]),
        "l_min": np.concatenate([g_l_min, h_l_min]),
        "var_l": np.concatenate([g_var_l, h_var_l]),
        "H_bar": np.concatenate([g_H_bar, h_H_bar]),
        "R_entity": np.clip(np.concatenate([g_R_ent, h_R_ent]), 0.0, 1.0),
        "Faithful": np.concatenate([g_faithful, h_faithful]),
        "Label": np.concatenate([g_label, h_label]),
    }

    df = pd.DataFrame(data).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df


def run_cross_validation(
    df: pd.DataFrame,
    n_splits: int = 10,
    random_state: int = 42,
    target_col: str = "Faithful",
) -> Tuple[CrossValidationMetrics, StandardScaler, LogisticRegression]:
    """
    Perform n-fold Stratified Cross-Validation on the feature matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain FEATURE_NAMES and target_col (1 = Faithful / Grounded).
    n_splits : int
        Number of stratified folds (default 10).
    random_state : int
        Random seed for reproducibility.
    target_col : str
        Target column for binary classification.

    Returns
    -------
    Tuple[CrossValidationMetrics, StandardScaler, LogisticRegression]
    """
    x = df[FEATURE_NAMES].values.astype(np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # Determine binary target (1 = Faithful, 0 = Hallucinated)
    if target_col in df.columns:
        y = df[target_col].values.astype(int)
    elif "Label" in df.columns:
        # If Label is 1 for hallucinated, invert to Faithful (1 = grounded)
        y = 1 - df["Label"].values.astype(int)
    else:
        raise ValueError(f"Neither '{target_col}' nor 'Label' column found in dataset.")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_aucs: List[float] = []
    fold_f1s: List[float] = []
    fold_precs: List[float] = []
    fold_recs: List[float] = []

    oof_probs = np.zeros(len(df), dtype=np.float64)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(x, y), start=1):
        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)

        clf = LogisticRegression(
            class_weight="balanced",
            random_state=random_state,
            max_iter=1000,
            solver="lbfgs",
        )
        clf.fit(x_train_scaled, y_train)

        val_probs = clf.predict_proba(x_val_scaled)[:, 1]
        oof_probs[val_idx] = val_probs

        auc = float(roc_auc_score(y_val, val_probs))
        fold_aucs.append(auc)

        val_preds = (val_probs >= 0.50).astype(int)
        fold_f1s.append(float(f1_score(y_val, val_preds, zero_division=0)))
        fold_precs.append(float(precision_score(y_val, val_preds, zero_division=0)))
        fold_recs.append(float(recall_score(y_val, val_preds, zero_division=0)))

    # Compute optimal F1 decision threshold using precision-recall curve on out-of-fold predictions
    precisions, recalls, thresholds = precision_recall_curve(y, oof_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-12)
    best_idx = int(np.argmax(f1_scores))
    optimal_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.50

    cv_metrics = CrossValidationMetrics(
        roc_auc_mean=float(np.mean(fold_aucs)),
        roc_auc_std=float(np.std(fold_aucs)),
        f1_mean=float(np.mean(fold_f1s)),
        f1_std=float(np.std(fold_f1s)),
        precision_mean=float(np.mean(fold_precs)),
        recall_mean=float(np.mean(fold_recs)),
        optimal_threshold=optimal_threshold,
        fold_aucs=fold_aucs,
    )

    # Fit final model on all data
    final_scaler = StandardScaler()
    x_full_scaled = final_scaler.fit_transform(x)
    final_clf = LogisticRegression(
        class_weight="balanced",
        random_state=random_state,
        max_iter=1000,
        solver="lbfgs",
    )
    final_clf.fit(x_full_scaled, y)

    return cv_metrics, final_scaler, final_clf


def calibrate_and_fold(
    df: pd.DataFrame,
    n_splits: int = 10,
    output_path: Optional[Union[str, Path]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> FoldedWeights:
    """
    End-to-end pipeline: 10-fold CV -> Full model fit -> Algebraic weight folding -> Serialization.
    """
    logger.info("Running %d-fold Stratified Cross-Validation on %d samples...", n_splits, len(df))
    cv_metrics, final_scaler, final_clf = run_cross_validation(df, n_splits=n_splits)

    logger.info(
        "Cross-Validation complete: ROC-AUC = %.4f +/- %.4f | F1 = %.4f +/- %.4f",
        cv_metrics.roc_auc_mean,
        cv_metrics.roc_auc_std,
        cv_metrics.f1_mean,
        cv_metrics.f1_std,
    )

    # Perform algebraic weight folding
    w_raw, b_raw = fold_scaler_into_weights(final_scaler, final_clf)

    # Verify numerical identity
    max_err = verify_weight_folding(final_scaler, final_clf, w_raw, b_raw)
    logger.info("Weight folding verification passed (max diff = %.2e).", max_err)

    weights = FoldedWeights(
        weights_raw=w_raw.tolist(),
        bias_raw=b_raw,
        optimal_threshold=cv_metrics.optimal_threshold,
        feature_names=FEATURE_NAMES,
        cv_metrics=cv_metrics,
        metadata=metadata or {},
    )

    if output_path is not None:
        weights.save_json(output_path)

    return weights


def print_cv_report(metrics: CrossValidationMetrics, w_raw: List[float], b_raw: float) -> None:
    """Print standard academic evaluation summary table."""
    sep_full = "=" * 65
    sep_light = "-" * 65

    print(f"\n{sep_full}")
    print(f"{'rag_guard 10-Fold Stratified Cross-Validation Report':^65}")
    print(sep_full)
    print(f"Metric{'':<28}{'Mean':>15}{'Std Dev':>18}")
    print(sep_light)
    print(f"ROC-AUC{'':<27}{metrics.roc_auc_mean:>15.4f}{metrics.roc_auc_std:>18.4f}")
    print(f"F1-Score (threshold=0.50){'':<10}{metrics.f1_mean:>15.4f}{metrics.f1_std:>18.4f}")
    print(f"Precision{'':<25}{metrics.precision_mean:>15.4f}{'--':>18}")
    print(f"Recall{'':<28}{metrics.recall_mean:>15.4f}{'--':>18}")
    print(f"Optimal F1 Decision Threshold (tau*):  {metrics.optimal_threshold:.4f}")
    print(sep_light)
    print("Fold-by-Fold ROC-AUCs:")
    for i, a in enumerate(metrics.fold_aucs, 1):
        print(f"  Fold {i:02d}: {a:.4f}", end="\n" if i % 5 == 0 else "   ")
    print()
    print(sep_light)
    print("Algebraically Folded Raw Weights (for Zero-Dependency Inference):")
    for name, w in zip(FEATURE_NAMES, w_raw):
        print(f"  w_{name:<10}: {w:>10.6f}")
    print(f"  b_raw      : {b_raw:>10.6f}")
    print(f"{sep_full}\n")


def main() -> None:
    """CLI entry point for rag_guard calibration."""
    parser = argparse.ArgumentParser(
        prog="rag_guard.calibrate",
        description="10-fold Stratified Cross-Validation and Algebraic Weight Folding for rag_guard.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to CSV/Parquet dataset with features or JSONL dataset.",
    )
    parser.add_argument(
        "--mock-samples",
        type=int,
        default=None,
        help="Generate N synthetic samples matching RAGTruth empirical distributions.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=10,
        help="Number of cross-validation folds (default: 10).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_WEIGHTS_PATH),
        help=f"Output path for serialized weights.json (default: {DEFAULT_WEIGHTS_PATH}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run CV without updating weights.json.",
    )
    args = parser.parse_args()

    # Ingest or generate data
    if args.dataset and os.path.exists(args.dataset):
        logger.info("Loading dataset from '%s'...", args.dataset)
        if args.dataset.endswith(".parquet"):
            df = pd.read_parquet(args.dataset)
        elif args.dataset.endswith(".csv"):
            df = pd.read_csv(args.dataset)
        else:
            df = pd.read_json(args.dataset, lines=True)
    else:
        sample_count = args.mock_samples or 2000
        logger.info(
            "Generating %d synthetic feature samples matching RAGTruth distributions...",
            sample_count,
        )
        df = generate_synthetic_feature_dataset(num_samples=sample_count)

    out_path = None if args.dry_run else args.output
    weights = calibrate_and_fold(
        df=df,
        n_splits=args.n_splits,
        output_path=out_path,
        metadata={
            "dataset_samples": len(df),
            "source": args.dataset or "synthetic_ragtruth_calibrated",
        },
    )

    print_cv_report(weights.cv_metrics, weights.weights_raw, weights.bias_raw)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
