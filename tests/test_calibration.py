"""
tests/test_calibration.py
-------------------------
Unit tests for 10-fold cross-validation and algebraic scaler folding.
Verifies exact algebraic equivalence, serialization, and detector compatibility.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from rag_guard.calibrate import (
    calibrate_and_fold,
    fold_scaler_into_weights,
    generate_synthetic_feature_dataset,
    run_cross_validation,
    verify_weight_folding,
)
from rag_guard.detector import CompositeHallucinationDetector


class TestCalibrationAndFolding(unittest.TestCase):
    def test_algebraic_weight_folding_exactness(self):
        # Create random feature data
        rng = np.random.default_rng(seed=123)
        x = rng.normal(loc=1.5, scale=3.0, size=(200, 5))
        y = (x[:, 0] + x[:, 4] - x[:, 2] > 0).astype(int)

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        clf = LogisticRegression(random_state=42)
        clf.fit(x_scaled, y)

        w_raw, b_raw = fold_scaler_into_weights(scaler, clf)

        # Mathematical verification
        max_err = verify_weight_folding(scaler, clf, w_raw, b_raw, num_test_samples=500)
        self.assertLess(max_err, 1e-10)

    def test_synthetic_feature_dataset(self):
        df = generate_synthetic_feature_dataset(num_samples=100)
        self.assertEqual(len(df), 100)
        expected_cols = {"l_bar", "l_min", "var_l", "H_bar", "R_entity", "Faithful", "Label"}
        self.assertTrue(expected_cols.issubset(set(df.columns)))
        self.assertTrue(np.all(np.isfinite(df[["l_bar", "l_min", "var_l", "H_bar", "R_entity"]].values)))

    def test_run_cross_validation_10_fold(self):
        df = generate_synthetic_feature_dataset(num_samples=200)
        cv_metrics, scaler, clf = run_cross_validation(df, n_splits=5, random_state=42)

        self.assertEqual(len(cv_metrics.fold_aucs), 5)
        self.assertGreaterEqual(cv_metrics.roc_auc_mean, 0.85)
        self.assertGreaterEqual(cv_metrics.f1_mean, 0.80)
        self.assertTrue(0.0 <= cv_metrics.optimal_threshold <= 1.0)

    def test_calibrate_and_save_roundtrip(self):
        df = generate_synthetic_feature_dataset(num_samples=150)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test_weights.json"
            weights = calibrate_and_fold(df, n_splits=3, output_path=out_path)

            self.assertTrue(out_path.exists())

            # Load into detector
            detector = CompositeHallucinationDetector(weights_path=out_path)
            self.assertEqual(len(detector.weights), 5)
            self.assertAlmostEqual(detector.threshold, weights.optimal_threshold, places=4)

            # Test evaluation with loaded detector
            res = detector.evaluate_vector([-0.18, -0.65, 0.08, 0.12, 1.0])
            self.assertFalse(res.is_hallucination)
            self.assertGreater(res.faithfulness_score, 0.50)


if __name__ == "__main__":
    unittest.main()
