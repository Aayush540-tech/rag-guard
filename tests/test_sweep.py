"""
tests/test_sweep.py
-------------------
Unit tests for threshold sensitivity sweep across Evident Conflict and Baseless Extrapolation splits.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag_guard.sweep import (
    DEFAULT_SWEEP_THRESHOLDS,
    SweepSummary,
    ThresholdResult,
    evaluate_split_at_threshold,
    generate_hallucination_split_dataset,
    run_threshold_sweep,
)
from rag_guard.detector import CompositeHallucinationDetector


class TestThresholdSweep(unittest.TestCase):
    def setUp(self):
        self.detector = CompositeHallucinationDetector()
        self.df = generate_hallucination_split_dataset(num_samples=60, random_state=42)

    def test_split_dataset_structure(self):
        self.assertEqual(len(self.df), 60)
        self.assertTrue({"split", "Faithful", "Label", "context", "response"}.issubset(set(self.df.columns)))
        splits = set(self.df["split"].unique())
        self.assertEqual(splits, {"grounded", "evident_conflict", "baseless_extrapolation"})

    def test_evaluate_split_single_threshold(self):
        res = evaluate_split_at_threshold(self.df, threshold=85.0, detector=self.detector, split_name="overall")
        self.assertIsInstance(res, ThresholdResult)
        self.assertEqual(res.threshold, 85.0)
        self.assertEqual(res.split_name, "overall")
        self.assertEqual(res.num_samples, 60)
        self.assertTrue(0.0 <= res.r_entity_auc <= 1.0)
        self.assertTrue(0.0 <= res.composite_auc <= 1.0)
        self.assertTrue(0.0 <= res.f1 <= 1.0)

    def test_run_threshold_sweep_and_save_json(self):
        summary = run_threshold_sweep(self.df, thresholds=[80.0, 85.0, 90.0], detector=self.detector)
        self.assertIsInstance(summary, SweepSummary)
        self.assertIn("overall", summary.results_by_split)
        self.assertIn("evident_conflict", summary.results_by_split)
        self.assertIn("baseless_extrapolation", summary.results_by_split)
        self.assertIn(summary.optimal_threshold_overall, [80.0, 85.0, 90.0])

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "sweep_results.json"
            summary.save_json(json_path)
            self.assertTrue(json_path.exists())


if __name__ == "__main__":
    unittest.main()
