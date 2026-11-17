"""
tests/test_intrinsic.py
-----------------------
Unit tests for IntrinsicMetricsExtractor and OnlineIntrinsicTracker.
Verifies numerical stability, Softmax exponentiation, and Welford variance tracking.
"""

from __future__ import annotations

import math
import unittest
import numpy as np

from rag_guard.intrinsic import (
    IntrinsicMetrics,
    IntrinsicMetricsExtractor,
    OnlineIntrinsicTracker,
)


class TestIntrinsicMetricsExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = IntrinsicMetricsExtractor(top_k=5)

    def test_empty_or_none_logprobs(self):
        self.assertEqual(self.extractor.compute_entropy_from_logprobs({}), 0.0)
        self.assertEqual(self.extractor.compute_entropy_from_logprobs([]), 0.0)
        self.assertEqual(self.extractor.compute_entropy_from_logprobs(None), 0.0)

    def test_single_token_entropy(self):
        # A deterministic single token distribution has 0 entropy
        self.assertEqual(self.extractor.compute_entropy_from_logprobs({"token": -0.05}), 0.0)
        self.assertEqual(self.extractor.compute_entropy_from_logprobs([-0.05]), 0.0)

    def test_uniform_distribution_entropy(self):
        # For top-2 equal probabilities, H = - (0.5 * log2(0.5) * 2) = 1.0 bit
        top_2_equal = {"a": -0.693147, "b": -0.693147}
        entropy = self.extractor.compute_entropy_from_logprobs(top_2_equal)
        self.assertAlmostEqual(entropy, 1.0, places=2)

    def test_numerical_stability_extreme_logprobs(self):
        # Extreme negative logprobs that could underflow if max(z) is not subtracted
        extreme_logprobs = {"a": -1000.0, "b": -1001.0}
        entropy = self.extractor.compute_entropy_from_logprobs(extreme_logprobs)
        self.assertTrue(math.isfinite(entropy))
        self.assertGreaterEqual(entropy, 0.0)

    def test_top_k_limiting(self):
        # Even if 10 tokens are supplied, only top-k (5) are considered
        many_tokens = {f"t_{i}": float(-i) for i in range(10)}
        entropy = self.extractor.compute_entropy_from_logprobs(many_tokens)
        self.assertTrue(math.isfinite(entropy))
        self.assertGreater(entropy, 0.0)


class TestOnlineIntrinsicTracker(unittest.TestCase):
    def test_welford_variance_matches_numpy(self):
        tracker = OnlineIntrinsicTracker()
        sample_logprobs = [-0.12, -0.45, -0.08, -1.89, -0.25, -0.05]
        sample_entropies = [0.15, 0.42, 0.05, 1.95, 0.35, 0.10]

        for lp, en in zip(sample_logprobs, sample_entropies):
            tracker.update(lp, en)

        np_mean = float(np.mean(sample_logprobs))
        np_var = float(np.var(sample_logprobs))
        np_min = float(np.min(sample_logprobs))
        np_mean_h = float(np.mean(sample_entropies))

        self.assertAlmostEqual(tracker.l_bar, np_mean, places=6)
        self.assertAlmostEqual(tracker.var_l, np_var, places=6)
        self.assertAlmostEqual(tracker.l_min, np_min, places=6)
        self.assertAlmostEqual(tracker.H_bar, np_mean_h, places=6)
        self.assertEqual(tracker.count, len(sample_logprobs))

    def test_empty_tracker_returns_zeros(self):
        tracker = OnlineIntrinsicTracker()
        self.assertEqual(tracker.l_bar, 0.0)
        self.assertEqual(tracker.l_min, 0.0)
        self.assertEqual(tracker.var_l, 0.0)
        self.assertEqual(tracker.H_bar, 0.0)
        self.assertEqual(tracker.count, 0)

    def test_snapshot_metrics(self):
        tracker = OnlineIntrinsicTracker()
        tracker.update(-0.5, 1.2)
        metrics = tracker.get_metrics(generated_text="test answer")
        self.assertIsInstance(metrics, IntrinsicMetrics)
        self.assertEqual(metrics.num_tokens, 1)
        self.assertEqual(metrics.generated_text, "test answer")
        self.assertEqual(len(metrics.to_vector()), 4)


if __name__ == "__main__":
    unittest.main()
