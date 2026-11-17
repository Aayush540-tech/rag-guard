"""
tests/test_detector.py
----------------------
Unit tests for CompositeHallucinationDetector and FaithfulnessResult.
Verifies algebraic folded scoring, numerical clipping, and zero-dependency inference.
"""

from __future__ import annotations

import unittest
import numpy as np

from rag_guard.detector import (
    CompositeHallucinationDetector,
    FaithfulnessResult,
)
from rag_guard.intrinsic import IntrinsicMetrics


class TestCompositeHallucinationDetector(unittest.TestCase):
    def setUp(self):
        self.detector = CompositeHallucinationDetector()

    def test_default_weights_loaded_correctly(self):
        self.assertEqual(self.detector.weights.shape, (5,))
        self.assertEqual(self.detector.threshold, 0.50)
        self.assertIsInstance(self.detector.bias, float)

    def test_dimension_validation(self):
        with self.assertRaises(ValueError):
            self.detector.evaluate_vector([1.0, 2.0, 3.0])

    def test_grounded_vs_hallucinated_scoring(self):
        # Grounded: high logprob, low variance, low entropy, full entity match
        grounded_x = [-0.15, -0.40, 0.05, 0.10, 1.0]
        res_grounded = self.detector.evaluate_vector(grounded_x)

        self.assertIsInstance(res_grounded, FaithfulnessResult)
        self.assertGreater(res_grounded.faithfulness_score, 0.50)
        self.assertFalse(res_grounded.is_hallucination)
        self.assertLess(res_grounded.hallucination_score, 0.50)

        # Hallucinated: negative logprob collapse, high variance, entropy spike, ungrounded entities
        hallucinated_x = [-2.50, -4.50, 2.80, 2.85, 0.0]
        res_hallucinated = self.detector.evaluate_vector(hallucinated_x)

        self.assertLess(res_hallucinated.faithfulness_score, 0.50)
        self.assertTrue(res_hallucinated.is_hallucination)
        self.assertGreater(res_hallucinated.hallucination_score, 0.50)

    def test_nan_and_inf_handling(self):
        # Even if NaNs or Infs leak in from upstream errors, detector should not crash
        corrupted_x = [np.nan, -1.0, np.inf, 0.5, 0.8]
        res = self.detector.evaluate_vector(corrupted_x)
        self.assertTrue(0.0 <= res.faithfulness_score <= 1.0)

    def test_convenience_evaluate_method(self):
        intrinsic = IntrinsicMetrics(
            l_bar=-0.20,
            l_min=-0.50,
            var_l=0.10,
            H_bar=0.15,
            num_tokens=10,
            generated_text="grounded sample",
        )
        res = self.detector.evaluate(intrinsic, r_entity=1.0)
        self.assertGreater(res.faithfulness_score, 0.50)
        self.assertFalse(res.is_hallucination)

    def test_batch_evaluate_vectorized(self):
        matrix = np.array([
            [-0.15, -0.40, 0.05, 0.10, 1.0],  # Grounded
            [-2.50, -4.50, 2.80, 2.85, 0.0],  # Hallucinated
        ])
        scores = self.detector.batch_evaluate(matrix)
        self.assertEqual(scores.shape, (2,))
        self.assertGreater(scores[0], 0.50)
        self.assertLess(scores[1], 0.50)


if __name__ == "__main__":
    unittest.main()
