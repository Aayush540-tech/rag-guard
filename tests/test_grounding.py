"""
tests/test_grounding.py
-----------------------
Unit tests for EntityGroundingEvaluator.
Verifies entity extraction, context pre-caching, and RapidFuzz threshold matching.
"""

from __future__ import annotations

import unittest
from rag_guard.grounding import EntityGroundingEvaluator


class TestEntityGroundingEvaluator(unittest.TestCase):
    def setUp(self):
        self.grounder = EntityGroundingEvaluator(fuzzy_threshold=85.0)

    def test_empty_answer_is_vacuously_grounded(self):
        ctx = "Apple released the iPhone 15 with 8GB RAM in September 2023."
        # If the answer claims no named entities, R_entity is defined as 1.0
        self.assertEqual(self.grounder.compute_grounding_ratio("", ctx), 1.0)
        self.assertEqual(self.grounder.compute_grounding_ratio("The device is fast and sleek.", ctx), 1.0)

    def test_empty_context_with_entities_is_ungrounded(self):
        ans = "The company reported $10B revenue in 2024."
        self.assertEqual(self.grounder.compute_grounding_ratio(ans, ""), 0.0)

    def test_numeric_grounding_exact_and_fuzzy(self):
        ctx = "The project budget is $45,000,000 and timeline is 18 months."
        grounded_ans = "Budget allocated is $45,000,000 across 18 months."
        ungrounded_ans = "Budget allocated is $12B across 36 months."

        r_grounded = self.grounder.compute_grounding_ratio(grounded_ans, ctx)
        r_ungrounded = self.grounder.compute_grounding_ratio(ungrounded_ans, ctx)

        self.assertEqual(r_grounded, 1.0)
        self.assertEqual(r_ungrounded, 0.0)

    def test_pre_cached_context_entities(self):
        ctx = "Dr. Vishal Gupta oversees the Big Data Analytics lab at NSUT with 30 students."
        cached_ec = self.grounder.cache_context_entities(ctx)
        self.assertIsInstance(cached_ec, set)
        self.assertGreater(len(cached_ec), 0)

        # Answer with matching entities
        ans_entities = {"30"}
        r = self.grounder.compute_grounding_ratio_cached(ans_entities, cached_ec)
        self.assertEqual(r, 1.0)

    def test_clean_entity_whitespace(self):
        raw = "  New   York  City  "
        self.assertEqual(EntityGroundingEvaluator.clean_entity(raw), "new york city")
        self.assertEqual(EntityGroundingEvaluator.clean_entity(" "), "")
        self.assertEqual(EntityGroundingEvaluator.clean_entity("a"), "")


if __name__ == "__main__":
    unittest.main()
