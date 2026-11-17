"""
tests/test_stream.py
--------------------
Unit tests and latency micro-benchmarks for RAGGuardStream.
Verifies context pre-caching, clause boundary buffering, early stopping,
generator wrapping, and strict sub-20ms total latency budget.
"""

from __future__ import annotations

import time
import unittest

from rag_guard.stream import (
    RAGGuardStream,
    StreamFinalResult,
    StreamStepResult,
)


class TestRAGGuardStream(unittest.TestCase):
    def setUp(self):
        self.context = (
            "NVIDIA reported record quarterly revenue of $22.1B in Q4 FY24, "
            "up 265% from a year ago. Data Center revenue reached $18.4B."
        )

    def test_context_precached_on_init(self):
        stream = RAGGuardStream(self.context)
        assert len(stream.context_entities) > 0
        # Context entities should be immediately available
        assert any("22.1" in e or "18.4" in e or "265" in e for e in stream.context_entities)

    def test_clause_boundary_buffering(self):
        stream = RAGGuardStream(self.context)

        # Tokens without clause delimiter
        step1 = stream.step("NVIDIA", logprob=-0.10, top_logprobs={"NVIDIA": -0.10})
        self.assertFalse(step1.is_clause_boundary)
        self.assertEqual(step1.num_tokens, 1)

        step2 = stream.step(" announced", logprob=-0.15)
        self.assertFalse(step2.is_clause_boundary)
        self.assertEqual(step2.num_tokens, 2)

        # Token with a period delimiter triggers clause boundary
        step3 = stream.step(" $22.1B revenue.", logprob=-0.05)
        self.assertTrue(step3.is_clause_boundary)
        self.assertEqual(step3.num_tokens, 3)

        # Entity should now be in answer entities
        self.assertTrue(any("22.1" in e for e in stream.answer_entities))
        self.assertEqual(step3.R_entity, 1.0)
        self.assertGreater(step3.faithfulness_score, 0.50)

    def test_finish_flushes_unpunctuated_tail(self):
        stream = RAGGuardStream(self.context)
        stream.step("Data", logprob=-0.1)
        stream.step(" Center", logprob=-0.1)
        stream.step(" revenue was $18.4B", logprob=-0.1)
        # Note: no period at end of stream

        final = stream.finish()
        self.assertIsInstance(final, StreamFinalResult)
        self.assertEqual(final.num_tokens, 3)
        self.assertEqual(final.generated_text, "Data Center revenue was $18.4B")
        self.assertTrue(any("18.4" in e for e in final.answer_entities))
        self.assertEqual(final.faithfulness_result.features["R_entity"], 1.0)
        self.assertGreater(final.faithfulness_result.faithfulness_score, 0.50)

    def test_hallucination_detection_in_stream(self):
        # Answer makes conflicting/fabricated claims with high uncertainty
        stream = RAGGuardStream(self.context, early_stop_threshold=0.30)

        # Emit fabricated tokens with steep logprob drop and high entropy
        tokens = [
            ("NVIDIA", -0.1, {"NVIDIA": -0.1}),
            (" reported", -0.2, {" reported": -0.2}),
            (" only", -3.5, {" only": -3.5, " also": -3.6, " but": -3.7}),
            (" $999M", -4.2, {" $999M": -4.2, " $100M": -4.3, " $500M": -4.4}),
            (" revenue.", -3.8, {" revenue.": -3.8, " profit.": -3.9}),
        ]

        last_step = None
        for tok, lp, top_lp in tokens:
            last_step = stream.step(tok, lp, top_lp)

        self.assertIsNotNone(last_step)
        final = stream.finish()

        self.assertTrue(final.faithfulness_result.is_hallucination)
        self.assertLess(final.faithfulness_result.faithfulness_score, 0.50)
        self.assertGreater(final.faithfulness_result.hallucination_score, 0.50)

    def test_wrap_generator(self):
        stream = RAGGuardStream(self.context)
        token_stream = [
            ("NVIDIA", -0.1, None),
            (" generated", -0.1, None),
            (" $22.1B.", -0.05, None),
        ]

        steps = list(stream.wrap_generator(token_stream))
        self.assertEqual(len(steps), 4)  # 3 intermediate steps + 1 final sentinel

        # Last item must contain final result
        _, final_res = steps[-1]
        self.assertIsNotNone(final_res)
        self.assertEqual(final_res.num_tokens, 3)

    def test_streaming_latency_budget(self):
        """Micro-benchmark verifying latency remains well within budget (< 20 ms)."""
        stream = RAGGuardStream(self.context)

        # Simulate 100 generated tokens across 4 sentences
        simulated_tokens = []
        for i in range(100):
            is_end = (i % 25 == 24)
            tok = f" word{i}." if is_end else f" word{i}"
            simulated_tokens.append((tok, -0.15))

        t_start = time.perf_counter()
        for tok, lp in simulated_tokens:
            stream.step(tok, lp)
        final = stream.finish()
        total_stream_overhead_ms = (time.perf_counter() - t_start) * 1000.0

        # Assert total overhead for 100 tokens with 4 clause NER boundary passes is < 20 ms
        self.assertLess(
            total_stream_overhead_ms,
            20.0,
            f"Streaming overhead {total_stream_overhead_ms:.2f} ms exceeded 20 ms budget!",
        )


if __name__ == "__main__":
    unittest.main()
