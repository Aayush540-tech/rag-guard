"""
tests/test_vllm_hook.py
-----------------------
Unit tests for AsyncRAGGuard and MockAsyncLLMEngine.
Verifies asynchronous token interception, early abort, and final result generation.
"""

from __future__ import annotations

import asyncio
import unittest

from rag_guard.vllm_hook import (
    AsyncRAGGuard,
    MockAsyncLLMEngine,
    StreamFinalResult,
    StreamStepResult,
)


class TestAsyncVLLMHook(unittest.TestCase):
    def setUp(self):
        self.context = (
            "Apple Inc. reported quarterly revenue of $119.58 billion in Q1 2024, "
            "up 2 percent year-over-year. iPhone revenue was $69.7 billion."
        )
        self.prompt = f"Context:\n{self.context}\n\nQuestion: What was Apple revenue?\nAnswer:"

    def test_mock_engine_stream_generation(self):
        async def _run():
            engine = MockAsyncLLMEngine(token_delay_ms=0.0)
            outputs = []
            async for out in engine.generate("test prompt", simulate_hallucination=False):
                outputs.append(out)
            return outputs

        outputs = asyncio.run(_run())
        self.assertGreater(len(outputs), 0)
        self.assertTrue(outputs[-1].finished)

    def test_async_rag_guard_grounded_stream(self):
        async def _run():
            guard = AsyncRAGGuard(
                engine=MockAsyncLLMEngine(token_delay_ms=0.0),
                early_stop_threshold=0.30,
                abort_on_hallucination=False,
            )
            tokens = []
            steps = []
            final_res = None

            async for tok, step_res, final in guard.stream_generate(
                prompt=self.prompt,
                context=self.context,
                simulate_hallucination=False,
            ):
                if tok:
                    tokens.append(tok)
                    steps.append(step_res)
                if final:
                    final_res = final

            return tokens, steps, final_res

        tokens, steps, final_res = asyncio.run(_run())

        self.assertGreater(len(tokens), 0)
        self.assertGreater(len(steps), 0)
        self.assertIsInstance(final_res, StreamFinalResult)
        self.assertFalse(final_res.faithfulness_result.is_hallucination)
        self.assertGreater(final_res.faithfulness_result.faithfulness_score, 0.50)

    def test_async_rag_guard_hallucination_early_abort(self):
        async def _run():
            engine = MockAsyncLLMEngine(token_delay_ms=0.0)
            guard = AsyncRAGGuard(
                engine=engine,
                early_stop_threshold=0.40,
                abort_on_hallucination=True,
            )

            tokens = []
            aborted_early = False

            async for tok, step_res, final in guard.stream_generate(
                prompt=self.prompt,
                context=self.context,
                request_id="test-req-abort",
                simulate_hallucination=True,
            ):
                if tok:
                    tokens.append(tok)
                if step_res.early_stop_recommended:
                    aborted_early = True

            return tokens, aborted_early, "test-req-abort" in engine.aborted_requests

        tokens, aborted_early, was_aborted = asyncio.run(_run())

        self.assertTrue(aborted_early)
        self.assertTrue(was_aborted)


if __name__ == "__main__":
    unittest.main()
