"""
rag_guard.vllm_hook
-------------------
Asynchronous vLLM integration middleware for real-time hallucination interception.
Hooks into vLLM's AsyncLLMEngine to stream verified tokens with zero pipeline disruptions,
compute running S_faith, and optionally abort generation when hallucinations are detected.
Includes transparent MockAsyncLLMEngine for CPU / local development environments.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from rag_guard.config import DEFAULT_TOP_K_LOGPROBS
from rag_guard.detector import (
    CompositeHallucinationDetector,
    FaithfulnessResult,
)
from rag_guard.stream import (
    RAGGuardStream,
    StreamFinalResult,
    StreamStepResult,
)

logger = logging.getLogger("rag_guard.vllm_hook")


# ---------------------------------------------------------------------------
# Lightweight structures simulating vLLM streaming RequestOutput
# ---------------------------------------------------------------------------
@dataclass
class StreamLogprobEntry:
    """Mirrors vLLM's Logprob namedtuple / class."""
    logprob: float
    rank: int = 0
    decoded_token: str = ""


@dataclass
class StreamCompletionOutput:
    """Mirrors vLLM's CompletionOutput during streaming."""
    index: int = 0
    text: str = ""
    token_ids: List[int] = field(default_factory=list)
    cumulative_logprob: float = 0.0
    logprobs: Optional[List[Dict[Any, StreamLogprobEntry]]] = None
    finish_reason: Optional[str] = None


@dataclass
class StreamRequestOutput:
    """Mirrors vLLM's RequestOutput emitted by AsyncLLMEngine.generate()."""
    request_id: str
    prompt: str
    outputs: List[StreamCompletionOutput]
    finished: bool = False


# ---------------------------------------------------------------------------
# Transparent Mock AsyncLLMEngine for development and headless testing
# ---------------------------------------------------------------------------
class MockAsyncLLMEngine:
    """
    Simulates vLLM's AsyncLLMEngine for local development and unit tests.
    Generates realistic streaming token outputs with logprob distributions.
    """

    def __init__(self, token_delay_ms: float = 2.0) -> None:
        self.token_delay_s = token_delay_ms / 1000.0
        self.aborted_requests: set[str] = set()

    def abort(self, request_id: str) -> None:
        """Abort an active in-flight request."""
        logger.info("MockAsyncLLMEngine: Aborted request '%s'", request_id)
        self.aborted_requests.add(request_id)

    async def generate(
        self,
        prompt: str,
        sampling_params: Optional[Any] = None,
        request_id: Optional[str] = None,
        simulate_hallucination: bool = False,
    ) -> AsyncIterator[StreamRequestOutput]:
        """
        Asynchronously yield streaming StreamRequestOutput objects token by token.
        """
        req_id = request_id or str(uuid.uuid4())
        rng = np.random.default_rng(seed=hash(req_id) % (2**32))

        if not simulate_hallucination:
            tokens = [
                ("Apple", -0.10, 5.0),
                (" recorded", -0.15, 4.5),
                (" $119.58B", -0.05, 6.0),
                (" revenue", -0.12, 5.2),
                (" in", -0.08, 5.5),
                (" Q1", -0.09, 4.8),
                (" 2024.", -0.04, 6.2),
            ]
        else:
            tokens = [
                ("Apple", -0.10, 4.5),
                (" reported", -0.20, 3.8),
                (" a", -1.80, 0.25),
                (" record", -2.40, 0.20),
                (" $999.00B", -3.95, 0.15),
                (" revenue", -3.50, 0.18),
                (" in", -2.80, 0.22),
                (" 2035.", -4.10, 0.12),
            ]

        accumulated_text = ""
        logprobs_history: List[Dict[Any, StreamLogprobEntry]] = []

        for idx, (token_str, mean_lp, gap_scale) in enumerate(tokens):
            if req_id in self.aborted_requests:
                logger.info("MockAsyncLLMEngine: Request '%s' terminated by early abort.", req_id)
                break

            if self.token_delay_s > 0:
                await asyncio.sleep(self.token_delay_s)

            accumulated_text += token_str

            # Construct top-k candidate logprobs dictionary
            top_k_dict: Dict[str, StreamLogprobEntry] = {}
            top_k_dict[token_str] = StreamLogprobEntry(logprob=float(mean_lp), rank=0, decoded_token=token_str)
            for k in range(1, DEFAULT_TOP_K_LOGPROBS):
                # Larger gap -> lower Shannon entropy (grounded)
                # Smaller gap -> higher Shannon entropy (hallucinated)
                alt_lp = float(mean_lp - (gap_scale * k + rng.exponential(scale=0.1)))
                top_k_dict[f"<alt_{k}>"] = StreamLogprobEntry(logprob=alt_lp, rank=k, decoded_token=f"<alt_{k}>")

            logprobs_history.append(top_k_dict)

            is_last = (idx == len(tokens) - 1)
            completion = StreamCompletionOutput(
                index=0,
                text=accumulated_text,
                token_ids=list(range(idx + 1)),
                cumulative_logprob=float(mean_lp * (idx + 1)),
                logprobs=list(logprobs_history),
                finish_reason="stop" if is_last else None,
            )

            yield StreamRequestOutput(
                request_id=req_id,
                prompt=prompt,
                outputs=[completion],
                finished=is_last,
            )


# ---------------------------------------------------------------------------
# High-Level Async vLLM Guardrail Middleware
# ---------------------------------------------------------------------------
class AsyncRAGGuard:
    """
    Asynchronous guardrail middleware interfacing with vLLM's AsyncLLMEngine.

    Wraps the streaming generation loop to:
      1. Intercept newly emitted tokens and candidate logprobs in real time.
      2. Run RAGGuardStream O(1) tracking and clause-boundary entity verification.
      3. Emit StreamStepResult along with token chunks for real-time UI/API rendering.
      4. Optionally abort vLLM inference early if S_faith drops below a critical threshold.
    """

    def __init__(
        self,
        engine: Optional[Any] = None,
        detector: Optional[CompositeHallucinationDetector] = None,
        early_stop_threshold: Optional[float] = None,
        abort_on_hallucination: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        engine : Any, optional
            vLLM AsyncLLMEngine instance (or MockAsyncLLMEngine). Defaults to MockAsyncLLMEngine.
        detector : CompositeHallucinationDetector, optional
            Calibrated zero-dependency detector.
        early_stop_threshold : float, optional
            If S_faith drops below this threshold, early_stop_recommended is asserted.
        abort_on_hallucination : bool, optional
            If True, calls engine.abort(request_id) when early_stop_recommended is asserted,
            immediately terminating upstream GPU generation.
        """
        self.engine = engine or MockAsyncLLMEngine()
        self.detector = detector or CompositeHallucinationDetector()
        self.early_stop_threshold = early_stop_threshold
        self.abort_on_hallucination = abort_on_hallucination

    async def stream_generate(
        self,
        prompt: str,
        context: str,
        sampling_params: Optional[Any] = None,
        request_id: Optional[str] = None,
        **engine_kwargs: Any,
    ) -> AsyncIterator[Tuple[str, StreamStepResult, Optional[StreamFinalResult]]]:
        """
        Generate tokens with real-time guardrail verification.

        Yields
        ------
        Tuple[token_delta, step_result, final_result]
          - Intermediate tokens yield: (token_delta, step_result, None)
          - Final completion yields: ("", sentinel_step, final_result)
        """
        req_id = request_id or str(uuid.uuid4())
        guard_stream = RAGGuardStream(
            context=context,
            detector=self.detector,
            early_stop_threshold=self.early_stop_threshold,
        )

        prev_text_len = 0
        step_result: Optional[StreamStepResult] = None
        was_aborted = False

        # Hook into vLLM engine's async stream
        results_generator = self.engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=req_id,
            **engine_kwargs,
        )

        async for request_output in results_generator:
            if not request_output.outputs:
                continue

            completion = request_output.outputs[0]
            current_text = completion.text or ""

            # Extract new token delta
            token_delta = current_text[prev_text_len:]
            if not token_delta:
                continue
            prev_text_len = len(current_text)

            # Extract latest token logprobs
            latest_lp_dict = None
            chosen_lp = 0.0
            if completion.logprobs:
                latest_lp_dict = completion.logprobs[-1]
                if isinstance(latest_lp_dict, dict):
                    # Selected token is the entry with highest logprob
                    best_entry = max(
                        latest_lp_dict.values(),
                        key=lambda e: getattr(e, "logprob", e),
                    )
                    chosen_lp = float(getattr(best_entry, "logprob", best_entry))

            # Step the streaming guardrail in O(1)
            step_result = guard_stream.step(
                token=token_delta,
                logprob=chosen_lp,
                top_logprobs=latest_lp_dict,
            )

            # Check if early abort should be triggered
            if (
                self.abort_on_hallucination
                and step_result.early_stop_recommended
                and hasattr(self.engine, "abort")
            ):
                logger.warning(
                    "AsyncRAGGuard: Aborting request '%s' early due to hallucination alert (S_faith = %.4f).",
                    req_id,
                    step_result.faithfulness_score,
                )
                self.engine.abort(req_id)
                was_aborted = True
                yield token_delta, step_result, None
                break

            yield token_delta, step_result, None

        # Finalize guardrail stream
        final_result = guard_stream.finish()

        # Emit final sentinel step
        sentinel_step = StreamStepResult(
            token="",
            num_tokens=guard_stream._tracker.count,
            faithfulness_score=final_result.faithfulness_result.faithfulness_score,
            hallucination_score=final_result.faithfulness_result.hallucination_score,
            is_clause_boundary=True,
            is_hallucination=final_result.faithfulness_result.is_hallucination,
            early_stop_recommended=was_aborted,
            l_bar=guard_stream._tracker.l_bar,
            l_min=guard_stream._tracker.l_min,
            var_l=guard_stream._tracker.var_l,
            H_bar=guard_stream._tracker.H_bar,
            R_entity=guard_stream._running_r_entity,
        )

        yield "", sentinel_step, final_result
