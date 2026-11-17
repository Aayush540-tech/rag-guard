"""
rag_guard.stream
----------------
Real-time streaming guardrail engine for auto-regressive decoding.
Pre-caches context entities once upon stream initialization, updates intrinsic
uncertainty in O(1) via Welford tracking, buffers token text, and restricts
spaCy NER extraction to sentence/clause boundaries.
Preserves total latency budget Delta t <= 20 ms.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np

from rag_guard.config import (
    DEFAULT_FUZZY_THRESHOLD,
    DEFAULT_TOP_K_LOGPROBS,
    STREAM_CLAUSE_DELIMITERS,
)
from rag_guard.detector import (
    CompositeHallucinationDetector,
    FaithfulnessResult,
)
from rag_guard.grounding import EntityGroundingEvaluator
from rag_guard.intrinsic import (
    IntrinsicMetrics,
    IntrinsicMetricsExtractor,
    OnlineIntrinsicTracker,
)

logger = logging.getLogger("rag_guard.stream")


@dataclass(frozen=True)
class StreamStepResult:
    """Real-time verification signal emitted per streaming token."""

    token: str
    num_tokens: int
    faithfulness_score: float  # Current running S_faith in [0.0, 1.0]
    hallucination_score: float  # Current running 1.0 - S_faith
    is_clause_boundary: bool  # True if a clause/sentence boundary was just processed
    is_hallucination: bool  # True if faithfulness_score < detector threshold
    early_stop_recommended: bool  # True if S_faith drops below early_stop_threshold
    l_bar: float  # Running mean logprob
    l_min: float  # Running min logprob
    var_l: float  # Running logprob variance
    H_bar: float  # Running mean top-k Shannon entropy
    R_entity: float  # Running entity grounding ratio

    def to_dict(self) -> Dict[str, Any]:
        """Convert step result to dictionary."""
        return {
            "token": self.token,
            "num_tokens": self.num_tokens,
            "faithfulness_score": round(self.faithfulness_score, 4),
            "hallucination_score": round(self.hallucination_score, 4),
            "is_clause_boundary": self.is_clause_boundary,
            "is_hallucination": self.is_hallucination,
            "early_stop_recommended": self.early_stop_recommended,
            "l_bar": round(self.l_bar, 4),
            "l_min": round(self.l_min, 4),
            "var_l": round(self.var_l, 4),
            "H_bar": round(self.H_bar, 4),
            "R_entity": round(self.R_entity, 4),
        }


@dataclass(frozen=True)
class StreamFinalResult:
    """Final guardrail evaluation emitted upon stream completion."""

    generated_text: str
    faithfulness_result: FaithfulnessResult
    num_tokens: int
    num_clauses: int
    answer_entities: Set[str]
    context_entities: Set[str]
    elapsed_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert final result to dictionary."""
        return {
            "generated_text": self.generated_text,
            "faithfulness_score": round(self.faithfulness_result.faithfulness_score, 4),
            "hallucination_score": round(self.faithfulness_result.hallucination_score, 4),
            "is_hallucination": self.faithfulness_result.is_hallucination,
            "num_tokens": self.num_tokens,
            "num_clauses": self.num_clauses,
            "answer_entities": sorted(list(self.answer_entities)),
            "context_entities": sorted(list(self.context_entities)),
            "elapsed_ms": round(self.elapsed_ms, 3),
            "features": self.faithfulness_result.to_dict()["features"],
        }


class RAGGuardStream:
    """
    Stateful streaming guardrail wrapper for real-time RAG inference.

    Features:
      - Once-per-stream context entity caching: E_C = extract(Context).
      - O(1) online logprob stats and top-k Shannon entropy via Welford tracking.
      - Clause-buffering: Triggers NER only when punctuation delimiters (. ! ? \\n ;) are encountered.
      - Continuous S_faith scoring for early stopping or alerting.
    """

    def __init__(
        self,
        context: str,
        detector: Optional[CompositeHallucinationDetector] = None,
        grounder: Optional[EntityGroundingEvaluator] = None,
        extractor: Optional[IntrinsicMetricsExtractor] = None,
        clause_delimiters: FrozenSet[str] = STREAM_CLAUSE_DELIMITERS,
        early_stop_threshold: Optional[float] = None,
    ) -> None:
        """
        Initialize streaming guardrail and pre-cache context entities.

        Parameters
        ----------
        context : str
            Retrieved context text C against which generation is grounded.
        detector : CompositeHallucinationDetector, optional
            Calibrated zero-dependency detector. Defaults to global weights.json.
        grounder : EntityGroundingEvaluator, optional
            Entity extractor and RapidFuzz matcher.
        extractor : IntrinsicMetricsExtractor, optional
            Top-k Shannon entropy calculation helper.
        clause_delimiters : FrozenSet[str], optional
            Punctuation marks that trigger boundary NER.
        early_stop_threshold : float, optional
            If S_faith drops below this threshold at a clause boundary,
            early_stop_recommended flag is asserted.
        """
        self.context = context or ""
        self.detector = detector or CompositeHallucinationDetector()
        self.grounder = grounder or EntityGroundingEvaluator()
        self.extractor = extractor or IntrinsicMetricsExtractor()
        self.clause_delimiters = clause_delimiters
        self.early_stop_threshold = early_stop_threshold

        # Step 1: Pre-cache context entities E_C once
        t0 = time.perf_counter()
        self._cached_context_entities: Set[str] = self.grounder.cache_context_entities(self.context)
        self._context_init_ms: float = (time.perf_counter() - t0) * 1000.0

        # Step 2: Initialize online streaming accumulators
        self._tracker = OnlineIntrinsicTracker()
        self._clause_buffer: List[str] = []
        self._full_text_chunks: List[str] = []
        self._accumulated_answer_entities: Set[str] = set()
        self._running_r_entity: float = 1.0  # Vacuously grounded before any entities claimed
        self._num_clauses: int = 0
        self._is_finished: bool = False
        self._start_time: float = time.perf_counter()

    @property
    def context_entities(self) -> Set[str]:
        """Pre-cached context entities E_C."""
        return set(self._cached_context_entities)

    @property
    def answer_entities(self) -> Set[str]:
        """Entities E_A extracted so far from the generated answer."""
        return set(self._accumulated_answer_entities)

    @property
    def current_text(self) -> str:
        """Full text generated so far."""
        return "".join(self._full_text_chunks)

    def step(
        self,
        token: str,
        logprob: float,
        top_logprobs: Optional[Union[Dict[Any, Any], Sequence[float], np.ndarray]] = None,
    ) -> StreamStepResult:
        """
        Process a single streaming decoding step.

        Parameters
        ----------
        token : str
            The newly emitted token string (e.g. " 450", " revenue").
        logprob : float
            Log-probability of the chosen token log P(t_i | Q, C, t_<i).
        top_logprobs : dict | sequence | ndarray, optional
            Candidate token logprobs at this step for top-k Shannon entropy.

        Returns
        -------
        StreamStepResult
        """
        if self._is_finished:
            raise RuntimeError("Cannot call step() on a stream that has already been finished.")

        # 1. Compute entropy in O(1)
        h_t = 0.0
        if top_logprobs is not None:
            h_t = self.extractor.compute_entropy_from_logprobs(top_logprobs)

        # 2. Update Welford tracker in O(1)
        self._tracker.update(float(logprob), float(h_t))

        # 3. Buffer text
        token_str = str(token)
        self._clause_buffer.append(token_str)
        self._full_text_chunks.append(token_str)

        # 4. Check for sentence/clause boundaries (. ! ? \n ;)
        is_boundary = any(delim in token_str for delim in self.clause_delimiters)

        if is_boundary:
            self._flush_clause_buffer()

        # 5. Evaluate current composite faithfulness
        vector = [
            self._tracker.l_bar,
            self._tracker.l_min,
            self._tracker.var_l,
            self._tracker.H_bar,
            self._running_r_entity,
        ]
        res = self.detector.evaluate_vector(vector)

        # 6. Check early stopping condition
        early_stop = False
        if self.early_stop_threshold is not None:
            early_stop = res.faithfulness_score < self.early_stop_threshold

        return StreamStepResult(
            token=token_str,
            num_tokens=self._tracker.count,
            faithfulness_score=res.faithfulness_score,
            hallucination_score=res.hallucination_score,
            is_clause_boundary=is_boundary,
            is_hallucination=res.is_hallucination,
            early_stop_recommended=early_stop,
            l_bar=self._tracker.l_bar,
            l_min=self._tracker.l_min,
            var_l=self._tracker.var_l,
            H_bar=self._tracker.H_bar,
            R_entity=self._running_r_entity,
        )

    def _flush_clause_buffer(self) -> None:
        """Extract entities from the buffered clause and update running R_entity."""
        if not self._clause_buffer:
            return

        clause_text = "".join(self._clause_buffer)
        self._clause_buffer.clear()
        self._num_clauses += 1

        # Extract entities from this specific clause (fast, clause is short)
        new_entities = self.grounder.extract_entities(clause_text)
        if new_entities:
            self._accumulated_answer_entities.update(new_entities)
            # Recompute grounding ratio with newly discovered entities
            self._running_r_entity = self.grounder.compute_grounding_ratio_cached(
                self._accumulated_answer_entities,
                self._cached_context_entities,
            )

    def finish(self) -> StreamFinalResult:
        """
        Close the stream, flush any residual text buffer, and return the final evaluation.

        Returns
        -------
        StreamFinalResult
        """
        if not self._is_finished:
            # Flush any trailing tokens that didn't end with a delimiter
            if self._clause_buffer:
                self._flush_clause_buffer()
            self._is_finished = True

        elapsed_ms = (time.perf_counter() - self._start_time) * 1000.0
        final_text = self.current_text

        # Final intrinsic metrics snapshot
        intrinsic = self._tracker.get_metrics(generated_text=final_text)

        # Final faithfulness result
        faith_res = self.detector.evaluate(intrinsic, self._running_r_entity)

        return StreamFinalResult(
            generated_text=final_text,
            faithfulness_result=faith_res,
            num_tokens=self._tracker.count,
            num_clauses=self._num_clauses,
            answer_entities=set(self._accumulated_answer_entities),
            context_entities=set(self._cached_context_entities),
            elapsed_ms=elapsed_ms,
        )

    def wrap_generator(
        self,
        token_stream: Iterator[Tuple[str, float, Optional[Any]]],
        abort_on_early_stop: bool = False,
    ) -> Iterator[Tuple[StreamStepResult, Optional[StreamFinalResult]]]:
        """
        Convenience generator wrapper over an iterator yielding (token, logprob, top_logprobs).

        Yields (step_result, None) on intermediate steps,
        and (step_result, final_result) on the final step.
        """
        for token, logprob, top_lp in token_stream:
            step_res = self.step(token, logprob, top_lp)
            if abort_on_early_stop and step_res.early_stop_recommended:
                final_res = self.finish()
                yield step_res, final_res
                return
            yield step_res, None

        final_res = self.finish()
        # Yield empty sentinel step with final_res if stream completed
        sentinel_step = StreamStepResult(
            token="",
            num_tokens=self._tracker.count,
            faithfulness_score=final_res.faithfulness_result.faithfulness_score,
            hallucination_score=final_res.faithfulness_result.hallucination_score,
            is_clause_boundary=True,
            is_hallucination=final_res.faithfulness_result.is_hallucination,
            early_stop_recommended=False,
            l_bar=self._tracker.l_bar,
            l_min=self._tracker.l_min,
            var_l=self._tracker.var_l,
            H_bar=self._tracker.H_bar,
            R_entity=self._running_r_entity,
        )
        yield sentinel_step, final_res
