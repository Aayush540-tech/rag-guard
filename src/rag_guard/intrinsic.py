"""
rag_guard.intrinsic
-------------------
Intrinsic uncertainty quantification and decoding-level entropy metrics.
Operates on raw log-probabilities with strict numerical stability.
Runtime dependencies: numpy only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from rag_guard.config import DEFAULT_TOP_K_LOGPROBS

logger = logging.getLogger("rag_guard.intrinsic")

# Numerical stability constant for log2 calculations
LOG2_EPSILON: float = 1e-12


@dataclass(frozen=True)
class IntrinsicMetrics:
    """Immutable container holding intrinsic uncertainty features."""

    l_bar: float  # Mean token log-probability
    l_min: float  # Minimum token log-probability
    var_l: float  # Log-probability population variance
    H_bar: float  # Mean top-k Shannon entropy in bits
    num_tokens: int = 0
    generated_text: str = ""

    def to_dict(self) -> Dict[str, Union[float, int, str]]:
        """Return metrics as a standard dictionary."""
        return {
            "l_bar": self.l_bar,
            "l_min": self.l_min,
            "var_l": self.var_l,
            "H_bar": self.H_bar,
            "num_tokens": self.num_tokens,
            "generated_text": self.generated_text,
        }

    def to_vector(self) -> List[float]:
        """Return 4-dimensional intrinsic feature vector [l_bar, l_min, var_l, H_bar]."""
        return [self.l_bar, self.l_min, self.var_l, self.H_bar]


class OnlineIntrinsicTracker:
    """
    O(1) incremental tracker for streaming token logprobs and predictive entropy.
    Uses Welford's algorithm for numerically stable running variance.
    """

    __slots__ = ("_count", "_mean_l", "_m2_l", "_min_l", "_mean_h")

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset internal accumulator state."""
        self._count: int = 0
        self._mean_l: float = 0.0
        self._m2_l: float = 0.0
        self._min_l: float = 0.0
        self._mean_h: float = 0.0

    def update(self, chosen_logprob: float, token_entropy: float) -> None:
        """
        Incorporate a new token's logprob and entropy in O(1) time and space.

        Parameters
        ----------
        chosen_logprob : float
            Log-probability log P(t_i | context, t_<i) of the emitted token.
        token_entropy : float
            Predictive Shannon entropy (bits) across candidate tokens at step i.
        """
        self._count += 1
        n = self._count

        # Running minimum
        if n == 1:
            self._min_l = chosen_logprob
        else:
            if chosen_logprob < self._min_l:
                self._min_l = chosen_logprob

        # Welford's running mean and M2 for log-probability
        delta = chosen_logprob - self._mean_l
        self._mean_l += delta / n
        delta2 = chosen_logprob - self._mean_l
        self._m2_l += delta * delta2

        # Running mean for predictive entropy
        self._mean_h += (token_entropy - self._mean_h) / n

    @property
    def count(self) -> int:
        return self._count

    @property
    def l_bar(self) -> float:
        return self._mean_l if self._count > 0 else 0.0

    @property
    def l_min(self) -> float:
        return self._min_l if self._count > 0 else 0.0

    @property
    def var_l(self) -> float:
        """Population variance of log-probabilities."""
        return (self._m2_l / self._count) if self._count > 0 else 0.0

    @property
    def H_bar(self) -> float:
        """Mean per-token predictive Shannon entropy in bits."""
        return self._mean_h if self._count > 0 else 0.0

    def get_metrics(self, generated_text: str = "") -> IntrinsicMetrics:
        """Return an immutable snapshot of current intrinsic metrics."""
        return IntrinsicMetrics(
            l_bar=self.l_bar,
            l_min=self.l_min,
            var_l=self.var_l,
            H_bar=self.H_bar,
            num_tokens=self._count,
            generated_text=generated_text,
        )


class IntrinsicMetricsExtractor:
    """
    Extracts intrinsic uncertainty signals from vLLM logprob structures
    and standalone logprob vectors.
    """

    def __init__(self, top_k: int = DEFAULT_TOP_K_LOGPROBS) -> None:
        self.top_k = top_k

    def compute_entropy_from_logprobs(
        self,
        top_logprobs: Union[Dict[Any, Any], Sequence[float], np.ndarray],
    ) -> float:
        """
        Compute top-k normalized Shannon entropy in bits with softmax stability.

        Formula:
          P(v_j) = exp(z_j - max(z)) / sum_m exp(z_m - max(z))
          H(t_i) = - sum_{j=1}^k P(v_j) * log2(P(v_j) + eps)

        Parameters
        ----------
        top_logprobs : dict, sequence, or ndarray
            Mapping of candidate tokens to logprob values (or objects with .logprob),
            or an array of logprob scalars.

        Returns
        -------
        float
            Shannon entropy in bits (>= 0.0). Returns 0.0 on invalid or empty input.
        """
        if not top_logprobs:
            return 0.0

        try:
            if isinstance(top_logprobs, dict):
                # Handles dict of objects with .logprob (e.g. vLLM Logprob) or dict of floats
                raw_values = [
                    getattr(val, "logprob", val)
                    for val in top_logprobs.values()
                ]
            else:
                raw_values = list(top_logprobs)

            # Limit to top-k candidates if more are present
            if len(raw_values) > self.top_k:
                raw_values.sort(reverse=True)
                raw_values = raw_values[: self.top_k]

            arr = np.asarray(raw_values, dtype=np.float64)
            if arr.size == 0 or np.all(np.isnan(arr)):
                return 0.0

            # Filter out any non-finite entries
            finite_mask = np.isfinite(arr)
            if not np.any(finite_mask):
                return 0.0
            arr = arr[finite_mask]

            if arr.size == 1:
                return 0.0

            # Numerically stable Softmax: subtract max value
            max_val = np.max(arr)
            shifted = arr - max_val
            exp_vals = np.exp(shifted)
            sum_exp = np.sum(exp_vals)

            if sum_exp <= 0.0 or not np.isfinite(sum_exp):
                return 0.0

            probs = exp_vals / sum_exp

            # Shannon entropy in bits (base 2)
            entropy_bits = -np.sum(
                np.where(probs > 0.0, probs * np.log2(probs + LOG2_EPSILON), 0.0)
            )

            # Guard against tiny floating point negatives
            return max(0.0, float(entropy_bits))

        except Exception as exc:
            logger.debug("Failed to compute entropy from logprobs: %s", exc)
            return 0.0

    def extract_token_metrics(self, vllm_request_output: Any) -> IntrinsicMetrics:
        """
        Parse a vLLM RequestOutput (or mock object) into an IntrinsicMetrics object.

        Parameters
        ----------
        vllm_request_output : RequestOutput | MockRequestOutput
            Output structure returned by vLLM engine.

        Returns
        -------
        IntrinsicMetrics
        """
        defaults = IntrinsicMetrics(
            l_bar=0.0,
            l_min=0.0,
            var_l=0.0,
            H_bar=0.0,
            num_tokens=0,
            generated_text="",
        )

        if not hasattr(vllm_request_output, "outputs") or not vllm_request_output.outputs:
            return defaults

        completion = vllm_request_output.outputs[0]
        generated_text = getattr(completion, "text", "") or ""
        token_logprobs_seq = getattr(completion, "logprobs", None) or []

        if not token_logprobs_seq:
            return IntrinsicMetrics(
                l_bar=0.0,
                l_min=0.0,
                var_l=0.0,
                H_bar=0.0,
                num_tokens=0,
                generated_text=generated_text,
            )

        chosen_logprobs: List[float] = []
        per_token_entropies: List[float] = []

        for token_entry in token_logprobs_seq:
            if not token_entry:
                continue

            # In vLLM, token_entry is a dict mapping token_id/token_str -> Logprob
            if isinstance(token_entry, dict):
                # Selected token has the maximum logprob among candidate options
                best_val = None
                for candidate in token_entry.values():
                    lp = getattr(candidate, "logprob", candidate)
                    if isinstance(lp, (int, float)) and np.isfinite(lp):
                        if best_val is None or lp > best_val:
                            best_val = float(lp)

                if best_val is not None:
                    chosen_logprobs.append(best_val)
                    per_token_entropies.append(self.compute_entropy_from_logprobs(token_entry))
            elif isinstance(token_entry, (int, float)):
                lp = float(token_entry)
                if np.isfinite(lp):
                    chosen_logprobs.append(lp)
                    per_token_entropies.append(0.0)

        if not chosen_logprobs:
            return IntrinsicMetrics(
                l_bar=0.0,
                l_min=0.0,
                var_l=0.0,
                H_bar=0.0,
                num_tokens=0,
                generated_text=generated_text,
            )

        lp_arr = np.asarray(chosen_logprobs, dtype=np.float64)
        en_arr = np.asarray(per_token_entropies, dtype=np.float64)

        return IntrinsicMetrics(
            l_bar=float(np.mean(lp_arr)),
            l_min=float(np.min(lp_arr)),
            var_l=float(np.var(lp_arr)),
            H_bar=float(np.mean(en_arr)) if en_arr.size > 0 else 0.0,
            num_tokens=int(lp_arr.size),
            generated_text=generated_text,
        )
