"""
rag_guard.detector
------------------
Zero-dependency composite hallucination detector and faithfulness scoring engine.
Evaluates the 5-dimensional feature vector [l_bar, l_min, var_l, H_bar, R_entity]
using algebraically pre-folded logistic regression weights.
Runtime dependencies: numpy only (no scikit-learn or torch required).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from rag_guard.config import DEFAULT_WEIGHTS_PATH, FEATURE_NAMES
from rag_guard.intrinsic import IntrinsicMetrics

logger = logging.getLogger("rag_guard.detector")


@dataclass(frozen=True)
class FaithfulnessResult:
    """Detection result and decomposition for a given query/response pair."""

    faithfulness_score: float  # S_faith in [0.0, 1.0] (1.0 = fully grounded/faithful)
    hallucination_score: float  # 1.0 - S_faith in [0.0, 1.0] (1.0 = definite hallucination)
    is_hallucination: bool  # True if faithfulness_score < optimal_threshold
    threshold: float  # Decision threshold used for classification
    features: Dict[str, float]  # Named decomposition of feature vector x
    logit: float  # Raw dot product w · x + b prior to sigmoid

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to serializable dictionary."""
        return {
            "faithfulness_score": round(self.faithfulness_score, 4),
            "hallucination_score": round(self.hallucination_score, 4),
            "is_hallucination": self.is_hallucination,
            "threshold": self.threshold,
            "logit": round(self.logit, 4),
            "features": {k: round(v, 4) for k, v in self.features.items()},
        }


class CompositeHallucinationDetector:
    """
    Lightweight, single-pass linear detector computing Composite Faithfulness Score S_faith.
    Executes in under 0.2 ms with zero external ML framework overhead.
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        weights_raw: Optional[Sequence[float]] = None,
        bias_raw: Optional[float] = None,
        optimal_threshold: Optional[float] = None,
    ) -> None:
        """
        Initialize detector with pre-calibrated weights.

        Parameters
        ----------
        weights_path : str | Path, optional
            Path to weights.json file. Defaults to bundled src/rag_guard/weights.json.
        weights_raw : Sequence[float], optional
            Directly override 5D raw weights [w1, w2, w3, w4, w5].
        bias_raw : float, optional
            Directly override scalar raw bias.
        optimal_threshold : float, optional
            Directly override decision threshold (default 0.50).
        """
        self.weights_path = Path(weights_path or DEFAULT_WEIGHTS_PATH)
        self.feature_names = FEATURE_NAMES

        if weights_raw is not None and bias_raw is not None:
            self._w_raw = np.asarray(weights_raw, dtype=np.float64)
            self._b_raw = float(bias_raw)
            self._threshold = float(optimal_threshold if optimal_threshold is not None else 0.50)
            self._metadata = {"source": "manual_override"}
        else:
            self._load_weights_from_file(self.weights_path)

        if self._w_raw.shape != (len(self.feature_names),):
            raise ValueError(
                f"Expected weights of shape ({len(self.feature_names)},), "
                f"got {self._w_raw.shape}"
            )

    def _load_weights_from_file(self, path: Path) -> None:
        """Load and parse weights.json."""
        if not path.exists():
            raise FileNotFoundError(
                f"Weights file not found at '{path}'. Run calibrate.py or specify valid weights."
            )

        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        self._w_raw = np.asarray(data["weights_raw"], dtype=np.float64)
        self._b_raw = float(data["bias_raw"])
        self._threshold = float(data.get("optimal_threshold", 0.50))
        self._metadata = data.get("calibration_metadata", {})

    @property
    def weights(self) -> np.ndarray:
        """Raw 5D weight vector."""
        return self._w_raw.copy()

    @property
    def bias(self) -> float:
        """Raw scalar bias."""
        return self._b_raw

    @property
    def threshold(self) -> float:
        """Classification decision threshold."""
        return self._threshold

    @property
    def metadata(self) -> Dict[str, Any]:
        """Calibration metadata."""
        return self._metadata.copy()

    @staticmethod
    def _sigmoid(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Numerically clipped sigmoid activation."""
        # Clip z to prevent exp() overflow / underflow
        z_clipped = np.clip(z, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def evaluate_vector(self, feature_vector: Sequence[float]) -> FaithfulnessResult:
        """
        Score a 5D feature vector [l_bar, l_min, var_l, H_bar, R_entity].

        Parameters
        ----------
        feature_vector : Sequence[float]
            Ordered values matching FEATURE_NAMES.

        Returns
        -------
        FaithfulnessResult
        """
        x = np.asarray(feature_vector, dtype=np.float64)
        if x.shape != (5,):
            raise ValueError(f"Feature vector must have exactly 5 elements, got shape {x.shape}")

        # Replace NaNs or Infs with neutral fallbacks
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        logit = float(np.dot(self._w_raw, x) + self._b_raw)
        s_faith = float(self._sigmoid(logit))
        h_score = max(0.0, min(1.0, 1.0 - s_faith))
        is_hallucination = bool(s_faith < self._threshold)

        feat_dict = {name: float(val) for name, val in zip(self.feature_names, x)}

        return FaithfulnessResult(
            faithfulness_score=s_faith,
            hallucination_score=h_score,
            is_hallucination=is_hallucination,
            threshold=self._threshold,
            features=feat_dict,
            logit=logit,
        )

    def evaluate(
        self,
        intrinsic: IntrinsicMetrics,
        r_entity: float,
    ) -> FaithfulnessResult:
        """
        Convenience evaluator combining IntrinsicMetrics object and grounding ratio.

        Parameters
        ----------
        intrinsic : IntrinsicMetrics
            Intrinsic uncertainty features (l_bar, l_min, var_l, H_bar).
        r_entity : float
            Deterministic entity grounding ratio.

        Returns
        -------
        FaithfulnessResult
        """
        vector = [
            intrinsic.l_bar,
            intrinsic.l_min,
            intrinsic.var_l,
            intrinsic.H_bar,
            float(r_entity),
        ]
        return self.evaluate_vector(vector)

    def batch_evaluate(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        Vectorized high-throughput batch scoring over N samples.

        Parameters
        ----------
        feature_matrix : np.ndarray of shape (N, 5)

        Returns
        -------
        np.ndarray of shape (N,) with faithfulness probabilities in [0.0, 1.0].
        """
        x = np.asarray(feature_matrix, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 5:
            raise ValueError(f"feature_matrix must have shape (N, 5), got {x.shape}")

        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        logits = np.dot(x, self._w_raw) + self._b_raw
        return self._sigmoid(logits)
