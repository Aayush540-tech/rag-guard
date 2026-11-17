"""
rag_guard
=========
Intrinsic Uncertainty Quantification and Deterministic Verification
for Real-Time RAG Hallucination Detection.
"""

from __future__ import annotations

from rag_guard.config import (
    DEFAULT_FUZZY_THRESHOLD,
    DEFAULT_TOP_K_LOGPROBS,
    FEATURE_NAMES,
    LATENCY_BUDGET_MS,
    TARGET_ENTITY_TYPES,
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

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # Config
    "DEFAULT_FUZZY_THRESHOLD",
    "DEFAULT_TOP_K_LOGPROBS",
    "FEATURE_NAMES",
    "LATENCY_BUDGET_MS",
    "TARGET_ENTITY_TYPES",
    # Intrinsic Uncertainty
    "IntrinsicMetrics",
    "IntrinsicMetricsExtractor",
    "OnlineIntrinsicTracker",
    # Deterministic Grounding
    "EntityGroundingEvaluator",
    # Composite Detection
    "CompositeHallucinationDetector",
    "FaithfulnessResult",
]
