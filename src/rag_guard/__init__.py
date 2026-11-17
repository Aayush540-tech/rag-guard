"""
rag_guard
=========
Intrinsic Uncertainty Quantification and Deterministic Verification
for Real-Time RAG Hallucination Detection.
"""

from __future__ import annotations

from rag_guard.calibrate import (
    CrossValidationMetrics,
    FoldedWeights,
    calibrate_and_fold,
)
from rag_guard.config import (
    DEFAULT_FUZZY_THRESHOLD,
    DEFAULT_TOP_K_LOGPROBS,
    DEFAULT_WEIGHTS_PATH,
    FEATURE_NAMES,
    LATENCY_BUDGET_MS,
    STREAM_CLAUSE_DELIMITERS,
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
from rag_guard.stream import (
    RAGGuardStream,
    StreamFinalResult,
    StreamStepResult,
)
from rag_guard.sweep import (
    DEFAULT_SWEEP_THRESHOLDS,
    SweepSummary,
    ThresholdResult,
    run_threshold_sweep,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # Config
    "DEFAULT_FUZZY_THRESHOLD",
    "DEFAULT_TOP_K_LOGPROBS",
    "DEFAULT_WEIGHTS_PATH",
    "FEATURE_NAMES",
    "LATENCY_BUDGET_MS",
    "STREAM_CLAUSE_DELIMITERS",
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
    # Real-Time Streaming Guardrail
    "RAGGuardStream",
    "StreamStepResult",
    "StreamFinalResult",
    # Calibration & Weight Folding
    "CrossValidationMetrics",
    "FoldedWeights",
    "calibrate_and_fold",
    # Threshold Sensitivity Sweep
    "DEFAULT_SWEEP_THRESHOLDS",
    "SweepSummary",
    "ThresholdResult",
    "run_threshold_sweep",
]
