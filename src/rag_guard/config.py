"""
rag_guard.config
----------------
Global constants, default hyperparameters, and operational constraints for rag_guard.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import FrozenSet

# Mathematical & Feature Extraction Hyperparameters
DEFAULT_TOP_K_LOGPROBS: int = 5
DEFAULT_FUZZY_THRESHOLD: float = 85.0

# Supported Named Entity Types for Grounding Verification
TARGET_ENTITY_TYPES: FrozenSet[str] = frozenset(
    {
        "PERSON",
        "ORG",
        "GPE",
        "DATE",
        "TIME",
        "CARDINAL",
        "MONEY",
        "QUANTITY",
        "PERCENT",
    }
)

# Streaming Delimiters: Sentence and Clause Boundaries
STREAM_CLAUSE_DELIMITERS: FrozenSet[str] = frozenset({".", "!", "?", "\n", ";"})

# System Performance Constraints
LATENCY_BUDGET_MS: float = 20.0

# Weight File Paths
DEFAULT_WEIGHTS_PATH: Path = Path(__file__).resolve().parent / "weights.json"

# Feature Names and Vector Ordering
FEATURE_NAMES: list[str] = [
    "l_bar",    # Mean token log-probability
    "l_min",    # Minimum token log-probability
    "var_l",    # Log-probability variance
    "H_bar",    # Mean predictive Shannon entropy (top-5)
    "R_entity", # Deterministic entity grounding ratio
]
