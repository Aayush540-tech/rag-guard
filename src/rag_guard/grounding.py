"""
rag_guard.grounding
-------------------
Deterministic entity grounding evaluation using spaCy NER, numeric regex,
and RapidFuzz Levenshtein set-similarity matching against pre-cached context.
Runtime dependencies: spacy (optional fallback), rapidfuzz.
"""

from __future__ import annotations

import logging
import re
from typing import FrozenSet, Iterable, Optional, Set

from rapidfuzz import fuzz

from rag_guard.config import DEFAULT_FUZZY_THRESHOLD, TARGET_ENTITY_TYPES

logger = logging.getLogger("rag_guard.grounding")

# Lazy-loaded spaCy language pipeline
_SPACY_NLP = None
_SPACY_TRIED = False


def _get_spacy_nlp():
    """Attempt to load spaCy en_core_web_sm once; gracefully degrade if absent."""
    global _SPACY_NLP, _SPACY_TRIED
    if not _SPACY_TRIED:
        _SPACY_TRIED = True
        try:
            import spacy
            _SPACY_NLP = spacy.load("en_core_web_sm")
            logger.info("spaCy en_core_web_sm loaded successfully.")
        except Exception as exc:
            _SPACY_NLP = None
            logger.warning(
                "spaCy en_core_web_sm unavailable (%s). "
                "Operating in regex-only entity extraction mode.",
                exc,
            )
    return _SPACY_NLP


class EntityGroundingEvaluator:
    """
    Extracts named entities and numeric tokens from text and computes
    grounding ratio R_entity against retrieved context passages.
    """

    # Comprehensive regex capturing financial, percentage, scientific, and numeric entities
    _NUMERIC_RE = re.compile(
        r"""
        (?:
            \$[\d,]+(?:\.\d+)?[BKMT]?
          | \d{1,3}(?:,\d{3})*(?:\.\d+)?[BKMT]?
          | \d{4}
          | \d+(?:\.\d+)?\s*%
          | \d+(?:\.\d+)?\s*(?:km|kg|mph|GHz|MHz|TB|MB|GB|KB|ms|ns|s|m|h)
        )
        """,
        re.VERBOSE | re.IGNORECASE,
    )

    def __init__(
        self,
        fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
        target_entity_types: FrozenSet[str] = TARGET_ENTITY_TYPES,
        nlp=None,
    ) -> None:
        self.fuzzy_threshold = fuzzy_threshold
        self.target_entity_types = target_entity_types
        self._nlp = nlp

    @property
    def nlp(self):
        if self._nlp is not None:
            return self._nlp
        return _get_spacy_nlp()

    @staticmethod
    def clean_entity(raw: str) -> str:
        """Strip whitespace, collapse consecutive spaces, and lowercase."""
        if not raw:
            return ""
        cleaned = " ".join(raw.split()).lower()
        return cleaned if len(cleaned) > 1 else ""

    def extract_entities(self, text: str) -> Set[str]:
        """
        Extract named entities and numeric tokens from text.

        Parameters
        ----------
        text : str
            Input text (context, answer, or streaming clause).

        Returns
        -------
        Set[str]
            Normalized, lowercase entity strings.
        """
        if not isinstance(text, str) or not text.strip():
            return set()

        entities: Set[str] = set()

        # 1. spaCy NER (if available)
        nlp_pipeline = self.nlp
        if nlp_pipeline is not None:
            try:
                doc = nlp_pipeline(text)
                for ent in doc.ents:
                    if ent.label_ in self.target_entity_types:
                        cleaned = self.clean_entity(ent.text)
                        if cleaned:
                            entities.add(cleaned)
            except Exception as exc:
                logger.debug("spaCy extraction error: %s", exc)

        # 2. High-precision numeric regex
        for match in self._NUMERIC_RE.finditer(text):
            token = match.group(0).strip().lower()
            if token:
                entities.add(token)

        return entities

    def cache_context_entities(self, context: str) -> Set[str]:
        """
        Extract and return entities from context for zero-overhead caching in streaming.

        Parameters
        ----------
        context : str
            Retrieved context text.

        Returns
        -------
        Set[str]
            Pre-computed context entity set E_C.
        """
        return self.extract_entities(context)

    def compute_grounding_ratio_cached(
        self,
        answer_entities: Set[str],
        cached_context_entities: Set[str],
    ) -> float:
        """
        Compute R_entity given pre-extracted answer and context entity sets.

        Parameters
        ----------
        answer_entities : Set[str]
            Entities E_A extracted from the answer.
        cached_context_entities : Set[str]
            Entities E_C pre-cached from the retrieved context.

        Returns
        -------
        float in [0.0, 1.0]
            1.0 if answer_entities is empty (vacuously grounded).
        """
        if not answer_entities:
            return 1.0

        if not cached_context_entities:
            return 0.0

        ctx_list = list(cached_context_entities)
        grounded_count = 0

        for ent in answer_entities:
            best_score = max(
                fuzz.token_set_ratio(ent, ctx_ent) for ctx_ent in ctx_list
            )
            if best_score >= self.fuzzy_threshold:
                grounded_count += 1

        return float(grounded_count / len(answer_entities))

    def compute_grounding_ratio(self, answer: str, context: str) -> float:
        """
        End-to-end evaluation of R_entity from raw answer and context strings.

        Parameters
        ----------
        answer : str
            Model generated response.
        context : str
            Retrieved context.

        Returns
        -------
        float in [0.0, 1.0]
        """
        e_answer = self.extract_entities(answer)
        e_context = self.extract_entities(context)
        return self.compute_grounding_ratio_cached(e_answer, e_context)
