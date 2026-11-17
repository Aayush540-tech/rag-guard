"""
rag_truth_phase2.py
===================
Production-grade intrinsic uncertainty and entity grounding detection pipeline
for Retrieval-Augmented Generation (RAG) systems.

Features:
  - Single vLLM forward-pass logprob/entropy extraction
  - spaCy NER + regex entity grounding via RapidFuzz
  - Logistic Regression composite classifier (sigmoid output)
  - Sub-20 ms latency target verified via micro-benchmark
  - Seamless CPU/mock fallback when CUDA / vLLM unavailable

Author  : Principal ML & Systems Engineer
Version : 2.0.0
Python  : 3.10+
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rag_truth_phase2")

# ---------------------------------------------------------------------------
# Optional heavy imports – graceful degradation
# ---------------------------------------------------------------------------
try:
    import spacy  # type: ignore
    _NLP = spacy.load("en_core_web_sm")
    _SPACY_AVAILABLE = True
    logger.info("spaCy en_core_web_sm loaded successfully.")
except Exception as _spacy_err:
    _NLP = None
    _SPACY_AVAILABLE = False
    logger.warning("spaCy unavailable (%s). Regex-only entity extraction active.", _spacy_err)

try:
    from vllm import LLM, SamplingParams  # type: ignore
    _VLLM_AVAILABLE = True
    logger.info("vLLM imported successfully.")
except Exception as _vllm_err:
    _VLLM_AVAILABLE = False
    logger.warning("vLLM unavailable (%s). Mock inference engine active.", _vllm_err)

try:
    import torch  # type: ignore
    _CUDA_AVAILABLE = torch.cuda.is_available()
except Exception:
    _CUDA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------
TARGET_ENTITY_TYPES: frozenset = frozenset(
    {"PERSON", "ORG", "GPE", "DATE", "TIME", "CARDINAL", "MONEY", "QUANTITY", "PERCENT"}
)
ENTITY_MATCH_THRESHOLD: float = 85.0
TOP_K_LOGPROBS: int = 5
LOGISTIC_RANDOM_STATE: int = 42
TRAIN_TEST_SPLIT_RATIO: float = 0.2
BENCHMARK_ITERATIONS: int = 100
LATENCY_BUDGET_MS: float = 20.0

# ---------------------------------------------------------------------------
# Dataclasses for vLLM output simulation
# ---------------------------------------------------------------------------
@dataclass
class MockLogprobEntry:
    """Mirrors vLLM's Logprob namedtuple structure."""
    logprob: float
    rank: int = 0
    decoded_token: str = ""


@dataclass
class MockCompletionOutput:
    """Mirrors vLLM's CompletionOutput structure."""
    text: str
    logprobs: list = field(default_factory=list)
    token_ids: list = field(default_factory=list)
    finish_reason: str = "stop"


@dataclass
class MockRequestOutput:
    """Mirrors vLLM's RequestOutput structure."""
    request_id: str
    prompt: str
    outputs: list = field(default_factory=list)
    finished: bool = True


# ===========================================================================
# CLASS 1 -- RAGTruthLoader
# ===========================================================================
class RAGTruthLoader:
    """
    Loads the RAGTruth benchmark dataset from a JSONL file or generates
    realistic mock data for dry-run pipeline validation.

    Dataset schema expected per line:
        {
            "id": str,
            "prompt": str,
            "context": str,
            "response": str,
            "labels": [{"start": int, "end": int, "hallucinated": bool}, ...]
        }
    """

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Read a JSONL file and parse it into a structured DataFrame.

        Parameters
        ----------
        file_path : str
            Absolute or relative path to the .jsonl dataset file.

        Returns
        -------
        pd.DataFrame
            Columns: id, prompt, context, response, labels, Label
            Label = 0 (grounded) | 1 (hallucinated)
        """
        if not os.path.exists(file_path):
            logger.warning("Dataset file '%s' not found. Falling back to mock data.", file_path)
            return self.generate_mock_data()

        records: list = []
        with open(file_path, "r", encoding="utf-8") as fh:
            for line_no, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed JSON at line %d: %s", line_no, exc)
                    continue

                sample_id    = obj.get("id", f"sample_{line_no}")
                prompt       = obj.get("prompt", "")
                context      = obj.get("context", "")
                response     = obj.get("response", "")
                labels       = obj.get("labels", [])
                binary_label = self._resolve_label(labels)

                records.append({
                    "id": sample_id,
                    "prompt": prompt,
                    "context": context,
                    "response": response,
                    "labels": labels,
                    "Label": binary_label,
                })

        if not records:
            logger.warning("JSONL file was empty or fully malformed. Falling back to mock data.")
            return self.generate_mock_data()

        df = pd.DataFrame(records)
        logger.info(
            "Loaded %d samples from '%s'  |  Hallucinated: %d  Grounded: %d",
            len(df), file_path, df["Label"].sum(), (df["Label"] == 0).sum(),
        )
        return df

    def generate_mock_data(self, num_samples: int = 50) -> pd.DataFrame:
        """
        Generate a realistic mock dataset for pipeline validation.

        Parameters
        ----------
        num_samples : int
            Number of mock samples to generate (default 50).

        Returns
        -------
        pd.DataFrame
            Same schema as load_data() output.
        """
        rng = np.random.default_rng(seed=2024)

        grounded_contexts = [
            (
                "The Eiffel Tower was completed in 1889 and stands 330 metres tall. "
                "It was designed by Gustave Eiffel for the 1889 World's Fair in Paris, France."
            ),
            (
                "Apple Inc. was founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne. "
                "The company is headquartered in Cupertino, California, with revenue of $394.3B in 2022."
            ),
            (
                "The Amazon River is approximately 6,400 km long and flows through Brazil. "
                "It discharges about 209,000 cubic metres per second into the Atlantic Ocean."
            ),
            (
                "NASA's Artemis program aims to land astronauts on the Moon by 2026. "
                "The budget for the programme is estimated at $93 billion through 2025."
            ),
            (
                "Python 3.11 was released on October 24, 2022. "
                "It features a 10-60% performance improvement over Python 3.10."
            ),
        ]
        grounded_responses = [
            "The Eiffel Tower, completed in 1889, stands 330 metres tall in Paris, France.",
            "Apple Inc. was founded in 1976 by Steve Jobs, Wozniak, and Wayne in Cupertino.",
            "The Amazon River is about 6,400 km long and flows through Brazil.",
            "NASA's Artemis program targets a Moon landing by 2026 with a $93 billion budget.",
            "Python 3.11 was released on October 24, 2022, offering up to 60% speed gains.",
        ]
        hallucinated_contexts = [
            (
                "Marie Curie was born in Warsaw in 1867 and won the Nobel Prize in Physics in 1903. "
                "She was the first woman to win a Nobel Prize."
            ),
            (
                "The Great Wall of China stretches approximately 21,196 km and was built over many centuries. "
                "Construction began during the 7th century BC."
            ),
            (
                "SpaceX was founded by Elon Musk in 2002. "
                "The Falcon 9 rocket has completed over 200 successful launches."
            ),
            (
                "The human genome contains approximately 3 billion base pairs and about 20,000 protein-coding genes. "
                "The Human Genome Project was completed in 2003."
            ),
            (
                "Mount Everest is 8,849 metres above sea level. "
                "It was first summited by Sir Edmund Hillary and Tenzing Norgay in 1953."
            ),
        ]
        hallucinated_responses = [
            "Marie Curie was born in Paris in 1872 and won the Nobel Prize in Chemistry in 1901.",
            "The Great Wall of China is 50,000 km long and was built in the Ming Dynasty exclusively.",
            "SpaceX was founded by Elon Musk in 2005 and has completed 50 successful Falcon 9 launches.",
            "The human genome has 5 billion base pairs and 100,000 genes, completed in 2010.",
            "Mount Everest is 9,200 metres tall and was first summited in 1960 by Hillary alone.",
        ]

        records: list = []
        half = num_samples // 2

        for i in range(half):
            ctx_idx = i % len(grounded_contexts)
            res_idx = i % len(grounded_responses)
            records.append({
                "id": f"grounded_{i:04d}",
                "prompt": f"Based on the context, answer the following question. Sample {i}.",
                "context": grounded_contexts[ctx_idx],
                "response": grounded_responses[res_idx],
                "labels": [],
                "Label": 0,
            })

        for i in range(num_samples - half):
            ctx_idx = i % len(hallucinated_contexts)
            res_idx = i % len(hallucinated_responses)
            span_start = rng.integers(0, 30)
            span_end   = span_start + rng.integers(10, 50)
            records.append({
                "id": f"hallucinated_{i:04d}",
                "prompt": f"Based on the context, answer the following question. Sample {i}.",
                "context": hallucinated_contexts[ctx_idx],
                "response": hallucinated_responses[res_idx],
                "labels": [{"start": int(span_start), "end": int(span_end), "hallucinated": True}],
                "Label": 1,
            })

        df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
        logger.info(
            "Generated %d mock samples  |  Grounded: %d  Hallucinated: %d",
            len(df), half, num_samples - half,
        )
        return df

    @staticmethod
    def _resolve_label(labels: list) -> int:
        """Map span-level labels to binary: 1 if any span is hallucinated, else 0."""
        if not isinstance(labels, list) or len(labels) == 0:
            return 0
        for span in labels:
            if isinstance(span, dict) and span.get("hallucinated", False):
                return 1
        return 0


# ===========================================================================
# CLASS 2 -- EntityGroundingEvaluator
# ===========================================================================
class EntityGroundingEvaluator:
    """
    Extracts named entities and computes R_entity using RapidFuzz token-set matching.
    Falls back to regex-based extraction when spaCy is unavailable.
    """

    _NUMERIC_RE = re.compile(
        r"""
        (?:
            \$[\d,]+(?:\.\d+)?[BKMT]?
          | \d{1,3}(?:,\d{3})*(?:\.\d+)?[BKMT]?
          | \d{4}
          | \d+(?:\.\d+)?\s*%
          | \d+(?:\.\d+)?\s*(?:km|kg|mph|GHz|TB|MB|GB|ms|ns)
        )
        """,
        re.VERBOSE | re.IGNORECASE,
    )

    def __init__(self) -> None:
        self._nlp = _NLP
        self._spacy_ok = _SPACY_AVAILABLE

    def extract_entities(self, text: str) -> set:
        """
        Extract named entities from text using spaCy NER and regex fallback.

        Parameters
        ----------
        text : str  Input text.

        Returns
        -------
        set[str]  Lowercase-normalised entity strings.
        """
        if not isinstance(text, str) or not text.strip():
            return set()

        entities: set = set()

        if self._spacy_ok and self._nlp is not None:
            try:
                doc = self._nlp(text)
                for ent in doc.ents:
                    if ent.label_ in TARGET_ENTITY_TYPES:
                        cleaned = self._clean_entity(ent.text)
                        if cleaned:
                            entities.add(cleaned)
            except Exception as exc:
                logger.debug("spaCy NER error: %s", exc)

        for match in self._NUMERIC_RE.finditer(text):
            token = match.group(0).strip()
            if token:
                entities.add(token.lower())

        return entities

    def compute_grounding_ratio(self, answer: str, context: str) -> float:
        """
        Compute R_entity: fraction of answer entities that fuzzy-match context entities.

        Parameters
        ----------
        answer  : str  Model-generated response.
        context : str  Source context / retrieved passages.

        Returns
        -------
        float in [0.0, 1.0] -- 1.0 if no answer entities (vacuously true).
        """
        e_answer  = self.extract_entities(answer)
        e_context = self.extract_entities(context)

        if len(e_answer) == 0:
            return 1.0

        if len(e_context) == 0:
            return 0.0

        e_context_list = list(e_context)
        grounded_count = 0

        for entity in e_answer:
            best_score = max(
                fuzz.token_set_ratio(entity, ctx_ent)
                for ctx_ent in e_context_list
            )
            if best_score >= ENTITY_MATCH_THRESHOLD:
                grounded_count += 1

        return grounded_count / len(e_answer)

    @staticmethod
    def _clean_entity(raw: str) -> str:
        """Strip whitespace, collapse internal spaces, lowercase."""
        cleaned = " ".join(raw.split()).lower()
        return cleaned if len(cleaned) > 1 else ""


# ===========================================================================
# CLASS 3 -- IntrinsicMetricsExtractor
# ===========================================================================
class IntrinsicMetricsExtractor:
    """
    Extracts intrinsic confidence metrics from vLLM logprob outputs.

    Metrics computed:
      l_bar : mean log-probability
      l_min : minimum log-probability
      var_l : log-probability variance (population)
      H_bar : mean per-token Shannon entropy in bits
    """

    def compute_entropy_from_logprobs(self, top_logprobs_dict: dict) -> float:
        """
        Numerically stable Softmax -> Shannon Entropy over top-k logprobs.

        Parameters
        ----------
        top_logprobs_dict : dict
            token_id -> object with .logprob attribute.

        Returns
        -------
        float  Shannon entropy in bits. 0.0 on error.
        """
        if not top_logprobs_dict:
            return 0.0

        try:
            raw_logprobs = np.array(
                [v.logprob for v in top_logprobs_dict.values()],
                dtype=np.float64,
            )

            if raw_logprobs.size == 0:
                return 0.0

            # Numerically stable softmax: subtract max before exponentiation
            shifted  = raw_logprobs - raw_logprobs.max()
            exp_vals = np.exp(shifted)
            probs    = exp_vals / exp_vals.sum()

            # Shannon entropy in bits: H = -sum(p * log2(p))
            entropy_bits = -np.sum(
                np.where(probs > 0.0, probs * np.log2(probs + 1e-12), 0.0)
            )
            return float(entropy_bits)

        except Exception as exc:
            logger.debug("Entropy computation error: %s", exc)
            return 0.0

    def extract_token_metrics(self, vllm_output_sample) -> dict:
        """
        Parse a single vLLM RequestOutput and compute all intrinsic metrics.

        Parameters
        ----------
        vllm_output_sample : RequestOutput | MockRequestOutput

        Returns
        -------
        dict  Keys: l_bar, l_min, var_l, H_bar, generated_text
        """
        defaults: dict = {
            "l_bar": 0.0,
            "l_min": 0.0,
            "var_l": 0.0,
            "H_bar": 0.0,
            "generated_text": "",
        }

        try:
            if not vllm_output_sample.outputs:
                return defaults

            completion = vllm_output_sample.outputs[0]
            generated_text = completion.text or ""
            token_logprobs_sequence = completion.logprobs or []

            if not token_logprobs_sequence:
                defaults["generated_text"] = generated_text
                return defaults

            chosen_logprobs: list = []
            per_token_entropies: list = []

            for token_dict in token_logprobs_sequence:
                if not isinstance(token_dict, dict) or not token_dict:
                    continue

                best_entry = max(token_dict.values(), key=lambda e: e.logprob)
                chosen_logprobs.append(best_entry.logprob)

                h_t = self.compute_entropy_from_logprobs(token_dict)
                per_token_entropies.append(h_t)

            if not chosen_logprobs:
                defaults["generated_text"] = generated_text
                return defaults

            lp_arr = np.array(chosen_logprobs, dtype=np.float64)
            en_arr = np.array(per_token_entropies, dtype=np.float64)

            return {
                "l_bar": float(np.mean(lp_arr)),
                "l_min": float(np.min(lp_arr)),
                "var_l": float(np.var(lp_arr)),
                "H_bar": float(np.mean(en_arr)) if en_arr.size > 0 else 0.0,
                "generated_text": generated_text,
            }

        except Exception as exc:
            logger.debug("Token metrics extraction error: %s", exc)
            return defaults


# ===========================================================================
# CLASS 4 -- vLLMInferenceEngine
# ===========================================================================
class vLLMInferenceEngine:
    """
    Wraps vLLM's LLM with GPU auto-detection and transparent mock fallback.

    Model : meta-llama/Llama-3.2-3B-Instruct
    Params: temperature=0.0, max_tokens=512, logprobs=5
    """

    _MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

    def __init__(self) -> None:
        self._use_mock = not (_VLLM_AVAILABLE and _CUDA_AVAILABLE)
        self._llm = None
        self._sampling_params = None
        self._rng = np.random.default_rng(seed=2024)

        if not self._use_mock:
            self._init_vllm()
        else:
            reason = (
                "vLLM not installed" if not _VLLM_AVAILABLE else "CUDA GPU not detected"
            )
            logger.info("Mock inference engine active (%s).", reason)

    def process_batch(self, prompts: list) -> list:
        """
        Run a single forward pass for all prompts.

        Parameters
        ----------
        prompts : list[str]

        Returns
        -------
        list[dict]  Each: {"generated_text": str, "raw_output": RequestOutput}
        """
        if self._use_mock:
            return self._mock_generate(prompts)
        return self._vllm_generate(prompts)

    def _init_vllm(self) -> None:
        """Initialise real vLLM LLM and SamplingParams."""
        try:
            self._sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=512,
                logprobs=TOP_K_LOGPROBS,
            )
            self._llm = LLM(
                model=self._MODEL_ID,
                dtype="float16",
                gpu_memory_utilization=0.85,
                max_model_len=2048,
                trust_remote_code=True,
            )
            logger.info("vLLM LLM initialised with model '%s'.", self._MODEL_ID)
        except Exception as exc:
            logger.error("vLLM initialisation failed (%s). Switching to mock.", exc)
            self._use_mock = True
            self._llm = None

    def _vllm_generate(self, prompts: list) -> list:
        """Run real vLLM generation."""
        outputs = self._llm.generate(prompts, self._sampling_params)
        results = []
        for out in outputs:
            text = out.outputs[0].text if out.outputs else ""
            results.append({"generated_text": text, "raw_output": out})
        return results

    def _mock_generate(self, prompts: list) -> list:
        """
        Simulate vLLM output with realistic logprob distributions.

        Even-index prompts  -> high confidence (grounded-like logprobs)
        Odd-index prompts   -> low confidence  (hallucinated-like logprobs)
        """
        results = []

        for idx, prompt in enumerate(prompts):
            n_tokens = self._rng.integers(30, 80)

            if idx % 2 == 0:
                chosen_lp   = self._rng.uniform(-0.8, -0.05, size=n_tokens)
                noise_scale = 0.3
            else:
                chosen_lp   = self._rng.uniform(-4.5, -1.5, size=n_tokens)
                noise_scale = 1.5

            token_logprobs_seq: list = []
            for t in range(n_tokens):
                top_k_dict: dict = {}
                base_lp = chosen_lp[t]
                for rank in range(TOP_K_LOGPROBS):
                    tok_id = int(self._rng.integers(1000, 32000))
                    lp_val = base_lp - self._rng.exponential(noise_scale) * rank
                    top_k_dict[tok_id] = MockLogprobEntry(
                        logprob=float(lp_val),
                        rank=rank,
                        decoded_token=f"<token_{tok_id}>",
                    )
                token_logprobs_seq.append(top_k_dict)

            mock_text = (
                "The answer is grounded in the provided context."
                if idx % 2 == 0
                else "The answer contains fabricated information not in the context."
            )

            completion = MockCompletionOutput(
                text=mock_text,
                logprobs=token_logprobs_seq,
                token_ids=list(range(n_tokens)),
                finish_reason="stop",
            )
            raw_out = MockRequestOutput(
                request_id=f"mock-{idx:06d}",
                prompt=prompt,
                outputs=[completion],
                finished=True,
            )
            results.append({"generated_text": mock_text, "raw_output": raw_out})

        return results


# ===========================================================================
# CLASS 5 -- CompositeHallucinationDetector
# ===========================================================================
class CompositeHallucinationDetector:
    """
    Trains and evaluates a logistic regression classifier over five features.

    S_faith = sigma(w1*l_bar + w2*l_min + w3*var_l + w4*H_bar + w5*R_entity + b)
    """

    FEATURE_COLS = ["l_bar", "l_min", "var_l", "H_bar", "R_entity"]
    FEATURE_DISPLAY = [
        ("l_bar",    "Mean Logprob (l_bar)",           "Sequence Conf."),
        ("l_min",    "Min Logprob (l_min)",             "Token Hesitation"),
        ("var_l",    "Logprob Variance (var)",          "Conf. Stability"),
        ("H_bar",    "Mean Shannon Entropy (H_bar)",    "Vocab Ambiguity"),
        ("R_entity", "Entity Grounding Ratio (R_ent)",  "Factual Overlap"),
    ]

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.model  = LogisticRegression(
            random_state=LOGISTIC_RANDOM_STATE,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        )
        self._is_trained = False

    def train_and_evaluate(self, df: pd.DataFrame) -> None:
        """
        Fit the composite classifier and print the ROC-AUC report.

        Parameters
        ----------
        df : pd.DataFrame  Must contain FEATURE_COLS + 'Label'.
        """
        for col in self.FEATURE_COLS + ["Label"]:
            if col not in df.columns:
                raise ValueError(f"DataFrame missing required column: '{col}'")

        X = df[self.FEATURE_COLS].values.astype(np.float64)
        y = df["Label"].values.astype(int)

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TRAIN_TEST_SPLIT_RATIO,
            random_state=LOGISTIC_RANDOM_STATE,
            stratify=y,
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled  = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)
        self._is_trained = True

        composite_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        composite_auc   = roc_auc_score(y_test, composite_proba)

        feature_aucs = []
        for col_key, display_name, signal_name in self.FEATURE_DISPLAY:
            feat_idx  = self.FEATURE_COLS.index(col_key)
            feat_vals = X_test[:, feat_idx]
            auc_pos   = roc_auc_score(y_test, feat_vals)
            auc_neg   = roc_auc_score(y_test, -feat_vals)
            best_auc  = max(auc_pos, auc_neg)
            feature_aucs.append((display_name, signal_name, col_key, best_auc))

        self._print_report(feature_aucs, composite_auc)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Return hallucination probabilities.

        Parameters
        ----------
        features : np.ndarray of shape (n, 5)

        Returns
        -------
        np.ndarray of shape (n,) with float32 probabilities.
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained before calling predict_proba().")
        scaled = self.scaler.transform(features)
        return self.model.predict_proba(scaled)[:, 1].astype(np.float32)

    @staticmethod
    def _print_report(feature_aucs: list, composite_auc: float) -> None:
        """Render the standardised ROC-AUC report table."""
        sep_full  = "=" * 60
        sep_light = "-" * 60
        header    = "Phase 2 Feature Isolation & ROC-AUC Report"

        print(f"\n{sep_full}")
        print(f"{header:^60}")
        print(sep_full)
        print(f"{'Feature':<32}{'Target Signal':<21}{'ROC-AUC':>7}")
        print(sep_light)

        for display_name, signal_name, _, auc_val in feature_aucs:
            print(f"{display_name:<32}{signal_name:<21}{auc_val:>7.3f}")

        print(sep_light)
        print(f"{'COMPOSITE MODEL (Sigmoid)':<32}{'All Signals':<21}{composite_auc:>7.3f}")
        print(f"{sep_full}\n")


# ===========================================================================
# CLASS 6 -- LatencyProfiler
# ===========================================================================
class LatencyProfiler:
    """
    Micro-benchmarks five post-generation processing stages across
    BENCHMARK_ITERATIONS iterations and verifies sub-20 ms total latency.

    Stages:
      1. Logprob / Entropy mathematics
      2. String concatenation & prompt formatting
      3. spaCy NER entity extraction
      4. RapidFuzz fuzzy entity matching
      5. Logistic Regression inference
    """

    STAGE_NAMES = [
        "Logprob/Entropy Math",
        "String Concatenation",
        "spaCy NER",
        "RapidFuzz Matching",
        "Logistic Inference",
    ]

    def __init__(
        self,
        extractor: IntrinsicMetricsExtractor,
        grounder: EntityGroundingEvaluator,
        detector: CompositeHallucinationDetector,
        engine: vLLMInferenceEngine,
    ) -> None:
        self._extractor = extractor
        self._grounder  = grounder
        self._detector  = detector
        self._engine    = engine

    def run_benchmark(self) -> None:
        """Execute BENCHMARK_ITERATIONS iterations and print latency report."""
        if not self._detector._is_trained:
            logger.warning("Detector not trained. Stage 5 will record near-zero latency.")

        mock_output    = self._engine._mock_generate(["Benchmark prompt"])[0]["raw_output"]
        sample_answer  = (
            "Apple Inc. was founded in 1976 in Cupertino, California, "
            "with revenue of $394.3B in fiscal year 2022."
        )
        sample_context = (
            "Apple Inc. was founded in 1976 by Steve Jobs and Steve Wozniak "
            "in Cupertino, California. The company reported revenue of $394.3B in 2022."
        )
        dummy_features = np.array([[0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float64)

        timings: list = [[] for _ in range(5)]

        for _ in range(BENCHMARK_ITERATIONS):
            # Stage 1 -- Logprob / Entropy math
            t0 = time.perf_counter()
            self._extractor.extract_token_metrics(mock_output)
            timings[0].append((time.perf_counter() - t0) * 1e3)

            # Stage 2 -- String concatenation
            t0 = time.perf_counter()
            _ = f"Context: {sample_context}\n\nQuestion: {sample_answer}\n\nAnswer:"
            timings[1].append((time.perf_counter() - t0) * 1e3)

            # Stage 3 -- spaCy NER
            t0 = time.perf_counter()
            self._grounder.extract_entities(sample_answer)
            timings[2].append((time.perf_counter() - t0) * 1e3)

            # Stage 4 -- RapidFuzz matching
            t0 = time.perf_counter()
            self._grounder.compute_grounding_ratio(sample_answer, sample_context)
            timings[3].append((time.perf_counter() - t0) * 1e3)

            # Stage 5 -- Logistic Regression inference
            t0 = time.perf_counter()
            if self._detector._is_trained:
                self._detector.predict_proba(dummy_features)
            timings[4].append((time.perf_counter() - t0) * 1e3)

        self._print_benchmark_report(timings)

    def _print_benchmark_report(self, timings: list) -> None:
        sep_full  = "=" * 70
        sep_light = "-" * 70
        title     = f"Latency Profiler -- {BENCHMARK_ITERATIONS} Iterations"

        print(sep_full)
        print(f"{title:^70}")
        print(sep_full)
        print(
            f"{'Stage':<26}"
            f"{'Mean (ms)':>10}"
            f"{'Median (ms)':>12}"
            f"{'P99 (ms)':>10}"
            f"{'Total Contrib':>14}"
        )
        print(sep_light)

        stage_means: list = []
        for idx, (name, times_ms) in enumerate(zip(self.STAGE_NAMES, timings), start=1):
            arr     = np.array(times_ms)
            mean_ms = float(np.mean(arr))
            med_ms  = float(np.median(arr))
            p99_ms  = float(np.percentile(arr, 99))
            stage_means.append(mean_ms)
            print(
                f"{idx}. {name:<24}"
                f"{mean_ms:>10.3f}"
                f"{med_ms:>12.3f}"
                f"{p99_ms:>10.3f}"
                f"{'N/A':>14}"
            )

        total_ms  = sum(stage_means)
        budget_ok = total_ms < LATENCY_BUDGET_MS
        verdict   = "PASS" if budget_ok else "FAIL"

        print(sep_light)
        print(f"{'TOTAL PIPELINE OVERHEAD':<26}{total_ms:>10.3f} ms")
        print(f"{'Latency Budget':<26}{LATENCY_BUDGET_MS:>10.1f} ms")
        print(f"{'Verdict':<26}{verdict:>10}")
        print(sep_full + "\n")


# ===========================================================================
# PIPELINE ORCHESTRATOR
# ===========================================================================
def build_feature_dataframe(
    df: pd.DataFrame,
    engine: vLLMInferenceEngine,
    extractor: IntrinsicMetricsExtractor,
    grounder: EntityGroundingEvaluator,
) -> pd.DataFrame:
    """
    Run the full feature extraction pipeline for every row in df.

    1. Format RAG prompt (context + question)
    2. Run vLLM (or mock) inference in a single batched forward pass
    3. Extract intrinsic token metrics (l_bar, l_min, var_l, H_bar)
    4. Compute entity grounding ratio (R_entity)
    """
    prompts: list = []
    for _, row in df.iterrows():
        prompt = (
            f"Context:\n{row['context']}\n\n"
            f"Question:\n{row['prompt']}\n\n"
            "Answer:"
        )
        prompts.append(prompt)

    logger.info("Running inference on %d prompts...", len(prompts))
    batch_results = engine.process_batch(prompts)
    logger.info("Inference complete. Extracting metrics...")

    l_bar_list:    list = []
    l_min_list:    list = []
    var_l_list:    list = []
    H_bar_list:    list = []
    R_entity_list: list = []

    for i, (result, (_, row)) in enumerate(zip(batch_results, df.iterrows())):
        metrics = extractor.extract_token_metrics(result["raw_output"])
        l_bar_list.append(metrics["l_bar"])
        l_min_list.append(metrics["l_min"])
        var_l_list.append(metrics["var_l"])
        H_bar_list.append(metrics["H_bar"])

        generated_answer = metrics["generated_text"] or row.get("response", "")
        r_ent = grounder.compute_grounding_ratio(generated_answer, row["context"])
        R_entity_list.append(r_ent)

        if (i + 1) % 10 == 0 or (i + 1) == len(batch_results):
            logger.info("  Processed %d / %d samples.", i + 1, len(batch_results))

    enriched = df.copy()
    enriched["l_bar"]    = l_bar_list
    enriched["l_min"]    = l_min_list
    enriched["var_l"]    = var_l_list
    enriched["H_bar"]    = H_bar_list
    enriched["R_entity"] = R_entity_list

    return enriched


# ===========================================================================
# ENTRY POINT
# ===========================================================================
def main() -> None:
    """
    Execute the full Phase-2 RAG hallucination detection pipeline.

    CLI Usage
    ---------
    # Real dataset (download first with download_ragtruth.py):
        python3 rag_truth_phase2.py --dataset ragtruth_dataset.jsonl

    # Synthetic stress-test with 18 000 mock samples:
        python3 rag_truth_phase2.py --mock-samples 18000

    # Batched inference (safer on large runs):
        python3 rag_truth_phase2.py --mock-samples 18000 --batch-size 512

    # Skip latency benchmark on large runs:
        python3 rag_truth_phase2.py --mock-samples 18000 --no-benchmark
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="rag_truth_phase2",
        description="RAG Truth Phase-2 intrinsic hallucination detection pipeline.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ragtruth_dataset.jsonl",
        help="Path to a .jsonl dataset file. Falls back to mock data if absent.",
    )
    parser.add_argument(
        "--mock-samples",
        type=int,
        default=None,
        metavar="N",
        help="Skip dataset loading and generate N synthetic samples instead.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        metavar="B",
        help="Process inference in chunks of B prompts (0 = single batch, default).",
    )
    parser.add_argument(
        "--no-benchmark",
        action="store_true",
        default=False,
        help="Skip the latency micro-benchmark (useful for large dataset runs).",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  RAG Truth -- Phase 2 Pipeline Initialising...")
    print("=" * 60 + "\n")

    # ── Step 1: Dataset ────────────────────────────────────────────────────
    loader = RAGTruthLoader()
    if args.mock_samples is not None:
        logger.info("Generating %d synthetic mock samples...", args.mock_samples)
        df = loader.generate_mock_data(num_samples=args.mock_samples)
    else:
        df = loader.load_data(args.dataset)

    logger.info("Dataset shape: %s  |  Label distribution:\n%s",
                df.shape, df["Label"].value_counts().to_string())

    # ── Step 2: Component instantiation ────────────────────────────────────
    extractor = IntrinsicMetricsExtractor()
    grounder  = EntityGroundingEvaluator()
    engine    = vLLMInferenceEngine()
    detector  = CompositeHallucinationDetector()

    # ── Step 3: Feature extraction (chunked if --batch-size supplied) ──────
    batch_size = args.batch_size
    if batch_size > 0 and len(df) > batch_size:
        logger.info(
            "Chunked mode: %d samples in batches of %d.",
            len(df), batch_size,
        )
        chunks: list = []
        for start in range(0, len(df), batch_size):
            chunk_df = df.iloc[start : start + batch_size].reset_index(drop=True)
            logger.info(
                "  Batch %d-%d / %d...",
                start + 1, min(start + batch_size, len(df)), len(df),
            )
            enriched_chunk = build_feature_dataframe(chunk_df, engine, extractor, grounder)
            chunks.append(enriched_chunk)
        enriched_df = pd.concat(chunks, ignore_index=True)
    else:
        enriched_df = build_feature_dataframe(df, engine, extractor, grounder)

    # ── Step 4: Feature statistics ─────────────────────────────────────────
    print("\n-- Feature Statistics " + "-" * 38)
    print(enriched_df[["l_bar", "l_min", "var_l", "H_bar", "R_entity", "Label"]].describe().round(4))
    print()

    # ── Step 5: Train & evaluate composite classifier ──────────────────────
    detector.train_and_evaluate(enriched_df)

    # ── Step 6: Latency micro-benchmark ────────────────────────────────────
    if not args.no_benchmark:
        profiler = LatencyProfiler(extractor, grounder, detector, engine)
        profiler.run_benchmark()
    else:
        logger.info("Latency benchmark skipped (--no-benchmark).")    

    print("Pipeline complete. All stages finished successfully.\n")


if __name__ == "__main__":
    main()
