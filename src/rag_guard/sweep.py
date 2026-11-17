"""
rag_guard.sweep
---------------
Threshold sensitivity sweep evaluation for RapidFuzz entity matching.
Evaluates tau_fuzzy in {0.75, 0.80, 0.85, 0.90, 0.95} (or 75.0 to 95.0) across
overall dataset, Evident Conflict, and Baseless Extrapolation splits.
Identifies the optimal threshold maximizing composite ROC-AUC and F1-score.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from rag_guard.config import DEFAULT_FUZZY_THRESHOLD, FEATURE_NAMES
from rag_guard.detector import CompositeHallucinationDetector
from rag_guard.grounding import EntityGroundingEvaluator
from rag_guard.intrinsic import IntrinsicMetrics, IntrinsicMetricsExtractor

logger = logging.getLogger("rag_guard.sweep")

# Default sweep thresholds evaluated in Phase 3
DEFAULT_SWEEP_THRESHOLDS: List[float] = [75.0, 80.0, 85.0, 90.0, 95.0]


@dataclass(frozen=True)
class ThresholdResult:
    """Evaluation metrics for a specific fuzzy matching threshold tau."""

    threshold: float
    split_name: str
    num_samples: int
    r_entity_auc: float
    composite_auc: float
    f1: float
    precision: float
    recall: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "threshold": self.threshold,
            "split_name": self.split_name,
            "num_samples": self.num_samples,
            "r_entity_auc": round(self.r_entity_auc, 4),
            "composite_auc": round(self.composite_auc, 4),
            "f1": round(self.f1, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
        }


@dataclass(frozen=True)
class SweepSummary:
    """Complete results of threshold sensitivity sweep."""

    results_by_split: Dict[str, List[ThresholdResult]]
    optimal_threshold_overall: float
    optimal_composite_auc: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire sweep summary to dictionary."""
        return {
            "optimal_threshold_overall": self.optimal_threshold_overall,
            "optimal_composite_auc": round(self.optimal_composite_auc, 4),
            "results_by_split": {
                split: [res.to_dict() for res in res_list]
                for split, res_list in self.results_by_split.items()
            },
            "metadata": self.metadata,
        }

    def save_json(self, output_path: Union[str, Path]) -> None:
        """Serialize sweep summary to JSON file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        logger.info("Saved sweep results to '%s'", out)


def generate_hallucination_split_dataset(
    num_samples: int = 1500,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic dataset annotated by hallucination split:
      - 'grounded': Faithful to context (Label = 0, Faithful = 1)
      - 'evident_conflict': Directly contradicts context entities/numbers (Label = 1, Faithful = 0)
      - 'baseless_extrapolation': Introduces external unsupported entities (Label = 1, Faithful = 0)
    """
    rng = np.random.default_rng(seed=random_state)
    per_split = num_samples // 3

    # Templates for grounded samples
    grounded_contexts = [
        "In Q3 2023, Tesla delivered 435,059 vehicles and generated $23.35B in revenue.",
        "Microsoft acquired GitHub in 2018 for $7.5 billion in stock.",
        "The Chandrayaan-3 mission touched down on the Moon on August 23, 2023 at 18:04 IST.",
        "Google DeepMind released Gemini 1.5 Pro featuring a 1,000,000 token context window.",
    ]
    grounded_answers = [
        "Tesla delivered 435,059 vehicles and recorded $23.35B revenue in Q3 2023.",
        "Microsoft completed its purchase of GitHub in 2018 for $7.5B in stock.",
        "Chandrayaan-3 achieved lunar touchdown on August 23, 2023 at 18:04 IST.",
        "Gemini 1.5 Pro from Google DeepMind offers a context window of 1,000,000 tokens.",
    ]

    # Templates for evident conflict (direct numerical/factual contradiction)
    conflict_contexts = [
        "Pfizer reported FY23 revenue of $58.5B, down 42% operational year-over-year.",
        "Boeing 787 Dreamliner has a maximum passenger capacity of 336 in typical 2-class seating.",
        "The Empire State Building stands 381 metres tall and has 102 floors.",
        "The James Webb Space Telescope primary mirror has a diameter of 6.5 metres.",
    ]
    conflict_answers = [
        "Pfizer reported FY23 revenue of $85.5B, up 24% operational year-over-year.",
        "Boeing 787 Dreamliner accommodates up to 550 passengers in typical 2-class seating.",
        "The Empire State Building stands 520 metres tall and has 145 floors.",
        "The James Webb Space Telescope primary mirror has a diameter of 12.0 metres.",
    ]

    # Templates for baseless extrapolation (unsupported external claims)
    extrap_contexts = [
        "OpenAI announced GPT-4 on March 14, 2023. It passed a simulated bar exam with a score in the top 10%.",
        "Meta released Llama 2 in July 2023 with weights made freely available for research and commercial use.",
        "The CERN Large Hadron Collider is 27 kilometres in circumference and is located near Geneva.",
        "Anthropic was founded in 2021 by former OpenAI executives Dario Amodei and Daniela Amodei.",
    ]
    extrap_answers = [
        "OpenAI announced GPT-4 on March 14, 2023, and signed a $10B exclusive defense contract with DARPA.",
        "Meta released Llama 2 in July 2023, claiming it surpassed GPT-4 on all 50 enterprise benchmarks.",
        "The CERN Large Hadron Collider is 27 km in circumference and confirmed supersymmetric dark matter particles in 2022.",
        "Anthropic was founded in 2021 by Dario Amodei and Daniela Amodei, securing $15B from sovereign wealth funds.",
    ]

    records = []

    # 1. Grounded controls
    for i in range(per_split):
        idx = i % len(grounded_contexts)
        l_bar = float(rng.normal(-0.18, 0.10))
        l_min = float(min(l_bar, rng.normal(-0.60, 0.15)))
        var_l = float(rng.gamma(1.5, 0.04))
        h_bar = float(rng.gamma(1.2, 0.06))
        records.append({
            "context": grounded_contexts[idx],
            "response": grounded_answers[idx],
            "l_bar": l_bar,
            "l_min": l_min,
            "var_l": var_l,
            "H_bar": h_bar,
            "split": "grounded",
            "Faithful": 1,
            "Label": 0,
        })

    # 2. Evident Conflict
    for i in range(per_split):
        idx = i % len(conflict_contexts)
        l_bar = float(rng.normal(-1.25, 0.35))
        l_min = float(min(l_bar, rng.normal(-2.80, 0.50)))
        var_l = float(rng.gamma(3.0, 0.25))
        h_bar = float(rng.gamma(3.5, 0.20))
        records.append({
            "context": conflict_contexts[idx],
            "response": conflict_answers[idx],
            "l_bar": l_bar,
            "l_min": l_min,
            "var_l": var_l,
            "H_bar": h_bar,
            "split": "evident_conflict",
            "Faithful": 0,
            "Label": 1,
        })

    # 3. Baseless Extrapolation
    for i in range(num_samples - 2 * per_split):
        idx = i % len(extrap_contexts)
        l_bar = float(rng.normal(-1.60, 0.40))
        l_min = float(min(l_bar, rng.normal(-3.40, 0.60)))
        var_l = float(rng.gamma(3.8, 0.30))
        h_bar = float(rng.gamma(4.8, 0.25))
        records.append({
            "context": extrap_contexts[idx],
            "response": extrap_answers[idx],
            "l_bar": l_bar,
            "l_min": l_min,
            "var_l": var_l,
            "H_bar": h_bar,
            "split": "baseless_extrapolation",
            "Faithful": 0,
            "Label": 1,
        })

    df = pd.DataFrame(records).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df


def evaluate_split_at_threshold(
    df: pd.DataFrame,
    threshold: float,
    detector: CompositeHallucinationDetector,
    split_name: str,
) -> ThresholdResult:
    """Evaluate grounding ratio and composite faithfulness for a dataset slice at threshold tau."""
    grounder = EntityGroundingEvaluator(fuzzy_threshold=threshold)

    r_entity_list = []
    composite_scores = []
    y_true = df["Faithful"].values.astype(int)

    for _, row in df.iterrows():
        r_ent = grounder.compute_grounding_ratio(row["response"], row["context"])
        r_entity_list.append(r_ent)

        feat_vec = [row["l_bar"], row["l_min"], row["var_l"], row["H_bar"], r_ent]
        res = detector.evaluate_vector(feat_vec)
        composite_scores.append(res.faithfulness_score)

    r_arr = np.asarray(r_entity_list, dtype=np.float64)
    s_arr = np.asarray(composite_scores, dtype=np.float64)

    # Compute metrics
    # If single class in split, AUC is undefined; handle safely
    if len(np.unique(y_true)) > 1:
        r_auc = float(roc_auc_score(y_true, r_arr))
        comp_auc = float(roc_auc_score(y_true, s_arr))
    else:
        r_auc = 1.0 if y_true[0] == 1 else 0.0
        comp_auc = 1.0 if y_true[0] == 1 else 0.0

    preds = (s_arr >= detector.threshold).astype(int)
    f1 = float(f1_score(y_true, preds, zero_division=0))
    prec = float(precision_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds, zero_division=0))

    return ThresholdResult(
        threshold=threshold,
        split_name=split_name,
        num_samples=len(df),
        r_entity_auc=r_auc,
        composite_auc=comp_auc,
        f1=f1,
        precision=prec,
        recall=rec,
    )


def run_threshold_sweep(
    df: pd.DataFrame,
    thresholds: Sequence[float] = DEFAULT_SWEEP_THRESHOLDS,
    detector: Optional[CompositeHallucinationDetector] = None,
) -> SweepSummary:
    """
    Run threshold sweep across all candidates and splits.

    Splits evaluated:
      - 'overall': Full dataset (grounded + conflict + extrapolation)
      - 'evident_conflict_vs_grounded': Grounded controls + Evident Conflict samples
      - 'baseless_extrapolation_vs_grounded': Grounded controls + Baseless Extrapolation samples
    """
    det = detector or CompositeHallucinationDetector()

    splits_to_eval = {
        "overall": df,
    }

    if "split" in df.columns:
        grounded_df = df[df["split"] == "grounded"]
        conflict_df = df[df["split"] == "evident_conflict"]
        extrap_df = df[df["split"] == "baseless_extrapolation"]

        if not conflict_df.empty and not grounded_df.empty:
            splits_to_eval["evident_conflict"] = pd.concat([grounded_df, conflict_df], ignore_index=True)

        if not extrap_df.empty and not grounded_df.empty:
            splits_to_eval["baseless_extrapolation"] = pd.concat([grounded_df, extrap_df], ignore_index=True)

    results_by_split: Dict[str, List[ThresholdResult]] = {split: [] for split in splits_to_eval}

    for tau in thresholds:
        logger.info("Evaluating tau_fuzzy = %.1f ...", tau)
        for split_name, split_df in splits_to_eval.items():
            res = evaluate_split_at_threshold(split_df, tau, det, split_name)
            results_by_split[split_name].append(res)

    # Find optimal threshold on overall split
    overall_results = results_by_split["overall"]
    best_res = max(overall_results, key=lambda r: (r.composite_auc, r.f1))

    return SweepSummary(
        results_by_split=results_by_split,
        optimal_threshold_overall=best_res.threshold,
        optimal_composite_auc=best_res.composite_auc,
        metadata={
            "num_samples": len(df),
            "thresholds_evaluated": list(thresholds),
            "detector_threshold": det.threshold,
        },
    )


def print_sweep_report(summary: SweepSummary) -> None:
    """Format and print standard academic threshold sensitivity report."""
    sep_full = "=" * 76
    sep_light = "-" * 76

    print(f"\n{sep_full}")
    print(f"{'RapidFuzz Threshold Sensitivity Sweep (tau_fuzzy in [75.0, 95.0])':^76}")
    print(sep_full)

    for split_name, res_list in summary.results_by_split.items():
        print(f"\nSplit: {split_name.upper()} (N = {res_list[0].num_samples})")
        print(sep_light)
        print(f"{'tau_fuzzy':<12}{'R_entity AUC':>14}{'Composite AUC':>16}{'F1-Score':>12}{'Precision':>11}{'Recall':>11}")
        print(sep_light)
        for r in res_list:
            is_best = (r.threshold == summary.optimal_threshold_overall and split_name == "overall")
            marker = " *" if is_best else ""
            print(
                f"{r.threshold:<12.1f}"
                f"{r.r_entity_auc:>14.4f}"
                f"{r.composite_auc:>16.4f}"
                f"{r.f1:>12.4f}"
                f"{r.precision:>11.4f}"
                f"{r.recall:>11.4f}"
                f"{marker}"
            )
        print(sep_light)

    print(f"\nRecommended Operational Threshold: tau_fuzzy = {summary.optimal_threshold_overall:.1f}")
    print(f"Optimal Composite ROC-AUC       : {summary.optimal_composite_auc:.4f}")
    print(f"(* indicates optimal overall operating point)")
    print(f"{sep_full}\n")


def main() -> None:
    """CLI entry point for threshold sensitivity sweep."""
    parser = argparse.ArgumentParser(
        prog="rag_guard.sweep",
        description="Threshold sensitivity sweep across Evident Conflict and Baseless Extrapolation.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset file with context, response, and intrinsic columns.",
    )
    parser.add_argument(
        "--mock-samples",
        type=int,
        default=300,
        help="Number of synthetic samples with split annotations (default: 300).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save sweep results JSON.",
    )
    args = parser.parse_args()

    if args.dataset and Path(args.dataset).exists():
        logger.info("Loading dataset from '%s'...", args.dataset)
        df = pd.read_json(args.dataset, lines=True) if args.dataset.endswith(".jsonl") else pd.read_csv(args.dataset)
    else:
        logger.info("Generating %d synthetic samples with split breakdown...", args.mock_samples)
        df = generate_hallucination_split_dataset(num_samples=args.mock_samples)

    summary = run_threshold_sweep(df)
    print_sweep_report(summary)

    if args.output_json:
        summary.save_json(args.output_json)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
