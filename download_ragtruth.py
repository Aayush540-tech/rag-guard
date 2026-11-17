"""
download_ragtruth.py
--------------------
Downloads the RAGTruth dataset from Hugging Face and converts it to the
ragtruth_dataset.jsonl format expected by rag_truth_phase2.py.

Usage:
    python3 download_ragtruth.py
"""

import json
import os

def download_and_convert():
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: Run  pip3 install datasets  first.")
        return

    print("Downloading RAGTruth from Hugging Face (ShengdingHu/RAGTruth)...")
    try:
        ds = load_dataset("ShengdingHu/RAGTruth", trust_remote_code=True)
    except Exception as e:
        print(f"Primary repo failed ({e}), trying alternate name...")
        try:
            ds = load_dataset("RAGTruth/RAGTruth", trust_remote_code=True)
        except Exception as e2:
            print(f"Alternate also failed ({e2}).")
            print("Falling back to manual JSONL generation with 18 000 mock samples.")
            _generate_large_mock(18000)
            return

    out_path = "ragtruth_dataset.jsonl"
    count = 0

    with open(out_path, "w", encoding="utf-8") as fh:
        for split_name in ds.keys():
            split = ds[split_name]
            print(f"  Processing split '{split_name}' ({len(split)} rows)...")
            for row in split:
                # Normalise field names – RAGTruth uses varying schemas
                record = {
                    "id":       str(row.get("id", row.get("qid", count))),
                    "prompt":   row.get("question", row.get("prompt", "")),
                    "context":  row.get("context",  row.get("passage", "")),
                    "response": row.get("answer",   row.get("response", "")),
                    "labels":   _parse_labels(row),
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

    print(f"\nSaved {count} samples to '{out_path}'.")
    print("Now run:  python3 rag_truth_phase2.py")


def _parse_labels(row: dict) -> list:
    """Try several known RAGTruth label formats."""
    # Format 1: pre-computed binary
    if "label" in row:
        val = row["label"]
        hallucinated = bool(val) if isinstance(val, (int, bool)) else (str(val).lower() in ("1", "true", "hallucinated"))
        return [{"start": 0, "end": 1, "hallucinated": hallucinated}] if hallucinated else []

    # Format 2: span list
    if "labels" in row and isinstance(row["labels"], list):
        return row["labels"]

    # Format 3: hallucination_spans
    if "hallucination_spans" in row:
        spans = row["hallucination_spans"]
        if spans:
            return [{"start": s.get("start", 0), "end": s.get("end", 1), "hallucinated": True}
                    for s in (spans if isinstance(spans, list) else [spans])]

    return []


def _generate_large_mock(n: int):
    """Generate n mock samples directly into ragtruth_dataset.jsonl."""
    import sys
    sys.path.insert(0, ".")
    from rag_truth_phase2 import RAGTruthLoader
    import logging
    logging.disable(logging.CRITICAL)

    loader = RAGTruthLoader()
    df = loader.generate_mock_data(num_samples=n)
    out_path = "ragtruth_dataset.jsonl"
    with open(out_path, "w") as fh:
        for _, row in df.iterrows():
            fh.write(row.to_json() + "\n")
    print(f"Generated {n} mock samples -> '{out_path}'")


if __name__ == "__main__":
    download_and_convert()
