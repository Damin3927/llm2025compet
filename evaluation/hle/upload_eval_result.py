"""
Upload judged evaluation results to the Hugging Face Hub.

Input format:
- A JSON file that is a dict keyed by question id.
- Each value is expected to contain at least:
  { "model": str, "response": str, "usage": dict, "judge_response": {...} }

The judge_response is expected to contain keys like:
  { "correct_answer": str, "model_answer": str, "reasoning": str,
    "correct": "yes"|"no", "confidence": int }

This script converts the JSON dict to a tabular dataset and pushes it.
"""

import argparse
import json
import os
from typing import Dict, Any, List

import numpy as np
from datasets import Dataset
from huggingface_hub import HfApi


def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Input JSON must be a dict keyed by id")
    return data


def flatten_records(data: Dict[str, Any]) -> Dict[str, List[Any]]:
    ids = sorted(list(data.keys()))
    # Build columns
    col_id: List[str] = []
    col_model: List[Any] = []
    col_response: List[Any] = []
    col_usage: List[str] = []

    # Judge columns
    col_judge_correct_answer: List[Any] = []
    col_judge_model_answer: List[Any] = []
    col_judge_reasoning: List[Any] = []
    col_judge_correct: List[Any] = []
    col_judge_confidence: List[Any] = []

    # Also keep raw judge_response as JSON string for convenience
    col_judge_response_raw: List[str] = []

    for i in ids:
        rec = data.get(i, {}) or {}
        jr = rec.get("judge_response", {}) or {}

        col_id.append(i)
        col_model.append(rec.get("model"))
        col_response.append(rec.get("response"))
        col_usage.append(json.dumps(rec.get("usage", {}), ensure_ascii=False))

        col_judge_correct_answer.append(jr.get("correct_answer"))
        col_judge_model_answer.append(jr.get("model_answer"))
        col_judge_reasoning.append(jr.get("reasoning"))
        col_judge_correct.append(jr.get("correct"))
        col_judge_confidence.append(jr.get("confidence"))
        col_judge_response_raw.append(json.dumps(jr, ensure_ascii=False))

    rows = {
        "id": col_id,
        "model": col_model,
        "response": col_response,
        "usage": col_usage,
        "judge_correct_answer": col_judge_correct_answer,
        "judge_model_answer": col_judge_model_answer,
        "judge_reasoning": col_judge_reasoning,
        "judge_correct": col_judge_correct,
        "judge_confidence": col_judge_confidence,
        "judge_response": col_judge_response_raw,
    }
    return rows


def calib_err(confidence: np.ndarray, correct: np.ndarray, p: str = "2", beta: int = 100) -> float:
    """Calibration error (same formulation as in eval.py)."""
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    if len(confidence) == 0:
        return 0.0
    bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta or 1)]
    bins[-1] = [bins[-1][0], len(confidence)]

    cerr = 0.0
    total_examples = float(len(confidence))
    for i in range(len(bins) - 1):
        b0, b1 = bins[i][0], bins[i][1]
        bin_confidence = confidence[b0:b1]
        bin_correct = correct[b0:b1]
        num_examples_in_bin = len(bin_confidence)

        if num_examples_in_bin > 0:
            difference = float(abs(np.nanmean(bin_confidence) - np.nanmean(bin_correct)))
            if p == '2':
                cerr += num_examples_in_bin / total_examples * (difference ** 2)
            elif p == '1':
                cerr += num_examples_in_bin / total_examples * difference
            elif p in ('infty', 'infinity', 'max'):
                cerr = float(max(cerr, difference))
            else:
                raise ValueError("p must be '1', '2', or 'infty'")

    if p == '2':
        cerr = float(np.sqrt(cerr))
    return float(cerr)


def compute_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    total = len(data)
    correct_flags: List[int] = []
    confidences: List[float] = []

    for _, rec in data.items():
        jr = (rec or {}).get("judge_response", {}) or {}
        correct_flags.append(1 if str(jr.get("correct", "")).lower() == "yes" else 0)
        conf = jr.get("confidence")
        try:
            conf = float(conf) if conf is not None else None
        except Exception:
            conf = None
        confidences.append(conf if conf is not None else float('nan'))

    correct_arr = np.array(correct_flags, dtype=float)
    conf_arr = np.array(confidences, dtype=float) / 100.0

    accuracy = float(np.nanmean(correct_arr)) if total > 0 else 0.0
    avg_conf = float(np.nanmean(conf_arr)) if total > 0 else 0.0
    calib = calib_err(conf_arr, correct_arr, p='2', beta=100) if total > 0 else 0.0

    # 95% Wald interval on percentage
    acc_pct = accuracy * 100.0
    half_width = 1.96 * float(np.sqrt(acc_pct * (100.0 - acc_pct) / max(total, 1)))

    return {
        "total": total,
        "correct": int(np.nansum(correct_arr)),
        "incorrect": int(total - np.nansum(correct_arr)),
        "accuracy_pct": round(acc_pct, 4),
        "accuracy_ci95_half_width_pct": round(half_width, 4),
        "avg_confidence_pct": round(avg_conf * 100.0, 4),
        "calibration_error_pct": round(calib * 100.0, 4),
    }


def build_readme(repo_id: str, metrics: Dict[str, Any], models: List[str]) -> str:
    lines = [
        f"# {repo_id}",
        "",
        "Judged HLE evaluation results uploaded via script.",
        "",
        "## Summary metrics",
        "",
        f"- Total examples: {metrics.get('total')}",
        f"- Accuracy: {metrics.get('accuracy_pct')}% ± {metrics.get('accuracy_ci95_half_width_pct')}%",
        f"- Average confidence: {metrics.get('avg_confidence_pct')}%",
        f"- Calibration error (L2): {metrics.get('calibration_error_pct')}%",
        "",
        "## Models",
        "",
        f"- {', '.join(models) if models else 'N/A'}",
        "",
        "## Columns",
        "",
        "- id",
        "- model",
        "- response",
        "- usage (JSON string)",
        "- judge_correct_answer",
        "- judge_model_answer",
        "- judge_reasoning",
        "- judge_correct",
        "- judge_confidence",
        "- judge_response (raw JSON)",
        "",
        "Generated by evaluation/hle/upload_eval_result.py",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Upload judged evaluation results to Hugging Face Hub")
    parser.add_argument("--input_json", type=str, required=True, help="Path to judged_*.json produced by eval.py")
    parser.add_argument("--dataset_name", type=str, required=True, help="Target dataset repo (e.g., user/hle-judged-llama3-8b)")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token")
    args = parser.parse_args()

    data = load_json(args.input_json)
    rows = flatten_records(data)
    ds = Dataset.from_dict(rows)
    print(f"Prepared dataset with {len(ds)} rows from: {args.input_json}")

    # Compute metrics and gather model list
    metrics = compute_metrics(data)
    models_set: set[str] = set()
    for k, rec in data.items():
        model_val = (rec or {}).get('model')
        if isinstance(model_val, str) and model_val:
            models_set.add(model_val)
    models = sorted(models_set)

    # Ensure public repo exists and upload README + metrics.json
    api = HfApi()
    print(f"Creating (or ensuring) public dataset repo: {args.dataset_name}")
    api.create_repo(repo_id=args.dataset_name, repo_type="dataset", private=False, exist_ok=True, token=args.hf_token)

    readme_text = build_readme(args.dataset_name, metrics, models)
    metrics_text = json.dumps(metrics, indent=2)

    print("Uploading README.md and metrics.json...")
    api.upload_file(
        repo_id=args.dataset_name,
        repo_type="dataset",
        path_or_fileobj=readme_text.encode("utf-8"),
        path_in_repo="README.md",
        token=args.hf_token,
    )
    api.upload_file(
        repo_id=args.dataset_name,
        repo_type="dataset",
        path_or_fileobj=metrics_text.encode("utf-8"),
        path_in_repo="metrics.json",
        token=args.hf_token,
    )

    print(f"Pushing dataset to Hub: {args.dataset_name} (public)")
    ds.push_to_hub(args.dataset_name, token=args.hf_token, private=False)
    print("Upload completed.")


if __name__ == "__main__":
    main()
