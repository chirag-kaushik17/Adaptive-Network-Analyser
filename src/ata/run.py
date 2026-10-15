from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .baseline import AdaptiveBaseline, StaticBaseline
from .features import FEATURE_COLUMNS, window_features
from .loader import load

MANUAL_CIC_INSTRUCTIONS = """Manual download required for CIC-IDS2017.
Download the TrafficLabelling archive from:
https://www.unb.ca/cic/datasets/ids-2017.html
Then extract the CSV files into the data/ directory, for example:
  data/Monday-WorkingHours.pcap_ISCX.csv
  data/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
"""


def summarize_evaluation(y_true, y_pred):
    y_true = [str(v).lower() for v in y_true]
    y_pred = [bool(v) for v in y_pred]
    tp = sum(1 for actual, pred in zip(y_true, y_pred) if actual != "normal" and pred)
    fp = sum(1 for actual, pred in zip(y_true, y_pred) if actual == "normal" and pred)
    fn = sum(1 for actual, pred in zip(y_true, y_pred) if actual != "normal" and not pred)
    actual_normal = sum(1 for v in y_true if v == "normal")
    precision_denom = tp + fp
    recall_denom = tp + fn
    precision = (tp / precision_denom) if precision_denom else "n/a"
    recall = (tp / recall_denom) if recall_denom else "n/a"
    false_alarm_rate = (fp / actual_normal) if actual_normal else "n/a"
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "false_alarm_rate": false_alarm_rate,
    }


def evaluate_csv(csv_path: str | Path, dataset: str, window: int = 1, warmup: int = 3, baseline_name: str = "adaptive", seed: int = 7):
    path = Path(csv_path)
    frame = load(path, dataset)
    windows = window_features(frame, window=window)
    if windows.empty:
        raise ValueError("No windows were generated from the provided CSV.")

    if baseline_name == "adaptive":
        model = AdaptiveBaseline(warmup=warmup, history=max(1, min(10, len(windows) - warmup)), refit_every=1, contamination=0.05, seed=seed)
    elif baseline_name == "static":
        model = StaticBaseline(warmup=warmup, history=max(1, min(10, len(windows) - warmup)), refit_every=1, contamination=0.05, seed=seed)
    else:
        raise ValueError(f"Unsupported baseline: {baseline_name!r}")

    results = model.run(windows[FEATURE_COLUMNS])
    post = results.iloc[warmup:] if warmup < len(results) else results.iloc[0:0]
    true_labels = windows["label"].iloc[warmup:] if warmup < len(windows) else pd.Series(dtype=str)
    metrics = summarize_evaluation(true_labels, post["flag"].tolist())
    return {
        "frame": frame,
        "windows": windows,
        "results": results,
        "post": post,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an adaptive or static isolation-forest baseline on network-flow windows.")
    parser.add_argument("--dataset", choices=["synth", "cicids2017"], required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--baseline", choices=["adaptive", "static"], default="adaptive")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if args.dataset == "cicids2017" and (not csv_path.exists() or not csv_path.is_file()):
        print(MANUAL_CIC_INSTRUCTIONS)
        print("CHECK: skipped (no valid CIC-IDS2017 CSV found).")
        return

    if args.dataset == "cicids2017":
        try:
            result = evaluate_csv(csv_path, args.dataset, window=args.window, warmup=args.warmup, baseline_name=args.baseline, seed=args.seed)
        except ValueError as exc:
            print(MANUAL_CIC_INSTRUCTIONS)
            print(f"CHECK: skipped ({exc})")
            return
    else:
        result = evaluate_csv(csv_path, args.dataset, window=args.window, warmup=args.warmup, baseline_name=args.baseline, seed=args.seed)

    frame = result["frame"]
    windows = result["windows"]
    metrics = result["metrics"]
    print(f"dataset: {csv_path.name}")
    print(f"time span: {frame['ts'].min():.2f}s to {frame['ts'].max():.2f}s")
    print(f"baseline: {args.baseline} (warmup={args.warmup}, window={args.window}, seed={args.seed})")
    print(f"total windows: {len(windows)}")
    print(f"attack windows: {int((windows['label'] != 'normal').sum())}")
    eval_windows = len(result["post"]) if not result["post"].empty else 0
    print(f"post-warm-up evaluation windows: {eval_windows}")
    print(f"TP: {metrics['tp']} FP: {metrics['fp']} FN: {metrics['fn']}")
    print(f"precision: {metrics['precision']}")
    print(f"recall: {metrics['recall']}")
    print(f"false-alarm rate: {metrics['false_alarm_rate']}")


if __name__ == "__main__":
    main()
