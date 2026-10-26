"""Evaluate the Layer 1 baseline on an existing flow CSV."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

from .baseline import AdaptiveBaseline, StaticBaseline
from .features import FEATURE_COLUMNS, window_features
from .labels import LABEL_FAMILY_ROWS
from .loader import InvalidCICHeaderError, load

MANUAL_CIC_INSTRUCTIONS = """Manual download required for CIC-IDS2017 TrafficLabelling CSVs.
Download from:
https://www.unb.ca/cic/datasets/ids-2017.html
Place the files in data/, for example:
  data/Monday-WorkingHours.pcap_ISCX.csv
  data/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"""
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULTS_PATH = PROJECT_ROOT / "configs" / "defaults.json"


def _default_contamination() -> float:
    try:
        defaults = json.loads(DEFAULTS_PATH.read_text(encoding="utf-8"))
        return float(defaults["contamination"])
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to read contamination from {DEFAULTS_PATH}: {exc}") from exc


def _default_drift_options() -> dict:
    try:
        defaults = json.loads(DEFAULTS_PATH.read_text(encoding="utf-8"))
        return defaults["drift"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to read drift settings from {DEFAULTS_PATH}: {exc}") from exc


def summarize_evaluation(y_true, y_pred):
    """Compute binary detection metrics; zero-denominator values are n/a."""
    y_true = [str(value).lower() for value in y_true]
    y_pred = [bool(value) for value in y_pred]
    tp = sum(actual != "normal" and predicted for actual, predicted in zip(y_true, y_pred))
    fp = sum(actual == "normal" and predicted for actual, predicted in zip(y_true, y_pred))
    fn = sum(actual != "normal" and not predicted for actual, predicted in zip(y_true, y_pred))
    actual_normal = sum(value == "normal" for value in y_true)
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": tp / precision_denominator if precision_denominator else "n/a",
        "recall": tp / recall_denominator if recall_denominator else "n/a",
        "false_alarm_rate": fp / actual_normal if actual_normal else "n/a",
    }


def _read_sidecar(csv_path: Path) -> dict | None:
    sidecar = csv_path.with_suffix(csv_path.suffix + ".meta.json")
    if not sidecar.is_file():
        return None
    return json.loads(sidecar.read_text(encoding="utf-8"))


def evaluate_csv(
    csv_path: str | Path,
    dataset: str,
    window: int = 1,
    warmup: int = 120,
    history: int = 300,
    refit_every: int | None = 30,
    baseline_name: str | None = None,
    layer1_mode: str | None = None,
    model_seed: int = 0,
    contamination: float | None = None,
    date_order: str = "auto",
    allow_any_dates: bool = False,
    refit_trigger: str = "periodic",
    drift_method: str | None = None,
):
    """Evaluate a detector against one CSV without changing or generating it."""
    if layer1_mode is not None:
        if layer1_mode == "static":
            resolved_baseline = "static"
        elif layer1_mode == "adaptive":
            resolved_baseline = "adaptive"
            refit_trigger = "periodic"
        elif layer1_mode == "drift":
            resolved_baseline = "adaptive"
            refit_trigger = "drift"
        else:
            raise ValueError(f"Unsupported layer1_mode: {layer1_mode!r}")
    elif baseline_name is not None:
        resolved_baseline = baseline_name
    else:
        resolved_baseline = "static"

    path = Path(csv_path)
    flows, load_stats = load(
        path,
        dataset,
        return_stats=True,
        date_order=date_order,
        allow_any_dates=allow_any_dates,
    )
    windows = window_features(flows, window=window)
    if windows.empty:
        raise ValueError("No windows were generated from the provided CSV.")
    if warmup > len(windows):
        raise ValueError(
            f"Insufficient windows for warm-up: received {len(windows)}, need {warmup}."
        )
    if contamination is None:
        contamination = _default_contamination()
    drift_options = _default_drift_options()
    common = {
        "warmup": warmup,
        "history": history,
        "refit_every": refit_every,
        "contamination": contamination,
        "seed": model_seed,
        "refit_trigger": refit_trigger,
        "drift_method": drift_method or drift_options["method"],
        "psi_threshold": drift_options["psi_threshold"],
        "min_drift_features": drift_options["min_features"],
        "drift_persistence": drift_options["persistence"],
        "test_windows": drift_options["test_windows"],
        "check_every": drift_options["check_every"],
        "drift_cooldown": drift_options["cooldown"],
    }
    if resolved_baseline == "adaptive":
        model = AdaptiveBaseline(**common)
    elif resolved_baseline == "static":
        model = StaticBaseline(**common)
    else:
        raise ValueError(f"Unsupported baseline: {resolved_baseline!r}")

    results = model.run(windows[FEATURE_COLUMNS])
    post = results.iloc[warmup:]
    true_labels = windows["label"].iloc[warmup:]
    metrics = summarize_evaluation(true_labels, post["flag"].tolist())

    warmup_attacks = int(windows["label"].iloc[:warmup].ne("normal").sum())
    if warmup_attacks:
        warnings.warn(
            f"Ground truth contains {warmup_attacks} attack windows inside warm-up.",
            UserWarning,
            stacklevel=2,
        )
    sidecar = _read_sidecar(path) if dataset == "synth" else None
    if sidecar:
        warmup_seconds = warmup * window
        for onset_key in ("attack_start", "drift_start", "stealth_start"):
            onset = sidecar.get(onset_key)
            if onset is not None and float(onset) < warmup_seconds:
                warnings.warn(
                    f"Synthetic metadata reports {onset_key}={onset} before "
                    f"warm-up ends at {warmup_seconds:g}s.",
                    UserWarning,
                    stacklevel=2,
                )

    return {
        "frame": flows,
        "windows": windows,
        "results": results,
        "post": post,
        "metrics": metrics,
        "load_stats": load_stats,
        "model": model,
        "baseline_config": common,
        "warmup_attack_windows": warmup_attacks,
        "sidecar": sidecar,
    }


def _print_result(csv_path: Path, dataset: str, window: int, baseline: str, output: dict) -> None:
    frame = output["frame"]
    windows = output["windows"]
    stats = output["load_stats"]
    metrics = output["metrics"]
    config = output["baseline_config"]
    periodic_refit = (
        f"every {config['refit_every']} windows/"
        f"{config['refit_every'] * window}s"
        if config["refit_trigger"] == "periodic"
        and config["refit_every"] is not None
        else "disabled"
    )
    print(f"dataset: {csv_path.name} ({dataset})")
    print(f"time span: {frame['ts'].min():.2f}s to {frame['ts'].max():.2f}s")
    print(
        f"baseline: {baseline} (warmup={config['warmup']} windows/"
        f"{config['warmup'] * window}s, history={config['history']} windows/"
        f"{config['history'] * window}s, "
        f"refit_trigger={config['refit_trigger']}, "
        f"periodic_refit={periodic_refit}, "
        f"drift_method={config['drift_method']}, "
        f"contamination={config['contamination']}, model_seed={config['seed']})"
    )
    print(f"total windows: {len(windows)}")
    print(f"attack windows: {int(windows['label'].ne('normal').sum())}")
    print(f"post-warm-up evaluation windows: {len(output['post'])}")
    print(f"discarded invalid timestamps: {stats['discarded_invalid_timestamps']}")
    if dataset == "cicids2017":
        print(f"date order used: {stats['date_order_used']}")
        print(f"date range: {stats['date_range'][0]} to {stats['date_range'][1]}")
        print("label-family mapping:")
        for raw_label, family in LABEL_FAMILY_ROWS:
            print(f"  {raw_label} -> {family}")
        print(f"label-family counts: {stats['label_counts']}")
        if stats["unknown_raw_labels"]:
            print(f"unknown raw labels mapped to other: {stats['unknown_raw_labels']}")
    print(f"TP: {metrics['tp']} FP: {metrics['fp']} FN: {metrics['fn']}")
    print(f"precision: {metrics['precision']}")
    print(f"recall: {metrics['recall']}")
    print(f"false-alarm rate: {metrics['false_alarm_rate']}")
    model = output["model"]
    print(
        f"refits: {model.n_refits} skipped refits: {model.skipped_refits}"
    )
    events = model.drift_events
    first_event = events[0]["position"] if events else None
    delay = None
    sidecar = output["sidecar"]
    if first_event is not None and sidecar and sidecar.get("drift_start") is not None:
        drift_start_window = float(sidecar["drift_start"]) / window
        delay = float(first_event) - drift_start_window
    print(f"drift events: {len(events)}")
    print(f"first drift event window: {first_event if first_event is not None else 'n/a'}")
    print(f"detection delay (windows): {delay if delay is not None else 'n/a'}")
    print(f"drift-triggered refits: {len(model.drift_refit_events)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an adaptive or static baseline on an existing flow CSV."
    )
    parser.add_argument("--dataset", choices=["synth", "cicids2017"], required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=120)
    parser.add_argument("--history", type=int, default=300)
    parser.add_argument("--refit-every", type=int, default=30)
    parser.add_argument("--refit-trigger", choices=["periodic", "drift"], default="periodic")
    parser.add_argument("--drift-method", choices=["psi", "ks"], default=None)
    parser.add_argument("--contamination", type=float, default=None)
    parser.add_argument(
        "--layer1-mode",
        choices=["static", "adaptive", "drift"],
        default=None,
        help="Layer 1 baseline mode (default: static)",
    )
    parser.add_argument(
        "--baseline",
        choices=["adaptive", "static"],
        default=None,
        help="Legacy baseline option",
    )
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--date-order", choices=["auto", "dmy", "mdy"], default="auto")
    parser.add_argument("--allow-any-dates", action="store_true")
    args = parser.parse_args()
    csv_path = Path(args.csv)

    if args.dataset == "cicids2017" and not csv_path.is_file():
        print(MANUAL_CIC_INSTRUCTIONS)
        print("CHECK: SKIPPED (CIC CSV is missing).")
        return

    if args.layer1_mode is not None:
        mode = args.layer1_mode
    elif args.baseline is not None:
        mode = args.baseline
    else:
        mode = "static"

    refit_trigger = args.refit_trigger
    if mode == "drift":
        refit_trigger = "drift"

    try:
        output = evaluate_csv(
            csv_path,
            args.dataset,
            window=args.window,
            warmup=args.warmup,
            history=args.history,
            refit_every=args.refit_every,
            layer1_mode=mode,
            model_seed=args.model_seed,
            contamination=args.contamination,
            date_order=args.date_order,
            allow_any_dates=args.allow_any_dates,
            refit_trigger=refit_trigger,
            drift_method=args.drift_method,
        )
    except InvalidCICHeaderError as exc:
        print(MANUAL_CIC_INSTRUCTIONS)
        print(f"CHECK: SKIPPED (invalid CIC header: {exc})")
        return
    except Exception as exc:
        print(f"CHECK: FAILED ({type(exc).__name__}: {exc})", file=sys.stderr)
        raise SystemExit(1) from exc
    _print_result(csv_path, args.dataset, args.window, mode, output)
    print("CHECK: PASSED")


if __name__ == "__main__":
    main()
