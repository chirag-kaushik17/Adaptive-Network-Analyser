"""Synthetic evaluation harness for the Layer 1 baselines."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from .baseline import AdaptiveBaseline, StaticBaseline
from .features import FEATURE_COLUMNS, window_features
from .synth import generate_synthetic_csv
from .run import DEFAULTS_PATH
from tests.thresholds import (
    PHASE5_DRIFT_FAR_WINS_REQUIRED,
    PHASE5_DRIFT_RECALL_TOLERANCE,
    PHASE5_STABLE_FAR_TOLERANCE,
    PHASE5_STABLE_RECALL_TOLERANCE,
)

METHODS = ("static", "adaptive_periodic", "adaptive_drift")
SCENARIOS = ("stable_attacks", "drift_attacks")
COLORBLIND = {
    "static": "#0072B2",
    "adaptive_periodic": "#E69F00",
    "adaptive_drift": "#009E73",
}
ROOT = Path(__file__).resolve().parents[2]


def time_ordered_split(
    indices,
    train_fraction: float = 0.65,
    gap: int = 0,
) -> tuple[list, list]:
    """Split ordered indices by position, leaving an optional gap."""
    values = list(indices)
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between zero and one.")
    if gap < 0:
        raise ValueError("gap cannot be negative.")
    if any(left >= right for left, right in zip(values, values[1:])):
        raise ValueError("indices must be strictly increasing.")
    boundary = int(len(values) * train_fraction)
    test_start = boundary + gap
    if boundary < 1 or test_start >= len(values):
        raise ValueError("split leaves an empty training or test set.")
    return values[:boundary], values[test_start:]


def calculate_metrics(
    labels,
    predictions,
    *,
    window_indices=None,
    episodes=(),
    window_seconds: float = 1.0,
) -> dict:
    """Calculate window-level binary and per-attack metrics."""
    labels = list(labels)
    actual = [str(label).lower() != "normal" for label in labels]
    predicted = [bool(value) for value in predictions]
    if len(actual) != len(predicted):
        raise ValueError("labels and predictions must have the same length.")
    if window_indices is None:
        window_indices = list(range(len(actual)))
    else:
        window_indices = list(window_indices)
    if len(window_indices) != len(actual):
        raise ValueError("window_indices must match labels and predictions.")
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive.")

    tp = sum(a and p for a, p in zip(actual, predicted))
    fp = sum(not a and p for a, p in zip(actual, predicted))
    tn = sum(not a and not p for a, p in zip(actual, predicted))
    fn = sum(a and not p for a, p in zip(actual, predicted))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    false_alarm_rate = fp / (fp + tn) if fp + tn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None
        and recall is not None
        and precision + recall > 0
        else (0.0 if precision == 0 and recall == 0 else None)
    )

    attack_types = sorted(
        {str(label).lower() for label in labels if str(label).lower() != "normal"}
    )
    per_attack_recall = {}
    for attack_type in attack_types:
        members = [
            (str(label).lower() == attack_type, flag)
            for label, flag in zip(labels, predicted)
            if str(label).lower() == attack_type
        ]
        hits = sum(flag for _, flag in members)
        per_attack_recall[attack_type] = hits / len(members) if members else None

    prediction_by_index = dict(zip(window_indices, predicted))
    detection_delays = []
    for episode in episodes:
        start_index = int(np.floor(float(episode["start"]) / window_seconds))
        end_index = int(np.ceil(float(episode["end"]) / window_seconds))
        detected = [
            index
            for index in range(start_index, end_index)
            if prediction_by_index.get(index, False)
        ]
        detection_delays.append(
            {
                "type": str(episode["type"]),
                "start": float(episode["start"]),
                "end": float(episode["end"]),
                "delay_seconds": (
                    max(0.0, detected[0] * window_seconds - float(episode["start"]))
                    if detected
                    else None
                ),
            }
        )
    return {
        "precision": precision,
        "recall": recall,
        "false_alarm_rate": false_alarm_rate,
        "f1": f1,
        "confusion": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
        "per_attack_type_recall": per_attack_recall,
        "detection_delay_per_episode_seconds": detection_delays,
    }


def _new_model(method: str, defaults: dict):
    shared = {
        "warmup": defaults["warmup_windows"],
        "history": defaults["history_windows"],
        "refit_every": defaults["refit_every_windows"],
        "contamination": defaults["contamination"],
        "seed": defaults["model_seed"],
    }
    if method == "static":
        return StaticBaseline(**shared)
    if method == "adaptive_periodic":
        return AdaptiveBaseline(**shared, refit_trigger="periodic")
    if method == "adaptive_drift":
        drift = defaults["drift"]
        return AdaptiveBaseline(
            **shared,
            refit_trigger="drift",
            drift_method=drift["method"],
            psi_threshold=drift["psi_threshold"],
            min_drift_features=drift["min_features"],
            drift_persistence=drift["persistence"],
            test_windows=drift["test_windows"],
            check_every=drift["check_every"],
            drift_cooldown=drift["cooldown"],
        )
    raise ValueError(f"Unsupported evaluation method: {method}")


def _run_one(
    windows: pd.DataFrame,
    sidecar: dict,
    method: str,
    defaults: dict,
) -> tuple[dict, dict, object]:
    model = _new_model(method, defaults)
    results = model.run(windows[FEATURE_COLUMNS])
    warmup = defaults["warmup_windows"]
    labels = windows["label"].iloc[warmup:]
    predicted = results["flag"].iloc[warmup:]
    metrics = calculate_metrics(
        labels,
        predicted,
        window_indices=windows.index[warmup:],
        episodes=sidecar.get("episodes", []),
        window_seconds=sidecar.get("window_seconds", 1.0),
    )
    metrics["baseline_diagnostics"] = {
        "refit_count": int(model.n_refits),
        "skipped_refits": int(model.skipped_refits),
        "drift_event_count": len(model.drift_events),
        "drift_events": model.drift_events,
        "drift_refit_events": model.drift_refit_events,
        "promotion_events": model.promotion_events,
    }
    return metrics, results, model


def _summary(values: list[float | None]) -> dict:
    usable = [float(value) for value in values if value is not None]
    return {
        "mean": statistics.mean(usable) if usable else None,
        "std": statistics.stdev(usable) if len(usable) > 1 else (0.0 if usable else None),
    }


def _aggregate(records: dict[str, dict]) -> dict:
    fields = ("precision", "recall", "false_alarm_rate", "f1")
    confusion_fields = {
        field: _summary(
            [item["confusion"][field] for item in records.values()]
        )
        for field in ("tp", "fp", "tn", "fn")
    }
    attack_types = sorted(
        {
            attack_type
            for item in records.values()
            for attack_type in item["per_attack_type_recall"]
        }
    )
    per_attack_recall = {
        attack_type: _summary(
            [
                item["per_attack_type_recall"].get(attack_type)
                for item in records.values()
            ]
        )
        for attack_type in attack_types
    }
    delay_by_attack: dict[str, list[float]] = {}
    for item in records.values():
        for episode in item["detection_delay_per_episode_seconds"]:
            if episode["delay_seconds"] is not None:
                delay_by_attack.setdefault(episode["type"], []).append(
                    float(episode["delay_seconds"])
                )
    return {
        "mean_std": {
            field: _summary([item[field] for item in records.values()])
            for field in fields
        },
        "confusion_mean_std": confusion_fields,
        "per_attack_type_recall_mean_std": per_attack_recall,
        "detection_delay_seconds_mean_std_by_attack": {
            attack_type: _summary(delays)
            for attack_type, delays in delay_by_attack.items()
        },
        "per_seed": records,
    }


def _evaluate_poisoning(
    windows: pd.DataFrame,
    sidecar: dict,
    defaults: dict,
    method: str,
) -> dict:
    warmup = defaults["warmup_windows"]
    feature_matrix = windows[FEATURE_COLUMNS].to_numpy(dtype=float)
    labels = windows["label"].iloc[warmup:]
    indices = windows.index[warmup:]
    episodes = sidecar.get("episodes", [])
    attack_rows = feature_matrix[
        windows["label"].ne("normal").to_numpy()
    ]
    if attack_rows.size == 0:
        raise ValueError("Poisoning evaluation requires attack windows.")
    clean_model = _new_model(method, defaults)
    clean_results = clean_model.run(feature_matrix)
    clean_metrics = calculate_metrics(
        labels,
        clean_results["flag"].iloc[warmup:],
        window_indices=indices,
        episodes=episodes,
        window_seconds=sidecar.get("window_seconds", 1.0),
    )
    poisoning = {}
    for fraction in (0.05, 0.15):
        poisoned = feature_matrix.copy()
        count = max(1, int(round(warmup * fraction)))
        poisoned[:count] = attack_rows[np.arange(count) % len(attack_rows)]
        model = _new_model(method, defaults)
        result = model.run(poisoned)
        metrics = calculate_metrics(
            labels,
            result["flag"].iloc[warmup:],
            window_indices=indices,
            episodes=episodes,
            window_seconds=sidecar.get("window_seconds", 1.0),
        )
        poisoning[str(int(fraction * 100))] = {
            "warmup_poison_fraction": fraction,
            "warmup_poison_rows": count,
            "clean_recall": clean_metrics["recall"],
            "poisoned_recall": metrics["recall"],
            "recall_drop": (
                clean_metrics["recall"] - metrics["recall"]
                if clean_metrics["recall"] is not None
                and metrics["recall"] is not None
                else None
            ),
        }
    return {
        "method": method,
        "rule_note": (
            "The buffer rejects flagged windows, but false negatives can still "
            "enter the history."
        ),
        "clean_recall": clean_metrics["recall"],
        "poisoning": poisoning,
    }


def _plot_timeline(windows, results, sidecar, model, output_path, seed):
    indices = np.asarray(windows.index, dtype=float)
    scores = results["score"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(13, 5))
    drift_start = sidecar.get("drift_start")
    drift_end = sidecar.get("drift_end")
    if drift_start is not None and drift_end is not None:
        ax.axvspan(drift_start, drift_end, color="#CC79A7", alpha=0.2, label="Drift ramp")
    for number, episode in enumerate(sidecar.get("episodes", [])):
        ax.axvspan(
            episode["start"],
            episode["end"],
            color="#D55E00",
            alpha=0.12,
            label="Attack episode" if number == 0 else None,
        )
    ax.plot(indices, scores, color="#0072B2", linewidth=1, label="Anomaly score")
    ax.axhline(0, color="#333333", linestyle="--", linewidth=1, label="Flag threshold (0)")
    for event in model.drift_events:
        ax.axvline(event["position"], color="#009E73", alpha=0.65, linewidth=1)
    for event in model.drift_refit_events:
        ax.axvline(event["position"], color="#E69F00", alpha=0.8, linestyle=":")
    ax.plot([], [], color="#009E73", label="Drift event")
    ax.plot([], [], color="#E69F00", linestyle=":", label="Drift refit")
    ax.set(
        title=f"Layer 1 score timeline: drift_attacks, seed {seed}",
        xlabel="Window index",
        ylabel="Anomaly score (flagged when > 0)",
    )
    ax.legend(loc="upper left", ncols=3, fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_comparison(aggregates, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    metrics = ("false_alarm_rate", "recall")
    scenario_positions = np.arange(len(SCENARIOS))
    bar_width = 0.24
    for method_index, method in enumerate(METHODS):
        offsets = (
            method_index - (len(METHODS) - 1) / 2
        ) * bar_width
        for axis, metric in zip(axes, metrics):
            means = [
                aggregates[scenario][method]["mean_std"][metric]["mean"]
                for scenario in SCENARIOS
            ]
            deviations = [
                aggregates[scenario][method]["mean_std"][metric]["std"]
                for scenario in SCENARIOS
            ]
            axis.bar(
                scenario_positions + offsets,
                means,
                width=bar_width,
                yerr=deviations,
                capsize=3,
                color=COLORBLIND[method],
                label=method.replace("_", " "),
            )
    for axis, metric in zip(axes, metrics):
        axis.set_xticks(scenario_positions, [s.replace("_", "\n") for s in SCENARIOS])
        axis.set_ylabel(metric.replace("_", " ").title())
        axis.set_title(f"Mean {metric.replace('_', ' ')} ± SD")
        axis.set_ylim(bottom=0)
        axis.legend(fontsize="small")
    fig.suptitle("Adaptive versus static Layer 1 (synthetic, seeds 200–204)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _go_no_go(aggregates) -> dict:
    drift = aggregates["drift_attacks"]
    stable = aggregates["stable_attacks"]
    adaptive_drift = drift["adaptive_drift"]["mean_std"]
    static_drift = drift["static"]["mean_std"]
    adaptive_stable = stable["adaptive_drift"]["mean_std"]
    static_stable = stable["static"]["mean_std"]
    seed_far_diffs = [
        drift["adaptive_drift"]["per_seed"][seed]["false_alarm_rate"]
        - drift["static"]["per_seed"][seed]["false_alarm_rate"]
        for seed in drift["adaptive_drift"]["per_seed"]
    ]
    far_wins = sum(value < 0 for value in seed_far_diffs)
    conditions = {
        "drift_attacks_lower_mean_far": (
            adaptive_drift["false_alarm_rate"]["mean"]
            < static_drift["false_alarm_rate"]["mean"]
        ),
        "drift_attacks_recall_within_5_points": (
            adaptive_drift["recall"]["mean"]
            >= static_drift["recall"]["mean"] - PHASE5_DRIFT_RECALL_TOLERANCE
        ),
        "drift_attacks_far_wins_at_least_4_of_5": (
            far_wins >= PHASE5_DRIFT_FAR_WINS_REQUIRED
        ),
        "stable_attacks_far_within_2_points": (
            adaptive_stable["false_alarm_rate"]["mean"]
            <= static_stable["false_alarm_rate"]["mean"]
            + PHASE5_STABLE_FAR_TOLERANCE
        ),
        "stable_attacks_recall_within_5_points": (
            adaptive_stable["recall"]["mean"]
            >= static_stable["recall"]["mean"]
            - PHASE5_STABLE_RECALL_TOLERANCE
        ),
    }
    return {
        "verdict": "GO" if all(conditions.values()) else "NO-GO",
        "conditions": conditions,
        "drift_attacks_far_wins": int(far_wins),
        "paired_far_difference_mean": statistics.mean(seed_far_diffs),
        "tolerances": {
            "drift_attacks_recall_drop_max": PHASE5_DRIFT_RECALL_TOLERANCE,
            "stable_attacks_far_increase_max": PHASE5_STABLE_FAR_TOLERANCE,
            "stable_attacks_recall_drop_max": PHASE5_STABLE_RECALL_TOLERANCE,
            "drift_attacks_far_wins_required": PHASE5_DRIFT_FAR_WINS_REQUIRED,
        },
    }


def run_evaluation(seeds: list[int], output_dir: Path) -> dict:
    defaults = json.loads(DEFAULTS_PATH.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    per_scenario: dict[str, dict[str, dict]] = {
        scenario: {method: {} for method in METHODS} for scenario in SCENARIOS
    }
    timeline_payload = None
    poisoning = {}

    with tempfile.TemporaryDirectory(prefix="ata-phase5-") as temporary:
        temp_dir = Path(temporary)
        for scenario in SCENARIOS:
            for seed in seeds:
                csv_path = temp_dir / f"{scenario}_{seed}.csv"
                flows = generate_synthetic_csv(
                    csv_path,
                    seed=seed,
                    scenario=scenario,
                )
                windows = window_features(flows, window=1)
                sidecar = json.loads(
                    csv_path.with_suffix(".csv.meta.json").read_text(
                        encoding="utf-8"
                    )
                )
                for method in METHODS:
                    metrics, results, model = _run_one(
                        windows, sidecar, method, defaults
                    )
                    per_scenario[scenario][method][str(seed)] = metrics
                    if scenario == "drift_attacks" and seed == seeds[0]:
                        timeline_payload = (windows, results, sidecar, model, seed)
                if scenario == "stable_attacks" and seed == seeds[0]:
                    poisoning = {
                        method: _evaluate_poisoning(
                            windows, sidecar, defaults, method
                        )
                        for method in ("static", "adaptive_periodic", "adaptive_drift")
                    }

    aggregates = {
        scenario: {
            method: _aggregate(per_scenario[scenario][method])
            for method in METHODS
        }
        for scenario in SCENARIOS
    }
    comparisons = {}
    for scenario in SCENARIOS:
        comparisons[scenario] = {}
        for adaptive_method in ("adaptive_periodic", "adaptive_drift"):
            comparisons[scenario][adaptive_method] = {}
            for metric in ("false_alarm_rate", "recall"):
                differences = {
                    seed: (
                        per_scenario[scenario][adaptive_method][seed][metric]
                        - per_scenario[scenario]["static"][seed][metric]
                        if per_scenario[scenario][adaptive_method][seed][metric] is not None
                        and per_scenario[scenario]["static"][seed][metric] is not None
                        else None
                    )
                    for seed in per_scenario[scenario]["static"]
                }
                comparisons[scenario][adaptive_method][metric] = {
                    "adaptive_minus_static_per_seed": differences,
                    "mean_difference": _summary(list(differences.values()))["mean"],
                }

    verdict = _go_no_go(aggregates)
    cic_files = list((ROOT / "data").glob("*TrafficLabelling*.csv"))
    research_status = "PASS" if verdict["verdict"] == "GO" else "FAIL"
    failed_conditions = [
        name
        for name, passed in verdict["conditions"].items()
        if not passed
    ]
    report = {
        "status": "synthetic, not evidence of real-world performance",
        "research_acceptance": {
            "status": research_status,
            "verdict": verdict["verdict"],
            "failed_conditions": failed_conditions,
        },
        "config": defaults,
        "seeds": seeds,
        "scenarios": aggregates,
        "paired_differences": comparisons,
        "poisoning": poisoning,
        "verdict": verdict,
        "diagnosis": (
            {
                "summary": (
                    "The frozen Layer 1 research criteria were not all met. "
                    "Phase 6 engineering status is independent of this result."
                ),
                "failed_conditions": failed_conditions,
                "likely_causes": [
                    "Drift-trigger behavior trades false alarms against attack recall under the configured synthetic distribution shift.",
                    "The hard-negative stable_attacks scenario can cause attack bursts to resemble benign multi-feature drift.",
                    "The synthetic ramp and attack bursts create correlated multi-feature distribution changes.",
                ],
                "options": [
                    "Tune drift parameters only on development seeds 1-3, then run the frozen seeds once for research evaluation.",
                    "Retain the NO-GO result and report the drift/static tradeoff without changing the frozen criteria.",
                    "Continue independent Layer 2 engineering while keeping Layer 1 research acceptance marked FAIL.",
                ],
            }
            if verdict["verdict"] == "NO-GO"
            else None
        ),
        "cic_trafficlabelling": (
            {"status": "available", "files": [str(path) for path in cic_files]}
            if cic_files
            else {"status": "SKIPPED", "reason": "No CIC TrafficLabelling CSV files in data/."}
        ),
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if timeline_payload is not None:
        _plot_timeline(
            *timeline_payload[:-1],
            output_dir / "score_timeline.png",
            timeline_payload[-1],
        )
    _plot_comparison(aggregates, output_dir / "adaptive_vs_static.png")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare ATA Layer 1 baselines on deterministic synthetic data."
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[200, 201, 202, 203, 204]
    )
    parser.add_argument("--out", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    if not args.seeds:
        parser.error("at least one seed is required.")
    report = run_evaluation(args.seeds, args.out)
    print(f"Verdict: {report['verdict']['verdict']}")
    print(json.dumps(report["verdict"]["conditions"], indent=2))
    for scenario in SCENARIOS:
        print(f"\n{scenario} (synthetic, not evidence of real-world performance)")
        for method in METHODS:
            summary = report["scenarios"][scenario][method]["mean_std"]
            print(
                f"  {method}: FAR={summary['false_alarm_rate']['mean']:.4f} "
                f"+/- {summary['false_alarm_rate']['std']:.4f}; "
                f"recall={summary['recall']['mean']:.4f} "
                f"+/- {summary['recall']['std']:.4f}"
            )
    print(f"Reports: {args.out.resolve()}")


if __name__ == "__main__":
    main()
