"""Choose the Layer 1 contamination default using development seeds only."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from ata.baseline import AdaptiveBaseline
from ata.features import FEATURE_COLUMNS, window_features
from ata.machine import core_count, cpu_model
from ata.run import summarize_evaluation
from ata.synth import generate_synthetic_data

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "defaults.json"
DEV_SEEDS = (1, 2, 3)
CANDIDATES = (0.005, 0.01, 0.02, 0.05)


def _run(features: pd.DataFrame, warmup: int, contamination: float):
    model = AdaptiveBaseline(
        warmup=warmup,
        history=300,
        refit_every=30,
        contamination=contamination,
        seed=0,
    )
    prediction = model.run(features[FEATURE_COLUMNS])
    metrics = summarize_evaluation(
        features["label"].iloc[warmup:], prediction["flag"].iloc[warmup:]
    )
    return metrics


def main() -> None:
    started = time.perf_counter()
    scenarios: dict[tuple[str, int], pd.DataFrame] = {}
    for seed in DEV_SEEDS:
        attack = generate_synthetic_data(seed=seed, attacks=True)
        normal = generate_synthetic_data(seed=seed, seconds=900, attacks=False)
        scenarios[("A", seed)] = window_features(attack, window=1)
        scenarios[("B", seed)] = window_features(normal, window=1)

    records: dict[float, dict] = {}
    for contamination in CANDIDATES:
        attack_recalls = []
        false_alarm_rates = []
        for scenario in ("A", "B"):
            for seed in DEV_SEEDS:
                values = scenarios[(scenario, seed)]
                metrics = _run(values, 120, contamination)
                if scenario == "A":
                    attack_recalls.append(float(metrics["recall"]))
                false_alarm_rates.append(float(metrics["false_alarm_rate"]))
        records[contamination] = {
            "mean_attack_recall": statistics.mean(attack_recalls),
            "mean_false_alarm_rate": statistics.mean(false_alarm_rates),
            "seed_attack_recall": attack_recalls,
            "seed_false_alarm_rate": false_alarm_rates,
        }

    eligible = [
        value
        for value, summary in records.items()
        if summary["mean_attack_recall"] > 0.60
    ]
    if eligible:
        chosen = min(
            eligible,
            key=lambda value: (
                records[value]["mean_false_alarm_rate"],
                value,
            ),
        )
    else:
        chosen = min(
            CANDIDATES,
            key=lambda value: (
                -records[value]["mean_attack_recall"],
                records[value]["mean_false_alarm_rate"],
                value,
            ),
        )

    print("Dev sweep: scenarios A (stable + attacks) and B (stable normal-only)")
    print("seeds=1,2,3; model_seed=0; warmup=120; history=300; refit_every=30")
    print("contamination | mean recall A | mean false-alarm rate A+B")
    for value in CANDIDATES:
        item = records[value]
        print(
            f"{value:13.3f} | {item['mean_attack_recall']:13.4f} | "
            f"{item['mean_false_alarm_rate']:26.4f}"
        )
    print(f"chosen contamination: {chosen:.3f}")
    print(
        "machine: "
        f"CPU={cpu_model()}, "
        f"logical_cores={core_count()}, Python={platform.python_version()}, "
        f"numpy={np.__version__}, pandas={pd.__version__}, scikit-learn={sklearn.__version__}"
    )
    print(f"elapsed_seconds: {time.perf_counter() - started:.2f}")

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(
            {
                "contamination": chosen,
                "warmup_windows": 120,
                "history_windows": 300,
                "refit_every_windows": 30,
                "model_seed": 0,
                "selection": {
                    "dev_seeds": list(DEV_SEEDS),
                    "candidates": list(CANDIDATES),
                    "criterion": "minimum mean false-alarm rate subject to mean scenario-A recall > 0.60; tie-break smaller contamination",
                    "results": {
                        str(value): records[value] for value in CANDIDATES
                    },
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {CONFIG_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
