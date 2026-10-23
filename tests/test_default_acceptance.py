import json
import platform
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
from tests.thresholds import (
    ATTACK_MEAN_FALSE_ALARM_MAX,
    ATTACK_MEAN_RECALL_MIN,
    NORMAL_MEAN_FALSE_ALARM_MAX,
)

ROOT = Path(__file__).resolve().parents[1]
SMOKE_SEEDS = (100, 101, 102)


def _evaluate(windows, contamination):
    model = AdaptiveBaseline(
        warmup=120,
        history=300,
        refit_every=30,
        contamination=contamination,
        seed=0,
    )
    results = model.run(windows[FEATURE_COLUMNS])
    metrics = summarize_evaluation(
        windows["label"].iloc[120:], results["flag"].iloc[120:]
    )
    return metrics


def test_fixed_defaults_meet_smoke_seed_acceptance_criteria():
    defaults = json.loads(
        (ROOT / "configs" / "defaults.json").read_text(encoding="utf-8")
    )
    contamination = float(defaults["contamination"])
    attack_metrics = []
    normal_metrics = []
    started = time.perf_counter()

    for seed in SMOKE_SEEDS:
        attack_flows = generate_synthetic_data(seed=seed)
        normal_flows = generate_synthetic_data(
            seed=seed, seconds=900, attacks=False
        )
        attack_metrics.append(
            _evaluate(window_features(attack_flows, window=1), contamination)
        )
        normal_metrics.append(
            _evaluate(window_features(normal_flows, window=1), contamination)
        )

    mean_recall = statistics.mean(item["recall"] for item in attack_metrics)
    mean_attack_far = statistics.mean(
        item["false_alarm_rate"] for item in attack_metrics
    )
    mean_normal_far = statistics.mean(
        item["false_alarm_rate"] for item in normal_metrics
    )
    print(
        "Smoke-seed acceptance: "
        f"attack recall={mean_recall:.4f}, attack FAR={mean_attack_far:.4f}, "
        f"normal-only FAR={mean_normal_far:.4f}"
    )
    for seed, attack, normal in zip(SMOKE_SEEDS, attack_metrics, normal_metrics):
        print(
            f"  data_seed={seed}: A(TP={attack['tp']}, FP={attack['fp']}, "
            f"FN={attack['fn']}, recall={attack['recall']:.4f}, "
            f"FAR={attack['false_alarm_rate']:.4f}); "
            f"B(FP={normal['fp']}, FAR={normal['false_alarm_rate']:.4f})"
        )
    print(
        "Smoke-seed evaluation machine: "
        f"elapsed={time.perf_counter() - started:.2f}s; "
        f"CPU={cpu_model()}; "
        f"logical_cores={core_count()}; Python={platform.python_version()}; "
        f"numpy={np.__version__}; pandas={pd.__version__}; "
        f"scikit-learn={sklearn.__version__}"
    )
    assert mean_recall > ATTACK_MEAN_RECALL_MIN
    assert mean_attack_far < ATTACK_MEAN_FALSE_ALARM_MAX
    assert mean_normal_far < NORMAL_MEAN_FALSE_ALARM_MAX
