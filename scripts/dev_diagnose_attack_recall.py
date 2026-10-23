"""Inspect per-family detector recall on development data only."""

from __future__ import annotations

import platform
import time

import numpy as np
import pandas as pd
import sklearn

from ata.baseline import AdaptiveBaseline
from ata.features import FEATURE_COLUMNS, window_features
from ata.machine import core_count, cpu_model
from ata.synth import generate_synthetic_data


def main() -> None:
    started = time.perf_counter()
    observations: dict[str, list[float]] = {}
    for seed in (1, 2, 3):
        windows = window_features(generate_synthetic_data(seed=seed), window=1)
        baseline = AdaptiveBaseline(
            warmup=120,
            history=300,
            refit_every=30,
            contamination=0.005,
            seed=0,
        )
        predictions = baseline.run(windows[FEATURE_COLUMNS])["flag"].iloc[120:]
        actual = windows["label"].iloc[120:]
        for attack_type in sorted(set(actual) - {"normal"}):
            mask = actual.eq(attack_type)
            observations.setdefault(attack_type, []).append(
                float(predictions.loc[mask].mean())
            )
    print("Development-seed attack-family recall at contamination=0.005:")
    for attack_type, values in sorted(observations.items()):
        print(f"{attack_type}: mean={np.mean(values):.4f} seeds={values}")
    print(
        "machine: "
        f"CPU={cpu_model()}, "
        f"logical_cores={core_count()}, Python={platform.python_version()}, "
        f"numpy={np.__version__}, pandas={pd.__version__}, "
        f"scikit-learn={sklearn.__version__}, "
        f"elapsed_seconds={time.perf_counter() - started:.2f}"
    )


if __name__ == "__main__":
    main()
