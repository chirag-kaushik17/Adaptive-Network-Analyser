"""Tune drift-monitor guards using development seeds 1-3 only."""

from __future__ import annotations

import itertools
import json
import statistics
from pathlib import Path

from ata.drift import DriftMonitor
from ata.features import FEATURE_COLUMNS, window_features
from ata.synth import generate_synthetic_data

ROOT = Path(__file__).resolve().parents[1]
DEV_SEEDS = (1, 2, 3)
SCENARIOS = ("stable", "stable_attacks", "drift")
GRID = {
    "psi_threshold": (0.25, 0.5),
    "min_features": (3, 4),
    "persistence": (2, 3),
    "test_windows": (40, 60),
    "check_every": (10, 20),
}


def _event_positions(features, *, warmup: int, options: dict) -> list[int]:
    monitor = DriftMonitor(
        features[:warmup],
        method="psi",
        cooldown=30,
        feature_names=FEATURE_COLUMNS,
        **options,
    )
    for position, row in enumerate(features[warmup:], start=warmup):
        monitor.update(row, position=position)
    return [int(event["position"]) for event in monitor.events]


def main() -> None:
    cases = {}
    for scenario in SCENARIOS:
        for seed in DEV_SEEDS:
            frame = generate_synthetic_data(seed=seed, scenario=scenario)
            windows = window_features(frame, window=1)
            cases[(scenario, seed)] = (
                windows[FEATURE_COLUMNS].to_numpy(dtype=float),
                frame.attrs["drift_start"],
            )

    results = []
    keys = tuple(GRID)
    for values in itertools.product(*(GRID[key] for key in keys)):
        options = dict(zip(keys, values))
        detections = {}
        false_run_count = 0
        delays = []
        for scenario in SCENARIOS:
            for seed in DEV_SEEDS:
                features, drift_start = cases[(scenario, seed)]
                positions = _event_positions(features, warmup=120, options=options)
                detections[f"{scenario}:{seed}"] = positions
                if scenario in {"stable", "stable_attacks"}:
                    false_run_count += bool(positions)
                elif positions:
                    delay = positions[0] - int(drift_start)
                    delays.append(delay)
                else:
                    delays.append(float("inf"))
        median_delay = statistics.median(delays)
        results.append(
            {
                **options,
                "false_runs": int(false_run_count),
                "median_drift_delay_windows": median_delay,
                "detections": detections,
            }
        )

    eligible = [item for item in results if item["false_runs"] == 0]
    if eligible:
        selected = min(
            eligible,
            key=lambda item: (item["median_drift_delay_windows"], *[item[key] for key in keys]),
        )
        verdict = "zero false events achieved"
    else:
        selected = min(
            results,
            key=lambda item: (
                item["false_runs"],
                item["median_drift_delay_windows"],
                *[item[key] for key in keys],
            ),
        )
        verdict = "NO ZERO-FALSE CANDIDATE; selected minimum measured false-run count"

    lines = [
        "Phase 4 drift tuning (dev seeds 1-3 only)",
        "Selection criterion: zero events on stable and stable_attacks, then smallest median detection delay on drift.",
        f"Verdict: {verdict}",
        "threshold min_features persistence test_windows check_every false_runs median_delay_windows detections",
    ]
    for item in results:
        lines.append(
            f"{item['psi_threshold']:.2f} {item['min_features']} "
            f"{item['persistence']} {item['test_windows']} {item['check_every']} "
            f"{item['false_runs']} {item['median_drift_delay_windows']} "
            f"{json.dumps(item['detections'], sort_keys=True)}"
        )
    lines.append(f"CHOSEN: {json.dumps(selected, sort_keys=True)}")
    output = ROOT / "logs" / "phase4_drift_sweep.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    defaults_path = ROOT / "configs" / "defaults.json"
    defaults = json.loads(defaults_path.read_text(encoding="utf-8"))
    defaults["drift"] = {
        "method": "psi",
        "psi_threshold": selected["psi_threshold"],
        "min_features": selected["min_features"],
        "persistence": selected["persistence"],
        "test_windows": selected["test_windows"],
        "check_every": selected["check_every"],
        "cooldown": 30,
        "selection": {
            "dev_seeds": list(DEV_SEEDS),
            "scenarios": list(SCENARIOS),
            "criterion": (
                "zero events on stable and stable_attacks, then smallest median "
                "detection delay on drift"
            ),
            "verdict": verdict,
            "false_runs": selected["false_runs"],
            "median_drift_delay_windows": selected[
                "median_drift_delay_windows"
            ],
            "candidate_count": len(results),
            "chosen_parameters": {
                key: selected[key] for key in keys
            },
        },
    }
    defaults_path.write_text(
        json.dumps(defaults, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
