"""Run the frozen Phase 4 drift gates and informational model preview."""

from __future__ import annotations

import tempfile
from pathlib import Path
import sys

from ata.run import evaluate_csv
from ata.synth import generate_synthetic_csv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from thresholds import (
    PHASE4_DRIFT_EVENTS_MIN,
    PHASE4_FIRST_EVENT_AFTER_START,
    PHASE4_STABLE_DRIFT_EVENTS_MAX,
)

SMOKE_SEEDS = (100, 101, 102)
STABLE_SEEDS = SMOKE_SEEDS + (200, 201, 202, 203, 204, 205, 206)
CASES = [
    *((seed, "stable") for seed in STABLE_SEEDS),
    *((seed, "stable_attacks") for seed in STABLE_SEEDS),
    *((seed, "drift") for seed in SMOKE_SEEDS),
    *((seed, "drift_attacks") for seed in SMOKE_SEEDS),
]


def _evaluate(path: Path, trigger: str):
    return evaluate_csv(
        path,
        "synth",
        warmup=120,
        history=300,
        refit_every=30,
        refit_trigger=trigger,
    )


def main() -> None:
    rows = []
    with tempfile.TemporaryDirectory(prefix="ata-phase4-") as temp:
        root = Path(temp)
        for seed, scenario in CASES:
            csv_path = root / f"{scenario}_{seed}.csv"
            generate_synthetic_csv(
                csv_path,
                seed=seed,
                scenario=scenario,
            )
            output = _evaluate(csv_path, "drift")
            events = output["model"].drift_events
            first = events[0]["position"] if events else None
            sidecar = output["sidecar"]
            delay = None
            if first is not None and sidecar["drift_start"] is not None:
                delay = first - sidecar["drift_start"]
            if scenario in {"stable", "stable_attacks"}:
                passed = len(events) <= PHASE4_STABLE_DRIFT_EVENTS_MAX
            elif scenario == "drift":
                passed = (
                    len(events) >= PHASE4_DRIFT_EVENTS_MIN
                    and (
                        not PHASE4_FIRST_EVENT_AFTER_START
                        or first > sidecar["drift_start"]
                    )
                )
            else:
                passed = True
            rows.append(
                (
                    seed,
                    scenario,
                    len(events),
                    first,
                    delay,
                    output["model"].n_refits,
                    "PASS" if passed else "FAIL",
                )
            )

        print("seed scenario drift_events first_event_window detection_delay_windows refits result")
        for row in rows:
            print(
                f"{row[0]} {row[1]} {row[2]} "
                f"{row[3] if row[3] is not None else 'n/a'} "
                f"{row[4] if row[4] is not None else 'n/a'} {row[5]} {row[6]}"
            )
        stable_rows = [
            row for row in rows if row[1] in {"stable", "stable_attacks"}
        ]
        false_event_runs = sum(row[2] > 0 for row in stable_rows)
        print(
            f"Stable/hard-negative false-drift rate: {false_event_runs}/"
            f"{len(stable_rows)} runs ({false_event_runs / max(1, len(stable_rows)):.3f})"
        )

        print("\nPreview, not the Phase 5 result (drift_attacks; smoke seeds):")
        print("seed baseline refit_trigger recall false_alarm_rate drift_events refits")
        for seed in SMOKE_SEEDS:
            csv_path = root / f"drift_attacks_{seed}.csv"
            for baseline, trigger in (
                ("adaptive", "periodic"),
                ("adaptive", "drift"),
                ("static", "periodic"),
            ):
                output = evaluate_csv(
                    csv_path,
                    "synth",
                    warmup=120,
                    history=300,
                    refit_every=30,
                    baseline_name=baseline,
                    refit_trigger=trigger,
                )
                metrics = output["metrics"]
                print(
                    f"{seed} {baseline} {trigger} {metrics['recall']} "
                    f"{metrics['false_alarm_rate']} "
                    f"{len(output['model'].drift_events)} "
                    f"{output['model'].n_refits}"
                )

    failed = [row for row in rows if row[-1] != "PASS"]
    if failed:
        raise SystemExit(f"{len(failed)} Phase 4 acceptance rows failed.")


if __name__ == "__main__":
    main()
