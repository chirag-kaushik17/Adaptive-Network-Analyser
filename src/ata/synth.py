"""Deterministic synthetic flow generator for tests and local experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ATTACK_TYPES = ("portscan", "ddos", "exfiltration")


def build_attack_schedule(
    attack_start: float,
    n_cycles: int,
    attack_duration: float,
    attack_gap: float,
    attack_order: str = "fixed",
    seed: int = 100,
) -> list[dict]:
    """Build non-overlapping attack episodes separated by normal-only gaps."""
    if n_cycles < 1 or attack_duration <= 0 or attack_gap < 0:
        raise ValueError("n_cycles and attack_duration must be positive; attack_gap cannot be negative.")
    if attack_order not in {"fixed", "shuffled"}:
        raise ValueError("attack_order must be 'fixed' or 'shuffled'.")

    rng = np.random.default_rng(seed)
    cursor = float(attack_start)
    episodes: list[dict] = []
    for cycle in range(n_cycles):
        order = list(ATTACK_TYPES)
        if attack_order == "shuffled":
            rng.shuffle(order)
        for attack_type in order:
            duration = float(attack_duration)
            if attack_order == "shuffled":
                duration = max(25.0, duration * float(rng.uniform(0.8, 1.2)))
            episodes.append(
                {
                    "type": attack_type,
                    "cycle": cycle,
                    "start": cursor,
                    "end": cursor + duration,
                    "duration": duration,
                }
            )
            cursor += duration
            if len(episodes) < n_cycles * len(ATTACK_TYPES):
                cursor += float(attack_gap)
    return episodes


def _normal_flows(
    rng: np.random.Generator,
    seconds: float,
    drift_start: float | None,
    drift_end: float | None,
    drift_strength: float,
) -> list[dict]:
    flows: list[dict] = []
    servers = ["10.0.0.1", "10.0.0.2", "10.0.0.3"]
    ports = np.array([80, 443, 53, 22, 8080])
    second = 0
    while second < seconds:
        ramp = 0.0
        if drift_start is not None and drift_end is not None and drift_end > drift_start:
            ramp = float(
                np.clip((second - drift_start) / (drift_end - drift_start), 0, 1)
            )
        count = int(rng.poisson(150 * (1 + 2 * drift_strength * ramp)))
        probabilities = (
            np.array([0.20, 0.45, 0.15, 0.10, 0.10]) * (1 - ramp)
            + np.array([0.10, 0.75, 0.05, 0.05, 0.05]) * ramp
        )
        byte_scale = 700 * (1 + drift_strength * ramp)
        for _ in range(count):
            timestamp = second + float(rng.random())
            if timestamp >= seconds:
                continue
            packets = max(1, int(rng.poisson(5)))
            flows.append(
                {
                    "ts": timestamp,
                    "src": f"10.0.0.{int(rng.integers(10, 50))}",
                    "dst": servers[int(rng.integers(0, len(servers)))],
                    "dport": int(rng.choice(ports, p=probabilities)),
                    "proto": str(rng.choice(["tcp", "udp", "tcp", "tcp"])),
                    "bytes": max(64, int(rng.exponential(byte_scale) * packets)),
                    "packets": packets,
                    "duration": float(max(0.01, rng.exponential(0.75))),
                    "syn": int(rng.random() < 0.12),
                    "label": "normal",
                }
            )
        second += 1
    return flows


def _flows_for_episode(rng: np.random.Generator, episode: dict) -> list[dict]:
    """Emit flows whose timestamps stay inside the episode's interval."""
    flows: list[dict] = []
    start = float(episode["start"])
    end = float(episode["end"])
    attack_type = str(episode["type"])
    second = int(start)
    while second < end:
        slot_start = max(start, float(second))
        slot_end = min(end, float(second + 1))
        count = {"portscan": 50, "ddos": 60, "exfiltration": 5}[attack_type]
        for index in range(count):
            timestamp = slot_start + float(rng.random()) * (slot_end - slot_start)
            if attack_type == "portscan":
                fields = {
                    "src": "10.0.0.5",
                    "dst": "10.0.0.3",
                    "dport": int(rng.integers(1, 65536)),
                    "proto": "tcp",
                    "bytes": int(rng.integers(20, 80)),
                    "packets": 1,
                    "duration": 0.001,
                    "syn": 1,
                }
            elif attack_type == "ddos":
                fields = {
                    "src": f"192.0.{index // 256}.{index % 256}",
                    "dst": "10.0.0.1",
                    "dport": 80,
                    "proto": "tcp",
                    "bytes": int(rng.integers(2000, 12000)),
                    "packets": int(rng.integers(5, 30)),
                    "duration": float(rng.exponential(0.1)),
                    "syn": 1,
                }
            else:
                fields = {
                    "src": "10.0.0.10",
                    "dst": "198.51.100.5",
                    "dport": 443,
                    "proto": "tcp",
                    "bytes": int(rng.integers(250000, 600000)),
                    "packets": int(rng.integers(80, 250)),
                    "duration": float(rng.exponential(1.5)),
                    "syn": 0,
                }
            flows.append({"ts": timestamp, **fields, "label": attack_type})
        second += 1
    return flows


def generate_synthetic_data(
    seconds: int | None = None,
    seed: int = 100,
    attack_start: float | None = None,
    attacks: bool = True,
    drift: bool = False,
    *,
    warmup: int = 120,
    window: float = 1.0,
    n_cycles: int = 3,
    attack_duration: float = 30.0,
    attack_gap: float = 40.0,
    tail_duration: float = 190.0,
    attack_order: str = "fixed",
    drift_start: float | None = None,
    drift_end: float | None = None,
    drift_strength: float = 0.0,
    stealth_start: float | None = None,
    stealth_duration: float = 0.0,
    stealth_amp: float = 0.0,
) -> pd.DataFrame:
    """Generate clean warm-up traffic, optional attacks, drift, and stealth."""
    if warmup < 1 or window <= 0:
        raise ValueError("warmup must be positive and window must be greater than zero.")
    if tail_duration < 0 or stealth_duration < 0 or stealth_amp < 0:
        raise ValueError("tail_duration, stealth_duration, and stealth_amp cannot be negative.")
    warmup_seconds = warmup * window
    explicit_attack_start = attack_start is not None
    explicit_drift_start = drift_start is not None
    explicit_stealth_start = stealth_start is not None
    if explicit_attack_start and attack_start < warmup_seconds:
        raise ValueError("attack_start cannot be earlier than the end of warm-up.")
    if explicit_drift_start and drift_start < warmup_seconds:
        raise ValueError("drift_start cannot be earlier than the end of warm-up.")
    if explicit_stealth_start and stealth_start < warmup_seconds:
        raise ValueError("stealth_start cannot be earlier than the end of warm-up.")
    if attack_start is None:
        attack_start = warmup_seconds

    drift_active = drift or drift_strength > 0
    effective_drift_strength = 1.0 if drift and drift_strength == 0 else drift_strength
    if drift_active and drift_start is None:
        drift_start = warmup_seconds
    if drift_active and drift_start is not None and drift_end is None:
        drift_end = max(drift_start + window, warmup_seconds + 300 * window)

    if attacks and stealth_duration > 0 and stealth_amp > 0 and stealth_start is None:
        stealth_start = warmup_seconds

    episodes = (
        build_attack_schedule(
            float(attack_start), n_cycles, attack_duration, attack_gap,
            attack_order, seed,
        )
        if attacks
        else []
    )
    scheduled_attack_end = float(episodes[-1]["end"]) if episodes else warmup_seconds
    stealth_end = (
        float(stealth_start) + stealth_duration
        if attacks and stealth_start is not None and stealth_duration > 0 and stealth_amp > 0
        else 0.0
    )
    required_end = max(
        scheduled_attack_end + (tail_duration if attacks else 0),
        float(drift_end) + tail_duration if drift_active and drift_end is not None else 0,
        stealth_end + tail_duration if stealth_end else 0,
        warmup_seconds + (tail_duration if not attacks and not drift_active else 0),
    )
    if seconds is not None:
        minimum_end = max(
            scheduled_attack_end if attacks else warmup_seconds,
            float(drift_end) if drift_active and drift_end is not None else 0,
            stealth_end,
        )
        if float(seconds) < minimum_end:
            raise ValueError("seconds ends before a configured traffic event.")
        if attacks and float(seconds) < required_end:
            raise ValueError("seconds is too short to contain the configured attacks and normal tail.")
        total_seconds = float(seconds)
    else:
        total_seconds = required_end

    rng = np.random.default_rng(seed)
    flows = _normal_flows(
        rng,
        total_seconds,
        drift_start if drift_active else None,
        drift_end if drift_active else None,
        effective_drift_strength if drift_active else 0.0,
    )
    for episode in episodes:
        flows.extend(_flows_for_episode(rng, episode))

    if stealth_end:
        stealth_start_value = float(stealth_start)
        for timestamp in np.arange(stealth_start_value, stealth_end, 0.2):
            flows.append(
                {
                    "ts": float(timestamp),
                    "src": "10.0.0.10",
                    "dst": "198.51.100.99",
                    "dport": 443,
                    "proto": "tcp",
                    "bytes": int(50000 + 100000 * stealth_amp),
                    "packets": 30,
                    "duration": 1.0,
                    "syn": 0,
                    "label": "exfiltration",
                }
            )

    frame = pd.DataFrame(flows).sort_values("ts", kind="stable").reset_index(drop=True)
    if frame.empty:
        raise ValueError("Synthetic data generation created no rows.")
    if attacks and (frame.loc[frame["label"].ne("normal"), "ts"] < warmup_seconds).any():
        raise AssertionError("Synthetic generator violated the clean warm-up invariant.")
    frame.attrs["duration_seconds"] = total_seconds
    frame.attrs["warmup_seconds"] = warmup_seconds
    return frame


def generate_synthetic_csv(
    path: str | Path,
    seconds: int | None = None,
    seed: int = 100,
    attack_start: float | None = None,
    attacks: bool = True,
    drift: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Write flows and schedule metadata, which is never used as model input."""
    frame = generate_synthetic_data(
        seconds=seconds,
        seed=seed,
        attack_start=attack_start,
        attacks=attacks,
        drift=drift,
        **kwargs,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    warmup_seconds = kwargs.get("warmup", 120) * kwargs.get("window", 1.0)
    active_attack_start = attack_start if attack_start is not None else warmup_seconds
    episodes = (
        build_attack_schedule(
            active_attack_start,
            kwargs.get("n_cycles", 3),
            kwargs.get("attack_duration", 30.0),
            kwargs.get("attack_gap", 40.0),
            kwargs.get("attack_order", "fixed"),
            seed,
        )
        if attacks
        else []
    )
    drift_active = drift or kwargs.get("drift_strength", 0.0) > 0
    active_drift_start = kwargs.get("drift_start")
    if drift_active and active_drift_start is None:
        active_drift_start = warmup_seconds
    active_drift_end = kwargs.get("drift_end")
    if drift_active and active_drift_end is None:
        active_drift_end = max(
            active_drift_start + kwargs.get("window", 1.0),
            warmup_seconds + 300 * kwargs.get("window", 1.0),
        )
    stealth_active = (
        attacks
        and kwargs.get("stealth_duration", 0.0) > 0
        and kwargs.get("stealth_amp", 0.0) > 0
    )
    active_stealth_start = kwargs.get("stealth_start") if stealth_active else None
    if stealth_active:
        if active_stealth_start is None:
            active_stealth_start = warmup_seconds
    metadata = {
        "seed": seed,
        "warmup_seconds": warmup_seconds,
        "attack_start": active_attack_start if attacks else None,
        "episodes": episodes,
        "attacks": attacks,
        "attack_order": kwargs.get("attack_order", "fixed"),
        "attack_duration": kwargs.get("attack_duration", 30.0),
        "attack_gap": kwargs.get("attack_gap", 40.0),
        "n_cycles": kwargs.get("n_cycles", 3),
        "tail_duration": kwargs.get("tail_duration", 190.0),
        "window_seconds": kwargs.get("window", 1.0),
        "duration_seconds": frame.attrs["duration_seconds"],
        "drift_start": active_drift_start if drift_active else None,
        "drift_end": active_drift_end if drift_active else None,
        "drift_strength": (
            1.0 if drift and kwargs.get("drift_strength", 0.0) == 0
            else kwargs.get("drift_strength", 0.0)
        ),
        "stealth_start": active_stealth_start,
        "stealth_duration": kwargs.get("stealth_duration", 0.0),
        "stealth_amp": kwargs.get("stealth_amp", 0.0),
    }
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic synthetic network flows.")
    parser.add_argument("--out", default="data/synth.csv")
    parser.add_argument("--seconds", type=int)
    parser.add_argument("--data-seed", "--seed", dest="data_seed", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=120)
    parser.add_argument("--window", type=float, default=1.0)
    parser.add_argument("--attack-start", type=float)
    parser.add_argument("--n-cycles", type=int, default=3)
    parser.add_argument("--attack-duration", type=float, default=30.0)
    parser.add_argument("--attack-gap", type=float, default=40.0)
    parser.add_argument("--tail-duration", type=float, default=190.0)
    parser.add_argument("--attack-order", choices=["fixed", "shuffled"], default="fixed")
    parser.add_argument("--no-attacks", action="store_true")
    parser.add_argument("--drift", action="store_true")
    parser.add_argument("--drift-start", type=float)
    parser.add_argument("--drift-end", type=float)
    parser.add_argument("--drift-strength", type=float, default=0.0)
    parser.add_argument("--stealth-start", type=float)
    parser.add_argument("--stealth-duration", type=float, default=0.0)
    parser.add_argument("--stealth-amp", type=float, default=0.0)
    args = parser.parse_args()

    output = Path(args.out)
    frame = generate_synthetic_csv(
        path=output,
        seconds=args.seconds,
        seed=args.data_seed,
        attack_start=args.attack_start,
        attacks=not args.no_attacks,
        drift=args.drift,
        warmup=args.warmup,
        window=args.window,
        n_cycles=args.n_cycles,
        attack_duration=args.attack_duration,
        attack_gap=args.attack_gap,
        tail_duration=args.tail_duration,
        attack_order=args.attack_order,
        drift_start=args.drift_start,
        drift_end=args.drift_end,
        drift_strength=args.drift_strength,
        stealth_start=args.stealth_start,
        stealth_duration=args.stealth_duration,
        stealth_amp=args.stealth_amp,
    )
    print(f"Generated {len(frame)} flows in {frame['ts'].min():.2f}-{frame['ts'].max():.2f}s")
    print(f"Label counts: {frame['label'].value_counts().sort_index().to_dict()}")
    print(f"Metadata: {output.with_suffix(output.suffix + '.meta.json')}")


if __name__ == "__main__":
    main()
