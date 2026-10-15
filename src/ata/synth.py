from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _normal_flows(rng: np.random.Generator, seconds: int, drift: bool = False) -> list[dict]:
    flows: list[dict] = []
    server_hosts = ["10.0.0.1", "10.0.0.2", "10.0.0.3"]
    port_choices = [80, 443, 53, 22, 8080]
    for second in range(seconds):
        rate = 150
        if drift:
            rate = int(150 + 50 * np.sin(second / max(seconds, 1) * 2 * np.pi) + 30 * (second / max(seconds, 1)))
        n = int(rng.poisson(rate))
        for _ in range(n):
            ts = second + float(rng.random())
            src = f"10.0.0.{int(rng.integers(10, 40))}"
            dst = server_hosts[int(rng.integers(0, len(server_hosts)))]
            dport = int(rng.choice(port_choices))
            proto = str(rng.choice(["tcp", "udp", "tcp", "tcp"]))
            packets = max(1, int(rng.poisson(5)))
            duration = float(max(0.01, rng.exponential(0.75)))
            byte_total = max(64, int(rng.exponential(700) * packets))
            syn = int(rng.random() < 0.12)
            flows.append(
                {
                    "ts": ts,
                    "src": src,
                    "dst": dst,
                    "dport": dport,
                    "proto": proto,
                    "bytes": byte_total,
                    "packets": packets,
                    "duration": duration,
                    "syn": syn,
                    "label": "normal",
                }
            )
    return flows


def _attack_flows(rng: np.random.Generator, seconds: int, attack_start: float) -> list[dict]:
    flows: list[dict] = []
    if attack_start is None:
        return flows

    portscan_count = max(40, int(seconds * 0.3))
    for idx in range(portscan_count):
        ts = attack_start + idx * 0.25 + rng.random() * 0.15
        if ts > seconds:
            break
        dport = int(rng.integers(1, 65535))
        flows.append(
            {
                "ts": ts,
                "src": "10.0.0.5",
                "dst": "10.0.0.3",
                "dport": dport,
                "proto": "tcp",
                "bytes": int(rng.integers(20, 80)),
                "packets": 1,
                "duration": 0.001,
                "syn": 1,
                "label": "PortScan",
            }
        )

    ddos_targets = ["10.0.0.1", "10.0.0.2"]
    for idx in range(int((seconds - attack_start) * 2)):
        ts = attack_start + idx * 0.5 + rng.random() * 0.2
        if ts > seconds:
            break
        for src_idx in range(30):
            source = f"192.168.{src_idx // 256}.{src_idx % 256}"
            flows.append(
                {
                    "ts": ts,
                    "src": source,
                    "dst": ddos_targets[idx % len(ddos_targets)],
                    "dport": 80,
                    "proto": "tcp",
                    "bytes": int(rng.integers(2000, 50000)),
                    "packets": int(rng.integers(5, 30)),
                    "duration": float(rng.exponential(0.1)),
                    "syn": 1,
                    "label": "DDoS",
                }
            )

    exfil_count = int((seconds - attack_start) * 0.7)
    for idx in range(exfil_count):
        ts = attack_start + idx * 0.7 + rng.random() * 0.2
        if ts > seconds:
            break
        flows.append(
            {
                "ts": ts,
                "src": "10.0.0.10",
                "dst": "198.51.100.5",
                "dport": 443,
                "proto": "tcp",
                "bytes": int(rng.integers(150000, 600000)),
                "packets": int(rng.integers(80, 250)),
                "duration": float(rng.exponential(1.5)),
                "syn": 0,
                "label": "Exfiltration",
            }
        )
    return flows


def generate_synthetic_data(
    seconds: int = 600,
    seed: int = 7,
    attack_start: float | None = None,
    attacks: bool = True,
    drift: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if attack_start is None and attacks:
        attack_start = max(0.0, seconds * 0.7)
    flows: list[dict] = _normal_flows(rng=rng, seconds=seconds, drift=drift)
    if attacks and attack_start is not None:
        flows.extend(_attack_flows(rng=rng, seconds=seconds, attack_start=float(attack_start)))
    frame = pd.DataFrame(flows)
    if frame.empty:
        raise ValueError("Synthetic data generation created no rows.")
    frame = frame.sort_values("ts").reset_index(drop=True)
    return frame


def generate_synthetic_csv(
    path: str | Path,
    seconds: int = 600,
    seed: int = 7,
    attack_start: float | None = None,
    attacks: bool = True,
    drift: bool = False,
) -> pd.DataFrame:
    frame = generate_synthetic_data(seconds=seconds, seed=seed, attack_start=attack_start, attacks=attacks, drift=drift)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic network-flow data.")
    parser.add_argument("--out", default="data/synth.csv")
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--attack-start", type=float, default=None)
    parser.add_argument("--no-attacks", action="store_true")
    parser.add_argument("--drift", action="store_true")
    args = parser.parse_args()
    frame = generate_synthetic_csv(
        path=args.out,
        seconds=args.seconds,
        seed=args.seed,
        attack_start=args.attack_start,
        attacks=not args.no_attacks,
        drift=args.drift,
    )
    print(f"Generated {len(frame)} rows in {frame['ts'].min():.2f}-{frame['ts'].max():.2f}s")


if __name__ == "__main__":
    main()
