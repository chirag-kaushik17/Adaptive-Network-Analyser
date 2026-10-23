import pandas as pd

from ata.loader import load
from ata.synth import generate_synthetic_data


def _cic_row(timestamp, label="BENIGN"):
    return {
        " Timestamp ": timestamp,
        " Source IP ": "10.0.0.1",
        " Destination IP ": "10.0.0.2",
        " Destination Port ": 80,
        " Protocol ": 6,
        " Flow Duration ": 1_000_000,
        " Total Fwd Packets ": 2,
        " Total Backward Packets ": 1,
        " Total Length of Fwd Packets ": 50,
        " Total Length of Bwd Packets ": 30,
        " SYN Flag Count ": 1,
        " Label ": label,
    }


def test_cic_loader_handles_messy_headers_pm_timestamp_and_invalid_rows(tmp_path):
    csv_path = tmp_path / "cic.csv"
    pd.DataFrame(
        [
            _cic_row("Wednesday-07-07-2017 03:01:00"),
            _cic_row("Wednesday-07-07-2017 03:02:00", "PortScan"),
            _cic_row("bad-ts"),
        ]
    ).to_csv(csv_path, index=False)

    df, stats = load(csv_path, "cicids2017", return_stats=True)

    assert list(df.columns) == [
        "ts", "src", "dst", "dport", "proto", "bytes", "packets",
        "duration", "syn", "label", "label_raw",
    ]
    assert len(df) == 2
    assert df["ts"].tolist() == [0.0, 60.0]
    assert df["label"].tolist() == ["normal", "portscan"]
    assert df["label_raw"].tolist() == ["benign", "portscan"]
    assert df["proto"].tolist() == ["tcp", "tcp"]
    assert df["bytes"].tolist() == [80.0, 80.0]
    assert df["packets"].tolist() == [3.0, 3.0]
    assert df["duration"].tolist() == [1.0, 1.0]
    assert df["syn"].tolist() == [1, 1]
    assert stats["discarded_invalid_timestamps"] == 1
    assert stats["date_order_used"] == "dmy"
    assert stats["date_range"] == ("2017-07-07", "2017-07-07")


def test_synthetic_data_is_deterministic_and_no_attacks_stays_normal():
    first = generate_synthetic_data(
        seed=100, attacks=True, warmup=2, attack_start=2, n_cycles=1,
        attack_duration=3, attack_gap=1, tail_duration=2,
    )
    second = generate_synthetic_data(
        seed=100, attacks=True, warmup=2, attack_start=2, n_cycles=1,
        attack_duration=3, attack_gap=1, tail_duration=2,
    )
    assert first.equals(second)
    assert set(first["label"]) == {"normal", "portscan", "ddos", "exfiltration"}

    normal_only = generate_synthetic_data(
        seconds=60, seed=101, warmup=10, attacks=False
    )
    drift = generate_synthetic_data(
        seconds=60, seed=102, warmup=10, attacks=False, drift=True,
        drift_start=10, drift_end=40, drift_strength=1,
    )
    assert set(normal_only["label"].unique()) == {"normal"}
    assert set(drift["label"].unique()) == {"normal"}
    assert len(drift) > 0


def test_load_does_synthetic_csv_round_trip(tmp_path):
    csv_path = tmp_path / "synth.csv"
    frame = generate_synthetic_data(
        seed=102, attacks=True, warmup=2, attack_start=2, n_cycles=1,
        attack_duration=3, attack_gap=1, tail_duration=2,
    )
    frame.to_csv(csv_path, index=False)
    loaded = load(csv_path, "synth")
    assert len(loaded) == len(frame)
    assert loaded["label"].isin(
        ["normal", "portscan", "ddos", "exfiltration"]
    ).all()
