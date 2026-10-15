import pandas as pd

from ata.loader import load
from ata.synth import generate_synthetic_data


def test_cic_loader_handles_messy_labels_and_invalid_timestamps(tmp_path):
    csv_path = tmp_path / "cic.csv"
    rows = [
        {
            "Timestamp": "Wednesday-07-07-2017 03:01:00",
            "Source IP": "10.0.0.1",
            "Destination IP": "10.0.0.2",
            "Destination Port": 80,
            "Protocol": 6,
            "Flow Duration": 1000000,
            "Total Fwd Packets": 2,
            "Total Bwd Packets": 1,
            "Total Length of Fwd Packets": 50,
            "Total Length of Bwd Packets": 30,
            "SYN Flag Count": 1,
            "Label": "BENIGN",
        },
        {
            "Timestamp": "Wednesday-07-07-2017 03:02:00",
            "Source IP": "10.0.0.1",
            "Destination IP": "10.0.0.2",
            "Destination Port": 443,
            "Protocol": 6,
            "Flow Duration": 2000000,
            "Total Fwd Packets": 3,
            "Total Bwd Packets": 1,
            "Total Length of Fwd Packets": 80,
            "Total Length of Bwd Packets": 45,
            "SYN Flag Count": 0,
            "Label": "PortScan",
        },
        {
            "Timestamp": "bad-ts",
            "Source IP": "10.0.0.1",
            "Destination IP": "10.0.0.2",
            "Destination Port": 22,
            "Protocol": 6,
            "Flow Duration": 1000,
            "Total Fwd Packets": 1,
            "Total Bwd Packets": 1,
            "Total Length of Fwd Packets": 10,
            "Total Length of Bwd Packets": 5,
            "SYN Flag Count": 0,
            "Label": "BENIGN",
        },
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    df, stats = load(csv_path, "cicids2017", return_stats=True)

    assert list(df.columns) == ["ts", "src", "dst", "dport", "proto", "bytes", "packets", "duration", "syn", "label"]
    assert len(df) == 2
    assert df.iloc[0]["ts"] == 0.0
    assert df.iloc[0]["label"] == "normal"
    assert df.iloc[1]["label"] == "portscan"
    assert df.iloc[1]["dport"] == 443
    assert stats["discarded_invalid_timestamps"] == 1


def test_synthetic_data_is_deterministic_and_handles_no_attacks():
    first = generate_synthetic_data(seconds=60, seed=7, attack_start=30, attacks=True)
    second = generate_synthetic_data(seconds=60, seed=7, attack_start=30, attacks=True)
    assert first.equals(second)

    normal_only = generate_synthetic_data(seconds=60, seed=7, attack_start=30, attacks=False)
    assert set(normal_only["label"].unique()) == {"normal"}

    drift = generate_synthetic_data(seconds=60, seed=7, attack_start=30, attacks=False, drift=True)
    assert set(drift["label"].unique()) == {"normal"}
    assert len(drift) > 0


def test_load_does_synthetic_csv_round_trip(tmp_path):
    csv_path = tmp_path / "synth.csv"
    df = generate_synthetic_data(seconds=30, seed=3, attack_start=15, attacks=True)
    df.to_csv(csv_path, index=False)
    loaded = load(csv_path, "synth")
    assert len(loaded) == len(df)
    assert loaded["label"].isin(["normal", "PortScan", "DDoS", "Exfiltration"]).all()
