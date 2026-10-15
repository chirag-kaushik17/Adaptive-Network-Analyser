import pandas as pd
import pytest

from ata.features import WindowAggregator, window_features
from ata.synth import generate_synthetic_data


def _make_flow(ts, src="10.0.0.1", dst="10.0.0.2", dport=80, proto="tcp", bytes_=100, packets=3, duration=0.2, syn=0, label="normal"):
    return {
        "ts": ts,
        "src": src,
        "dst": dst,
        "dport": dport,
        "proto": proto,
        "bytes": bytes_,
        "packets": packets,
        "duration": duration,
        "syn": syn,
        "label": label,
    }


def test_batch_and_incremental_features_match_on_synthetic_data():
    df = generate_synthetic_data(seconds=20, seed=2, attack_start=10, attacks=True)
    batch = window_features(df, window=1)
    aggregator = WindowAggregator(window=1)
    aggregator.add_many(df.to_dict("records"))
    ready = aggregator.pop_ready()
    final = aggregator.flush()
    combined = pd.concat([ready, final], ignore_index=True).sort_values("window").reset_index(drop=True)
    batch_out = batch.reset_index().rename(columns={"index": "window"}).reset_index(drop=True)
    merged = combined.reset_index(drop=True)
    assert merged.equals(batch_out)


def test_half_open_boundaries_and_empty_window_skipping():
    df = pd.DataFrame(
        [
            _make_flow(0.1),
            _make_flow(1.0),
            _make_flow(1.9),
            _make_flow(2.0),
            _make_flow(2.7),
        ]
    )
    result = window_features(df, window=1)
    assert list(result.index) == [0, 1, 2]
    assert result.loc[1, "flows"] == 2
    assert result.loc[2, "flows"] == 2


def test_out_of_order_error_and_double_flush_and_label_rule():
    unsorted = pd.DataFrame([_make_flow(2.0), _make_flow(1.0)])
    with pytest.raises(ValueError, match="chronological"):
        window_features(unsorted, window=1)

    aggregator = WindowAggregator(window=1)
    aggregator.add_many([_make_flow(0.1), _make_flow(0.2), _make_flow(0.3)])
    first = aggregator.flush()
    second = aggregator.flush()
    assert len(first) == 1
    assert second.empty

    attack_group = pd.DataFrame(
        [
            _make_flow(0.1, label="PortScan"),
            _make_flow(0.2, label="PortScan"),
            _make_flow(0.3, label="DDoS"),
            _make_flow(0.4, label="DDoS"),
            _make_flow(0.5, label="normal"),
            _make_flow(0.6, label="normal"),
        ]
    )
    out = window_features(attack_group, window=1, min_attack_flows=3, min_attack_frac=0.02)
    assert out.iloc[0]["label"] == "ddos"


def test_portscan_window_has_high_uniq_dport_and_syn_ratio():
    portscan_window = pd.DataFrame(
        [
            _make_flow(0.1, dport=1234, syn=1, label="PortScan"),
            _make_flow(0.2, dport=4321, syn=1, label="PortScan"),
            _make_flow(0.3, dport=9876, syn=1, label="PortScan"),
            _make_flow(0.4, dport=80, syn=0, label="normal"),
            _make_flow(0.5, dport=53, syn=0, label="normal"),
        ]
    )
    out = window_features(portscan_window, window=1)
    assert out.iloc[0]["uniq_dport"] >= 3
    assert out.iloc[0]["syn_ratio"] > 0.5
    assert out.iloc[0]["label"] == "portscan"
