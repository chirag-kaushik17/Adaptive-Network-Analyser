import pandas as pd
import pytest

from ata.features import WindowAggregator, window_features


def _make_flow(
    ts, src="10.0.0.1", dst="10.0.0.2", dport=80, proto="tcp",
    bytes_=100, packets=3, duration=0.2, syn=0, label="normal",
):
    return {
        "ts": ts, "src": src, "dst": dst, "dport": dport, "proto": proto,
        "bytes": bytes_, "packets": packets, "duration": duration,
        "syn": syn, "label": label,
    }


def test_batch_and_incremental_features_match_in_values_order_and_indices():
    df = pd.DataFrame(
        [
            _make_flow(0.1),
            _make_flow(0.2, src="10.0.0.2", dport=443),
            _make_flow(1.0, label="PortScan", syn=1, dport=22),
            _make_flow(1.2, label="DDoS", src="10.0.0.8", dport=80),
            _make_flow(1.4, label="normal", src="10.0.0.8", dport=80),
            _make_flow(3.1, label="exfiltration", bytes_=500_000),
        ]
    )
    batch = window_features(df, window=1)
    aggregator = WindowAggregator(window=1)
    aggregator.add_many(df.to_dict("records"))
    ready = aggregator.pop_ready()
    final = aggregator.flush()
    combined = pd.concat([ready, final], ignore_index=True).set_index("window")
    pd.testing.assert_frame_equal(batch, combined)
    assert list(batch.index) == [0, 1, 3]


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
    assert result.loc[0, "flows"] == 1
    assert result.loc[1, "flows"] == 2
    assert result.loc[2, "flows"] == 2

    missing = pd.DataFrame([_make_flow(0.1), _make_flow(2.0)])
    assert list(window_features(missing, window=1).index) == [0, 2]


def test_out_of_order_error_flush_once_and_adding_after_flush_errors():
    unsorted = pd.DataFrame([_make_flow(2.0), _make_flow(1.0)])
    with pytest.raises(ValueError, match="chronological"):
        window_features(unsorted, window=1)

    aggregator = WindowAggregator(window=1)
    with pytest.raises(ValueError, match="chronological"):
        aggregator.add_many([_make_flow(2.0), _make_flow(1.0)])

    aggregator = WindowAggregator(window=1)
    aggregator.add_many([_make_flow(0.1), _make_flow(0.2)])
    first = aggregator.flush()
    second = aggregator.flush()
    assert len(first) == 1
    assert second.empty
    with pytest.raises(ValueError, match="already been flushed"):
        aggregator.add(_make_flow(1.0))


def test_attack_label_or_threshold_and_lexical_tie_breaking():
    group = pd.DataFrame(
        [
            _make_flow(0.1, label="PortScan"),
            _make_flow(0.2, label="PortScan"),
            _make_flow(0.3, label="DDoS"),
            _make_flow(0.4, label="DDoS"),
            _make_flow(0.5, label="normal"),
            _make_flow(0.6, label="normal"),
        ]
    )
    by_count = window_features(
        group, window=1, min_attack_flows=3, min_attack_frac=0.75
    )
    assert by_count.iloc[0]["label"] == "ddos"

    by_fraction = window_features(
        group, window=1, min_attack_flows=5, min_attack_frac=0.02
    )
    assert by_fraction.iloc[0]["label"] == "ddos"

    no_threshold = window_features(
        group, window=1, min_attack_flows=5, min_attack_frac=0.75
    )
    assert no_threshold.iloc[0]["label"] == "normal"


def test_portscan_window_has_many_ports_and_high_syn_ratio():
    flows = pd.DataFrame(
        [
            _make_flow(0.1, dport=1234, syn=1, label="PortScan"),
            _make_flow(0.2, dport=4321, syn=1, label="PortScan"),
            _make_flow(0.3, dport=9876, syn=1, label="PortScan"),
            _make_flow(0.4, dport=80, syn=0, label="normal"),
            _make_flow(0.5, dport=53, syn=0, label="normal"),
        ]
    )
    result = window_features(flows, window=1)
    assert result.iloc[0]["uniq_dport"] >= 3
    assert result.iloc[0]["syn_ratio"] > 0.5
    assert result.iloc[0]["label"] == "portscan"
