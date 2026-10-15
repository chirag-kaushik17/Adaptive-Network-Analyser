import numpy as np
import pandas as pd
import pytest

from ata.baseline import AdaptiveBaseline, StaticBaseline
from ata.features import window_features
from ata.run import summarize_evaluation


@pytest.fixture
def normal_feature_frame():
    rows = []
    for i in range(8):
        rows.append({
            "ts": float(i),
            "src": f"10.0.0.{1 + (i % 3)}",
            "dst": "10.0.0.1",
            "dport": 80,
            "proto": "tcp",
            "bytes": 100 + i,
            "packets": 2,
            "duration": 0.1,
            "syn": 0,
            "label": "normal",
        })
    return pd.DataFrame(rows)


def test_adaptive_baseline_warmup_outputs_and_buffer_protection():
    rows = []
    for i in range(6):
        rows.append({
            "ts": float(i),
            "src": "10.0.0.1",
            "dst": "10.0.0.2",
            "dport": 80,
            "proto": "tcp",
            "bytes": 100,
            "packets": 2,
            "duration": 0.1,
            "syn": 0,
            "label": "normal",
        })
    rows[3] = {**rows[3], "bytes": 100000, "packets": 100, "label": "PortScan"}
    df = pd.DataFrame(rows)
    windows = window_features(df, window=1)
    feature_matrix = windows[["flows", "bytes_total", "bytes_mean", "pkts_mean", "dur_mean", "syn_ratio", "uniq_src", "uniq_dst", "uniq_dport", "dport_entropy", "top_src_share"]]
    base = AdaptiveBaseline(warmup=3, history=3, refit_every=1, contamination=0.05, seed=7)

    def force_flag(row):
        if np.allclose(row, feature_matrix.iloc[3].to_numpy()):
            return 9.0, True
        return 0.0, False

    base._score_flag = force_flag
    scores = base.run(feature_matrix)

    assert scores["score"].iloc[:3].isna().all()
    assert scores["flag"].iloc[:3].tolist() == [False, False, False]
    assert 3 not in base.buffer_indices
    assert len(base.buffer_indices) <= len(scores)


def test_run_matches_step_and_preserves_custom_indices():
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(8, 10)))
    X.index = [10, 11, 12, 13, 14, 15, 16, 17]
    adaptive = AdaptiveBaseline(warmup=3, history=3, refit_every=2, contamination=0.1, seed=9)
    run_df = adaptive.run(X)

    second = AdaptiveBaseline(warmup=3, history=3, refit_every=2, contamination=0.1, seed=9)
    stepped = []
    for idx, row in X.iterrows():
        stepped.append(second.step(row.to_numpy(), index=idx))
    stepped_df = pd.DataFrame(stepped, index=X.index, columns=["score", "flag"])
    assert run_df.index.equals(stepped_df.index)
    assert run_df["score"].equals(stepped_df["score"])
    assert run_df["flag"].equals(stepped_df["flag"])


def test_static_baseline_never_refits_and_validation_fails():
    X = pd.DataFrame(np.random.default_rng(4).normal(size=(8, 5)))
    static = StaticBaseline(warmup=3, history=2, refit_every=2, contamination=0.1, seed=1)
    static.run(X)
    assert static._refit_calls == 0

    with pytest.raises(ValueError, match="warmup"):
        AdaptiveBaseline(warmup=1, history=2, refit_every=1, contamination=0.1, seed=1)
    with pytest.raises(ValueError, match="history"):
        AdaptiveBaseline(warmup=3, history=0, refit_every=1, contamination=0.1, seed=1)
    with pytest.raises(ValueError, match="contamination"):
        AdaptiveBaseline(warmup=3, history=2, refit_every=1, contamination=0.0, seed=1)
    with pytest.raises(ValueError, match="non-finite"):
        AdaptiveBaseline(warmup=3, history=2, refit_every=1, contamination=0.1, seed=1).step([0.0, np.nan, 1.0])


def test_flagged_windows_count_toward_refit_interval_and_normal_only_summary_is_numeric():
    X = pd.DataFrame(
        np.vstack(
            [
                np.zeros((3, 4)),
                np.full((1, 4), 1000.0),
                np.zeros((3, 4)),
            ]
        )
    )
    adaptive = AdaptiveBaseline(warmup=3, history=3, refit_every=2, contamination=0.1, seed=8)
    adaptive.run(X)
    assert adaptive._post_warmup_seen == 4
    assert adaptive._refit_calls >= 1

    labels = pd.Series(["normal"] * 6)
    metrics = summarize_evaluation(labels, [False, False, False, False, False, False])
    assert metrics["false_alarm_rate"] in {0.0, "n/a"}
