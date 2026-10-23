import numpy as np
import pandas as pd
import pytest

from ata.baseline import AdaptiveBaseline, StaticBaseline
from ata.run import summarize_evaluation

FEATURES = [
    "flows", "bytes_total", "bytes_mean", "pkts_mean", "dur_mean",
    "syn_ratio", "uniq_src", "uniq_dst", "uniq_dport", "dport_entropy",
    "top_src_share",
]


def _stable_matrix(rows=30, columns=4):
    rng = np.random.default_rng(100)
    return rng.normal(0, 0.03, size=(rows, columns))


def test_real_outlier_is_flagged_and_never_enters_adaptive_buffer():
    baseline = AdaptiveBaseline(
        warmup=40, history=20, refit_every=1, contamination=0.05, seed=0
    )
    normal = _stable_matrix(rows=40)
    for index, row in enumerate(normal):
        baseline.step(row, index=index)

    outlier = np.array([1000.0, -1200.0, 900.0, 1500.0])
    score, flag = baseline.step(outlier, index=9001)

    assert score > 0
    assert flag is True
    assert 9001 not in baseline.buffer_indices
    assert not any(np.array_equal(row, outlier) for row in baseline.buffer)


def test_warmup_output_run_matches_step_and_preserves_custom_indices():
    X = pd.DataFrame(_stable_matrix(rows=12, columns=5), index=range(100, 112))
    adaptive = AdaptiveBaseline(
        warmup=4, history=5, refit_every=2, contamination=0.1, seed=0
    )
    run_df = adaptive.run(X)

    repeated = AdaptiveBaseline(
        warmup=4, history=5, refit_every=2, contamination=0.1, seed=0
    )
    stepped = [
        repeated.step(row.to_numpy(), index=index)
        for index, row in X.iterrows()
    ]
    stepped_df = pd.DataFrame(stepped, index=X.index, columns=["score", "flag"])

    assert run_df.index.equals(X.index)
    assert run_df["score"].iloc[:4].isna().all()
    assert run_df["flag"].iloc[:4].tolist() == [False] * 4
    assert run_df["score"].equals(stepped_df["score"])
    assert run_df["flag"].equals(stepped_df["flag"])
    assert adaptive.buffer_indices[:4] == list(X.index[:4])


def test_static_does_not_grow_buffer_or_refit():
    static = StaticBaseline(
        warmup=5, history=5, refit_every=2, contamination=0.1, seed=0
    )
    static.run(_stable_matrix(rows=12))
    assert len(static.buffer_indices) == static.warmup
    assert static.n_refits == 0


def test_refit_interval_counts_flagged_windows_and_history_can_be_shorter():
    baseline = AdaptiveBaseline(
        warmup=8, history=4, refit_every=2, contamination=0.05, seed=0
    )
    rows = _stable_matrix(rows=8)
    for row in rows:
        baseline.step(row)
    baseline.step(np.array([1000.0] * 4))
    baseline.step(np.array([-900.0] * 4))
    assert baseline._post_warmup_seen == 2
    assert baseline.n_refits == 1


def test_refit_is_skipped_until_minimum_eligible_rows_are_available():
    baseline = AdaptiveBaseline(
        warmup=2, history=5, refit_every=1, contamination=0.1, seed=0,
        min_fit_rows=4,
    )
    values = _stable_matrix(rows=5, columns=3)
    baseline.step(values[0])
    baseline.step(values[1])
    baseline.step(values[2])
    assert baseline.skipped_refits == 1


def test_score_only_does_not_change_state_and_none_disables_refits():
    X = _stable_matrix(rows=12)
    baseline = AdaptiveBaseline(
        warmup=5, history=5, refit_every=None, contamination=0.1, seed=0
    )
    for index, row in enumerate(X[:5]):
        baseline.step(row, index=index)
    before_indices = list(baseline.buffer_indices)
    before_buffer = [row.copy() for row in baseline.buffer]
    before_seen = baseline._post_warmup_seen
    before_refits = baseline.n_refits
    first = baseline.score_only(X[5])
    second = baseline.score_only(X[5])
    assert first == second
    assert baseline.buffer_indices == before_indices
    assert all(np.array_equal(a, b) for a, b in zip(before_buffer, baseline.buffer))
    assert baseline._post_warmup_seen == before_seen
    assert baseline.n_refits == before_refits
    for row in X[5:]:
        baseline.step(row)
    assert baseline.n_refits == 0


def test_parameter_nonfinite_and_insufficient_data_validation():
    with pytest.raises(ValueError, match="warmup"):
        AdaptiveBaseline(warmup=1, history=2, refit_every=1, contamination=0.1)
    with pytest.raises(ValueError, match="history"):
        AdaptiveBaseline(warmup=3, history=1, refit_every=1, contamination=0.1)
    with pytest.raises(ValueError, match="refit_every"):
        AdaptiveBaseline(warmup=3, history=2, refit_every=0, contamination=0.1)
    with pytest.raises(ValueError, match="contamination"):
        AdaptiveBaseline(warmup=3, history=2, refit_every=1, contamination=0)
    with pytest.raises(ValueError, match="min_fit_rows"):
        AdaptiveBaseline(
            warmup=3, history=2, refit_every=1, contamination=0.1,
            min_fit_rows=3,
        )
    baseline = AdaptiveBaseline(
        warmup=3, history=2, refit_every=1, contamination=0.1
    )
    with pytest.raises(ValueError, match="non-finite"):
        baseline.step([0.0, np.nan, 1.0])
    with pytest.raises(ValueError, match="Insufficient warm-up"):
        baseline.run(_stable_matrix(rows=2))


def test_normal_only_scenario_has_numeric_false_alarm_rate():
    baseline = AdaptiveBaseline(
        warmup=5, history=10, refit_every=2, contamination=0.05, seed=0
    )
    values = _stable_matrix(rows=20, columns=11)
    results = baseline.run(values)
    metrics = summarize_evaluation(
        ["normal"] * len(values), results["flag"].tolist()
    )
    assert isinstance(metrics["false_alarm_rate"], float)
    assert 0 <= metrics["false_alarm_rate"] <= 1
    assert metrics["tp"] == 0


def test_evaluation_metrics_use_n_a_for_zero_denominators():
    empty = summarize_evaluation([], [])
    no_normal = summarize_evaluation(["portscan"], [True])
    assert empty["precision"] == "n/a"
    assert empty["recall"] == "n/a"
    assert empty["false_alarm_rate"] == "n/a"
    assert no_normal["false_alarm_rate"] == "n/a"
