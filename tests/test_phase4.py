import json

import numpy as np

from ata.baseline import AdaptiveBaseline
from ata.drift import DriftMonitor, population_stability_index
from ata.features import FEATURE_COLUMNS, window_features
from ata.run import summarize_evaluation
from ata.synth import generate_synthetic_csv, generate_synthetic_data


def test_psi_identical_and_shifted_samples():
    reference = np.linspace(0, 1, 200)
    assert population_stability_index(reference, reference) == 0.0
    assert population_stability_index(reference, reference + 10) > 0.25


def test_constant_and_degenerate_features_are_skipped_and_nonfinite_is_safe():
    reference = np.column_stack((np.ones(20), np.arange(20, dtype=float)))
    monitor = DriftMonitor(
        reference,
        min_features=1,
        test_windows=4,
        check_every=1,
        cooldown=0,
        feature_names=["constant", "varying"],
    )
    for row in ([1.0, 100.0], [1.0, 101.0], [np.inf, 102.0], [np.nan, 103.0]):
        monitor.update(row)
    result = monitor.check()
    assert result.statistics["constant"] is None
    assert np.isfinite(result.statistics["varying"])
    assert result.n_drifted == 1


def test_monitor_persistence_and_cooldown():
    monitor = DriftMonitor(
        np.arange(20, dtype=float).reshape(-1, 1),
        min_features=1,
        test_windows=4,
        persistence=2,
        check_every=1,
        cooldown=3,
        psi_threshold=0.1,
    )
    for index in range(4):
        assert not monitor.update([100.0 + index], position=index)
    assert monitor.update([104.0], position=4)
    assert len(monitor.events) == 1
    assert monitor.events[0]["psi_values"]
    assert not monitor.update([105.0], position=5)
    assert not monitor.update([106.0], position=6)
    assert not monitor.update([107.0], position=7)
    fired = [
        monitor.update([100.0 + index], position=index)
        for index in range(8, 13)
    ]
    assert any(fired)
    assert len(monitor.events) == 2


def test_ks_mode_detects_a_shift_and_named_drift_requires_post_drift_tail():
    reference = np.linspace(0, 1, 100).reshape(-1, 1)
    monitor = DriftMonitor(
        reference,
        method="ks",
        min_features=1,
        test_windows=10,
        persistence=1,
        check_every=1,
        cooldown=0,
        psi_threshold=0.5,
    )
    fired = [
        monitor.update([10.0 + index], position=index)
        for index in range(10)
    ]
    assert any(fired)

    try:
        generate_synthetic_data(seconds=500, seed=4, scenario="drift")
    except ValueError as exc:
        assert "post-drift tail" in str(exc)
    else:
        raise AssertionError("A truncated named drift scenario must be rejected.")


def test_monitor_uses_feature_rows_only_and_ignores_nonfinite_values():
    rows = np.arange(66, dtype=float).reshape(6, 11)
    monitor = DriftMonitor(
        rows[:4],
        min_features=1,
        test_windows=2,
        check_every=1,
        cooldown=0,
        feature_names=FEATURE_COLUMNS,
    )
    monitor.update(rows[4], position=4)
    monitor.update(rows[5], position=5)
    assert set(monitor.last_result.statistics) == set(FEATURE_COLUMNS)
    assert "label" not in monitor.last_result.statistics
    assert isinstance(monitor.update([np.nan] * 11, position=6), bool)


def test_named_scenarios_are_deterministic_and_sidecar_has_drift_fields(tmp_path):
    first = generate_synthetic_data(seed=4, scenario="drift")
    second = generate_synthetic_data(seed=4, scenario="drift")
    assert first.equals(second)
    assert first["label"].eq("normal").all()

    output = tmp_path / "drift.csv"
    generate_synthetic_csv(output, seed=4, scenario="drift")
    sidecar = json.loads(
        output.with_suffix(".csv.meta.json").read_text(encoding="utf-8")
    )
    assert sidecar["drift_start"] is not None
    assert sidecar["drift_end"] is not None
    assert sidecar["duration_seconds"] > sidecar["drift_end"]
    assert sidecar["attacks"] is False


def test_periodic_trigger_preserves_output_and_drift_trigger_disables_periodic_refits():
    flows = generate_synthetic_data(
        seed=7, seconds=90, attacks=False, warmup=10
    )
    windows = window_features(flows)[FEATURE_COLUMNS]
    defaults = dict(
        warmup=10,
        history=20,
        refit_every=5,
        contamination=0.05,
        seed=0,
    )
    implicit = AdaptiveBaseline(**defaults)
    explicit = AdaptiveBaseline(**defaults, refit_trigger="periodic")
    implicit_result = implicit.run(windows)
    explicit_result = explicit.run(windows)
    assert implicit_result.equals(explicit_result)
    assert implicit.n_refits == explicit.n_refits

    drift = AdaptiveBaseline(
        **defaults,
        refit_trigger="drift",
        psi_threshold=1e9,
        min_drift_features=1,
        test_windows=4,
        check_every=1,
        drift_cooldown=0,
    )
    drift.run(windows)
    assert drift.n_refits == 0
    assert drift.drift_events == []


def test_drift_trigger_adapts_to_persistent_shift_using_recent_stream_rows():
    rng = np.random.default_rng(19)
    warmup = rng.normal(0.0, 0.1, size=(20, 4))
    shifted = rng.normal(4.0, 0.1, size=(30, 4))
    rows = np.vstack((warmup, shifted))
    model = AdaptiveBaseline(
        warmup=20,
        history=20,
        refit_every=5,
        contamination=0.05,
        seed=0,
        refit_trigger="drift",
        min_drift_features=1,
        psi_threshold=0.1,
        drift_persistence=1,
        test_windows=5,
        check_every=1,
        drift_cooldown=0,
    )

    model.run(rows)

    assert model.drift_events
    assert model.drift_refit_events
    assert model.n_refits == len(model.drift_refit_events)
    assert model.drift_refit_events[0]["rows_used"] == 20
    np.testing.assert_allclose(
        model.drift_monitor.reference,
        np.vstack(model._recent_stream_rows[-20:]),
    )


def test_periodic_smoke_seed_matches_phase3r_metric_fingerprint():
    windows = window_features(
        generate_synthetic_data(seed=100), window=1
    )
    model = AdaptiveBaseline(
        warmup=120,
        history=300,
        refit_every=30,
        contamination=0.005,
        seed=0,
    )
    results = model.run(windows[FEATURE_COLUMNS])
    metrics = summarize_evaluation(
        windows["label"].iloc[120:],
        results["flag"].iloc[120:],
    )
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (269, 3, 1)
    assert model.n_refits == 26
