"""Tests for simplified ATA detection design:
- Static Layer 1 default does not refit
- Layer 2 runs on every window and alerts on unflagged windows
- Layer 1-only flag with Layer 2 'normal' produces 'unknown_anomaly'
- No-classifier mode still works as expected
"""

import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from ata.alerts import build_alert_record, describe_alert
from ata.baseline import StaticBaseline, AdaptiveBaseline
from ata.features import FEATURE_COLUMNS
from ata.pipeline import AtaPipeline


def _flow(ts, *, dport=80, bytes_val=100, label=None):
    flow = {
        "ts": float(ts),
        "src": "192.168.1.10",
        "dst": "10.0.0.1",
        "dport": int(dport),
        "proto": "tcp",
        "bytes": float(bytes_val),
        "packets": 1.0,
        "duration": 0.1,
        "syn": 1,
    }
    if label is not None:
        flow["label"] = label
    return flow


def test_static_default_does_not_refit():
    pipeline = AtaPipeline(warmup=2, history=4, refit_every=1)
    assert pipeline.layer1_mode == "static"
    assert isinstance(pipeline.baseline, StaticBaseline)

    # Process 10 windows (each 1s)
    flows = [_flow(i + 0.1) for i in range(10)]
    alerts = pipeline.process_flows(flows)
    alerts += pipeline.flush()

    assert pipeline.windows_processed >= 10
    assert pipeline.baseline.n_refits == 0
    assert pipeline.stats()["refits"] == 0
    assert "promotions" not in pipeline.stats()


def test_explicit_adaptive_and_drift_modes():
    pipe_adapt = AtaPipeline(warmup=2, history=4, refit_every=1, layer1_mode="adaptive")
    assert pipe_adapt.layer1_mode == "adaptive"
    assert isinstance(pipe_adapt.baseline, AdaptiveBaseline)
    assert pipe_adapt.baseline.refit_trigger == "periodic"

    pipe_drift = AtaPipeline(warmup=2, history=4, refit_every=1, layer1_mode="drift")
    assert pipe_drift.layer1_mode == "drift"
    assert isinstance(pipe_drift.baseline, AdaptiveBaseline)
    assert pipe_drift.baseline.refit_trigger == "drift"


def test_layer2_runs_on_unflagged_window_and_can_alert():
    # Mock classifier that predicts "portscan" (class index 1)
    mock_clf = MagicMock()
    mock_clf.predict.return_value = np.array([1])
    mock_clf.predict_proba.return_value = np.array([[0.05, 0.90, 0.03, 0.02]])

    mock_explainer = MagicMock()
    mock_explainer.shap_values.return_value = [
        np.zeros((1, len(FEATURE_COLUMNS))) for _ in range(4)
    ]
    # Set positive SHAP for portscan
    mock_explainer.shap_values.return_value[1] = np.full((1, len(FEATURE_COLUMNS)), 0.2)

    pipeline = AtaPipeline(
        warmup=2,
        history=4,
        classifier=mock_clf,
        explainer=mock_explainer,
        model_name="xgboost",
    )

    # Send warm-up flows to complete 2 windows (window 0 and window 1)
    pipeline.process_flows([_flow(0.1), _flow(1.1), _flow(2.0)])
    assert pipeline.ready

    # Send flow for window 2 (ts 2.1) and close it with ts 3.0
    # Layer 1 considers this normal (unflagged), but mock_clf predicts portscan
    alerts = pipeline.process_flows([_flow(2.1), _flow(3.0)])
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert["predicted_class"] == "portscan"
    assert alert["class_probability"] == pytest.approx(0.90)
    assert alert["layer1_flagged"] is False
    assert len(alert["top_features"]) == 3
    summary = describe_alert(alert)
    assert "Layer 1 did not flag the window" in summary


def test_layer1_only_flagged_window_with_layer2_normal_gives_unknown_anomaly():
    mock_clf = MagicMock()
    mock_clf.predict.return_value = np.array([0])
    mock_clf.predict_proba.return_value = np.array([[0.95, 0.02, 0.02, 0.01]])

    pipeline = AtaPipeline(
        warmup=5,
        history=10,
        contamination=0.1,
        classifier=mock_clf,
        model_name="xgboost",
    )

    # Warm-up with slightly varying normal flows (complete 5 windows)
    warmup_flows = [_flow(i + 0.1, bytes_val=100 + i * 10) for i in range(5)] + [_flow(5.0)]
    pipeline.process_flows(warmup_flows)
    assert pipeline.ready

    # Window 5: massive anomaly in bytes to trigger Layer 1 flag, closed by flow at 6.0
    alerts = pipeline.process_flows([
        _flow(5.1, bytes_val=100000000.0),
        _flow(6.0, bytes_val=100),
    ])

    assert len(alerts) == 1
    alert = alerts[0]
    assert alert["predicted_class"] == "unknown_anomaly"
    assert alert["class_probability"] is None
    assert alert["top_features"] == []
    assert alert["layer1_flagged"] is True

    summary = describe_alert(alert)
    assert "Layer 2 classified it as normal" in summary


def test_no_classifier_mode_still_works():
    pipeline = AtaPipeline(
        warmup=5,
        history=10,
        contamination=0.1,
        classifier=None,
    )

    # Warm up (complete 5 windows)
    warmup_flows = [_flow(i + 0.1, bytes_val=100 + i * 10) for i in range(5)] + [_flow(5.0)]
    pipeline.process_flows(warmup_flows)
    assert pipeline.ready

    # Normal window: unflagged, no alert
    alerts1 = pipeline.process_flows([_flow(5.1, bytes_val=120), _flow(6.0, bytes_val=100)])
    assert len(alerts1) == 0

    # Anomalous window: flagged by Layer 1, closed by flow at 7.0 -> alert with predicted_class="unknown"
    alerts2 = pipeline.process_flows([
        _flow(6.1, bytes_val=100000000.0),
        _flow(7.0, bytes_val=100),
    ])
    assert len(alerts2) == 1
    alert = alerts2[0]
    assert alert["predicted_class"] == "unknown"
    assert alert["top_features"] == []
    assert alert["layer1_flagged"] is True
    assert alert["model_name"] == "unavailable"
    assert "no classifier" in describe_alert(alert)
