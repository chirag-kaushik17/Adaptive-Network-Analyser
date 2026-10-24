import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ata.api import create_app
from ata.baseline import AdaptiveBaseline
from ata.bench import run_benchmark
from ata.features import FEATURE_COLUMNS, window_features
from ata.pipeline import AtaPipeline, validate_flow_batch
from ata.replay import replay
from ata.store import AlertStore
from ata.stream import FlowStream, iter_flow_batches
from ata.synth import generate_synthetic_csv


def _flow(ts, *, bytes_value=100, label=None):
    flow = {
        "ts": ts,
        "src": "10.0.0.1",
        "dst": "10.0.0.2",
        "dport": 443,
        "proto": "tcp",
        "bytes": bytes_value,
        "packets": 2,
        "duration": 0.2,
        "syn": 0,
    }
    if label is not None:
        flow["label"] = label
    return flow


def _alert(alert_id, window_index, attack_type="portscan"):
    return {
        "alert_id": alert_id,
        "window_index": window_index,
        "event_timestamp": float(window_index),
        "layer1_score": 0.12,
        "predicted_class": attack_type,
        "class_probability": 0.9,
        "top_features": [
            {"feature": "dport_entropy", "value": 1.5, "shap_value": 0.3},
            {"feature": "uniq_dport", "value": 3, "shap_value": -0.2},
            {"feature": "syn_ratio", "value": 0.8, "shap_value": 0.1},
        ],
        "model_name": "random_forest",
        "model_version": "test",
        "created_at": "2026-10-24T00:00:00+00:00",
        "drift_metadata": None,
    }


def test_flow_validation_is_chronological_and_batch_atomic():
    valid = [_flow(0), _flow(1)]
    assert len(validate_flow_batch(valid)) == 2
    pipeline = AtaPipeline(warmup=2, history=4)
    with pytest.raises(ValueError, match="out of order"):
        pipeline.process_flows([_flow(0), _flow(2), _flow(1)])
    assert pipeline.flows_processed == 0
    assert pipeline.windows_processed == 0
    assert pipeline.last_flow_ts is None


def test_pipeline_streaming_windows_and_scores_match_batch_workflow():
    flows = [
        _flow(0.1, bytes_value=100),
        _flow(0.4, bytes_value=200),
        _flow(1.1, bytes_value=300),
        _flow(1.9, bytes_value=400),
        _flow(2.2, bytes_value=500),
        _flow(3.1, bytes_value=600),
        _flow(4.2, bytes_value=700),
    ]
    batch_windows = window_features(pd.DataFrame(flows), window=1.0)
    expected = AdaptiveBaseline(
        warmup=2,
        history=5,
        refit_every=30,
        contamination=0.1,
        seed=3,
    ).run(batch_windows[FEATURE_COLUMNS])
    pipeline = AtaPipeline(
        window=1.0,
        warmup=2,
        history=5,
        refit_every=30,
        contamination=0.1,
        model_seed=3,
        telemetry_limit=20,
    )
    alerts = []
    for flow in flows:
        alerts.extend(pipeline.process_flows([flow]))
    alerts.extend(pipeline.flush())
    actual = pipeline.recent_windows

    assert [row["window_index"] for row in actual] == batch_windows.index.tolist()
    for row, (_, feature_row) in zip(actual, batch_windows.iterrows()):
        for feature in FEATURE_COLUMNS:
            assert row["features"][feature] == pytest.approx(feature_row[feature])
    actual_scores = np.asarray(
        [np.nan if row["score"] is None else row["score"] for row in actual]
    )
    np.testing.assert_allclose(actual_scores, expected["score"], equal_nan=True)
    assert [row["flagged"] for row in actual] == expected["flag"].tolist()
    assert pipeline.flush() == []
    assert len(alerts) == pipeline.alerts_generated
    assert all(alert["predicted_class"] == "unknown" for alert in alerts)
    assert all(alert["top_features"] == [] for alert in alerts)


def test_pipeline_rejects_non_finite_and_invalid_flow_fields_without_mutation():
    pipeline = AtaPipeline(warmup=2, history=4)
    for flow in (_flow(float("nan")), {**_flow(0), "proto": "gre"}):
        with pytest.raises(ValueError):
            pipeline.process_flows([flow])
    assert pipeline.flows_processed == 0


def test_alert_store_round_trip_filter_pagination_and_injection_string():
    store = AlertStore(":memory:")
    store.add(_alert("a1", 1))
    store.add(_alert("a2", 2, "ddos"))
    store.add(_alert("a3", 3))
    assert store.get("a2") == _alert("a2", 2, "ddos")
    assert [item["alert_id"] for item in store.list(limit=1)] == ["a3"]
    assert [item["alert_id"] for item in store.list(limit=2, offset=1)] == [
        "a2",
        "a1",
    ]
    assert [item["alert_id"] for item in store.list(attack_type="ddos")] == ["a2"]
    assert store.list(attack_type="' OR 1=1 --") == []
    assert store.list(since=2.0)[0]["alert_id"] == "a3"
    assert store.list(limit=1, offset=20) == []
    assert store.count() == 3
    store.clear()
    assert store.count() == 0
    store.close()


def test_api_validation_order_flush_score_read_only_and_alert_routes():
    store = AlertStore(":memory:")
    app = create_app(
        {
            "database": ":memory:",
            "store": store,
            "warmup": 2,
            "history": 5,
            "refit_every": 30,
            "contamination": 0.1,
            "max_batch_size": 4,
        }
    )
    with TestClient(app) as client:
        assert client.get("/health").json()["pipeline_ready"] is False
        response = client.post(
            "/flows",
            json={"flows": [_flow(0), _flow(1), _flow(2)]},
        )
        assert response.status_code == 200
        assert response.json()["flows_accepted"] == 3
        assert response.json()["windows_processed"] == 2
        assert client.get("/health").json()["pipeline_ready"] is True

        stats_before = client.get("/stats").json()
        features = {name: 1.0 for name in FEATURE_COLUMNS}
        score = client.post("/score", json=features)
        assert score.status_code == 200
        stats_after = client.get("/stats").json()
        for field in ("flows_processed", "windows_processed", "flags", "refits"):
            assert stats_before[field] == stats_after[field]

        rejected = client.post(
            "/flows",
            json={"flows": [_flow(4), _flow(3)]},
        )
        assert rejected.status_code == 409
        assert client.get("/stats").json()["flows_processed"] == 3
        assert client.post(
            "/flows", json={"flows": [{**_flow(3), "dport": 65536}]}
        ).status_code == 422
        assert client.post(
            "/flows", json={"flows": [{**_flow(3), "bytes": -1}]}
        ).status_code == 422
        assert client.post(
            "/flows", json={"flows": [{**_flow(3), "proto": "gre"}]}
        ).status_code == 422
        assert client.post(
            "/flows",
            json={"flows": [_flow(3), _flow(4), _flow(5), _flow(6), _flow(7)]},
        ).status_code == 413
        assert client.get("/alerts").status_code == 200
        assert client.get("/alerts/not-present").status_code == 404
        assert client.get("/stats").json()["stored_alerts"] == 0
        assert client.post("/flush").json()["pipeline_closed"] is True
        assert client.post("/flows", json={"flows": [_flow(3)]}).status_code == 422

    store.close()


def test_api_concurrent_same_timestamp_requests_preserve_flow_count():
    app = create_app(
        {
            "database": ":memory:",
            "warmup": 2,
            "history": 4,
            "refit_every": 30,
            "contamination": 0.1,
        }
    )

    def send_one(_):
        with TestClient(app) as client:
            return client.post("/flows", json={"flows": [_flow(0)]}).status_code

    with ThreadPoolExecutor(max_workers=4) as executor:
        statuses = list(executor.map(send_one, range(8)))
    assert statuses == [200] * 8
    assert app.state.pipeline.flows_processed == 8
    app.state.alert_store.close()


def test_flow_stream_batches_and_empty_csv(tmp_path):
    csv_path = tmp_path / "flows.csv"
    generate_synthetic_csv(
        csv_path,
        seconds=6,
        seed=10,
        attacks=False,
        warmup=2,
        tail_duration=4,
    )
    flows = list(FlowStream(csv_path, dataset="synth"))
    assert flows
    assert [flow["ts"] for flow in flows] == sorted(flow["ts"] for flow in flows)
    batches = list(iter_flow_batches(iter(flows), 17))
    assert sum(map(len, batches)) == len(flows)
    assert max(map(len, batches)) <= 17

    empty = tmp_path / "empty.csv"
    pd.DataFrame(columns=["ts", "src", "dst", "dport", "proto", "bytes", "packets", "duration", "syn", "label"]).to_csv(empty, index=False)
    assert list(FlowStream(empty, dataset="synth")) == []


def test_replay_reports_connection_errors_without_silent_success(tmp_path):
    csv_path = tmp_path / "flows.csv"
    generate_synthetic_csv(
        csv_path,
        seconds=6,
        seed=11,
        attacks=False,
        warmup=2,
        tail_duration=4,
    )
    report_path = tmp_path / "replay.json"
    with pytest.raises(RuntimeError, match="Replay failed"):
        replay(
            csv_path,
            url="http://127.0.0.1:1",
            batch_size=25,
            output=report_path,
            timeout=0.1,
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "FAIL"
    assert report["error"]


def test_benchmark_writes_measured_stage_report(tmp_path):
    csv_path = tmp_path / "flows.csv"
    generate_synthetic_csv(
        csv_path,
        seconds=8,
        seed=12,
        attacks=False,
        warmup=2,
        tail_duration=6,
    )
    report_path = tmp_path / "throughput.json"
    report = run_benchmark(csv_path, batch_size=200, output=report_path)
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["status"] == "PASS"
    assert report["flow_count"] > 0
    assert report["window_count_batch_aggregation"] == report["window_count_pipeline"]
    assert report["window_count_batch_aggregation"] == report["window_count_api"]
    assert set(report["stages"]) == {
        "window_aggregation",
        "full_pipeline",
        "api_ingestion",
    }
    assert "TestClient" in report["stages"]["api_ingestion"]["transport"]
