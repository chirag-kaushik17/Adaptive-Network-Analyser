import json

from ata.alerts import describe_alert
from ata.dashboard import (
    format_alert_row,
    load_dashboard_data,
    load_report,
    valid_api_url,
)


def test_dashboard_module_imports_and_validates_api_urls():
    assert valid_api_url("http://127.0.0.1:8000")
    assert valid_api_url("https://localhost/api")
    assert not valid_api_url("not-a-url")
    assert not valid_api_url("file:///tmp/report.json")


def test_missing_api_is_explicit_and_does_not_create_metrics(tmp_path):
    snapshot = load_dashboard_data(
        "http://127.0.0.1:1",
        report_dir=tmp_path,
        timeout_seconds=0.1,
    )
    assert snapshot["health"] is None
    assert snapshot["stats"] is None
    assert snapshot["alerts"] is None
    assert snapshot["health_error"]
    assert snapshot["reports"]["metrics.json"][0] is None
    assert snapshot["reports"]["classifier_metrics.json"][0] is None


def test_report_loader_handles_missing_valid_and_malformed_reports(tmp_path):
    missing, missing_error = load_report(str(tmp_path / "missing.json"))
    assert missing is None and missing_error

    valid_path = tmp_path / "valid.json"
    valid_path.write_text(json.dumps({"research_acceptance": {"verdict": "NO-GO"}}))
    report, error = load_report(str(valid_path))
    assert error is None
    assert report["research_acceptance"]["verdict"] == "NO-GO"

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("{broken")
    invalid, invalid_error = load_report(str(invalid_path))
    assert invalid is None and invalid_error

    list_path = tmp_path / "list.json"
    list_path.write_text("[1, 2]")
    invalid_root, root_error = load_report(str(list_path))
    assert invalid_root is None
    assert root_error == "Report root must be a JSON object."


def test_alert_table_and_explanation_show_optional_fields_as_unavailable():
    alert = {
        "alert_id": "alert-1",
        "window_index": 12,
        "predicted_class": "unknown",
        "layer1_score": 0.2,
        "top_features": [],
        "drift_metadata": None,
    }
    row = format_alert_row(alert)
    assert row["alert_id"] == "alert-1"
    assert row["severity"] == "unavailable"
    assert row["class_probability"] == "unavailable"
    assert row["top_features"] == "unavailable"
    assert "no classifier" in describe_alert(alert)


def test_dashboard_loads_actual_historical_reports_without_changing_gate():
    from ata.dashboard import ROOT

    snapshot = load_dashboard_data(
        "bad-url",
        report_dir=ROOT / "reports",
        timeout_seconds=0.1,
    )
    phase5, phase5_error = snapshot["reports"]["metrics.json"]
    phase6, phase6_error = snapshot["reports"]["classifier_metrics.json"]
    assert phase5_error is None
    assert phase6_error is None
    assert phase5["research_acceptance"]["verdict"] == "NO-GO"
    assert phase6["layer1_research_acceptance"]["status"] == "FAIL"
