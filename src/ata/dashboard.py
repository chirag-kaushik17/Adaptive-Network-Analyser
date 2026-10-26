"""Streamlit view of live ATA API telemetry and saved evaluation reports."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import pandas as pd
import streamlit as st

from ata.alerts import describe_alert

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_API_URL = "http://127.0.0.1:8000"


@st.cache_data(ttl=5, show_spinner=False)
def fetch_json(
    url: str,
    timeout_seconds: float = 2.0,
    params: tuple[tuple[str, str], ...] = (),
) -> tuple[Any | None, str | None]:
    """Fetch a JSON endpoint with a short cache and explicit failure details."""
    try:
        response = httpx.get(
            url,
            params=dict(params),
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        return response.json(), None
    except (httpx.HTTPError, ValueError) as exc:
        return None, str(exc)


@st.cache_data(ttl=30, show_spinner=False)
def load_report(path_text: str) -> tuple[dict[str, Any] | None, str | None]:
    path = Path(path_text)
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, str(exc)
    if not isinstance(content, dict):
        return None, "Report root must be a JSON object."
    return content, None


def valid_api_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def format_alert_row(alert: dict[str, Any]) -> dict[str, Any]:
    features = alert.get("top_features")
    if isinstance(features, list):
        feature_text = ", ".join(
            f"{item.get('feature', 'unavailable')} "
            f"(SHAP {item.get('shap_value', 'unavailable')})"
            for item in features
            if isinstance(item, dict)
        ) or "unavailable"
    else:
        feature_text = "unavailable"
    return {
        "alert_id": alert.get("alert_id", "unavailable"),
        "event_timestamp_seconds": alert.get("event_timestamp", "unavailable"),
        "window_index": alert.get("window_index", "unavailable"),
        "predicted_attack_type": alert.get("predicted_class", "unavailable"),
        "layer1_flagged": alert.get("layer1_flagged", "unavailable"),
        "anomaly_score": alert.get("layer1_score", "unavailable"),
        "class_probability": alert.get("class_probability", "unavailable"),
        "top_features": feature_text,
        "drift_status": (
            "drift event"
            if alert.get("drift_metadata")
            else ("no event recorded" if "drift_metadata" in alert else "unavailable")
        ),
    }


def load_dashboard_data(
    api_url: str,
    report_dir: str | Path = ROOT / "reports",
    *,
    timeout_seconds: float = 2.0,
    alert_params: tuple[tuple[str, str], ...] = (("limit", "1000"),),
) -> dict[str, Any]:
    if not valid_api_url(api_url):
        health, health_error = None, "Enter a valid http:// or https:// API URL."
        stats, stats_error = None, health_error
        alerts, alerts_error = None, health_error
    else:
        root = api_url.rstrip("/")
        health, health_error = fetch_json(
            f"{root}/health", timeout_seconds=timeout_seconds
        )
        stats, stats_error = fetch_json(
            f"{root}/stats", timeout_seconds=timeout_seconds
        )
        alerts, alerts_error = fetch_json(
            f"{root}/alerts",
            timeout_seconds=timeout_seconds,
            params=alert_params,
        )
    reports = {}
    for filename in ("metrics.json", "classifier_metrics.json"):
        reports[filename] = load_report(
            str(Path(report_dir) / filename)
        )
    return {
        "health": health,
        "health_error": health_error,
        "stats": stats,
        "stats_error": stats_error,
        "alerts": alerts,
        "alerts_error": alerts_error,
        "reports": reports,
    }


def _metric(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _render_live(snapshot: dict[str, Any], api_url: str) -> None:
    st.header("System overview")
    health = snapshot["health"]
    stats = snapshot["stats"]
    if health is None:
        st.error(f"API unavailable: {snapshot['health_error']}")
    elif health.get("status") == "ok":
        st.success(
            f"API healthy — version {health.get('version', 'unavailable')}; "
            f"Layer 1 ready: {health.get('pipeline_ready', 'unavailable')}; "
            f"Layer 2 loaded: "
            f"{health.get('models_loaded', {}).get('layer2', 'unavailable')}"
        )
    else:
        st.warning("API returned a non-healthy status.")

    if stats is None:
        st.info(f"Operational counters unavailable: {snapshot['stats_error']}")
        return
    columns = st.columns(4)
    for column, label, key in zip(
        columns,
        ("Windows processed", "Alerts stored", "Drift events", "Refits"),
        ("windows_processed", "stored_alerts", "drift_events", "refits"),
    ):
        column.metric(label, _metric(stats.get(key)))

    st.subheader("Live/replayed traffic telemetry")
    st.caption(
        f"Source: recent windows returned by {api_url}. These are current API "
        "session/replay observations, not historical evaluation data."
    )
    records = stats.get("recent_windows")
    if not isinstance(records, list) or not records:
        st.info("No window telemetry has been returned by the API yet.")
        return
    frame = pd.DataFrame(records)
    if "event_timestamp" in frame and "flow_count" in frame:
        st.caption("Traffic volume by flow-window (stream-relative seconds).")
        st.line_chart(frame.set_index("event_timestamp")[["flow_count"]])
    if "event_timestamp" in frame and "score" in frame:
        scored = frame.dropna(subset=["score"])
        if not scored.empty:
            st.caption("Layer 1 anomaly score by stream-relative window time.")
            st.line_chart(scored.set_index("event_timestamp")[["score"]])
    if "event_timestamp" in frame and "flagged" in frame:
        frame["flagged_count"] = frame["flagged"].astype(int)
        st.caption("Flagged (1) versus unflagged (0) window decisions.")
        st.bar_chart(frame.set_index("event_timestamp")[["flagged_count"]])
        drift = (
            frame[frame["drift_event"].astype(bool)]
            if "drift_event" in frame
            else frame.iloc[0:0]
        )
        if not drift.empty:
            st.caption("Drift events visible in the returned telemetry.")
            st.dataframe(
                drift[["window_index", "event_timestamp", "score"]],
                hide_index=True,
                use_container_width=True,
            )
    st.caption(
        "Throughput is measured from process start and is not a real-time "
        "capacity guarantee."
    )
    st.write(
        f"Flows/s: {_metric(stats.get('flows_per_second'))}; "
        f"Windows/s: {_metric(stats.get('windows_per_second'))}"
    )


def _render_alerts(snapshot: dict[str, Any], api_url: str) -> None:
    st.header("Alert explorer")
    response = snapshot["alerts"]
    if response is None:
        st.warning(f"Alerts unavailable: {snapshot['alerts_error']}")
        return
    alerts = response.get("items", []) if isinstance(response, dict) else []
    if not alerts:
        st.info("The API returned no alerts for this query.")
        return
    attack_types = sorted(
        {
            str(alert.get("predicted_class", "unknown"))
            for alert in alerts
        }
    )
    selected_type = st.selectbox("Attack type", ["all", *attack_types])
    selected_limit = st.selectbox("Result count", [25, 50, 100, 250, 500, 1000], index=2)
    since_enabled = st.checkbox("Filter by stream-relative event time")
    since = st.number_input("Minimum event second", min_value=0.0, value=0.0, disabled=not since_enabled)
    params: list[tuple[str, str]] = [("limit", str(selected_limit))]
    if selected_type != "all":
        params.append(("attack_type", selected_type))
    if since_enabled:
        params.append(("since", str(since)))
    filtered_response, error = fetch_json(
        f"{api_url.rstrip('/')}/alerts",
        params=tuple(params),
    )
    if error:
        st.warning(f"Could not refresh filtered alerts: {error}")
        return
    filtered = filtered_response.get("items", [])
    rows = [format_alert_row(alert) for alert in filtered]
    st.dataframe(rows, hide_index=True, use_container_width=True)
    alert_ids = [alert.get("alert_id") for alert in filtered if alert.get("alert_id")]
    if not alert_ids:
        return
    selected_id = st.selectbox("Inspect alert", alert_ids)
    detail, detail_error = fetch_json(
        f"{api_url.rstrip('/')}/alerts/{selected_id}"
    )
    if detail_error or detail is None:
        st.warning(f"Alert detail unavailable: {detail_error}")
        return
    st.subheader("Alert explanation")
    st.write(describe_alert(detail))
    st.caption(
        "A model prediction is an indication for investigation, not proof of a "
        "confirmed cyberattack."
    )
    st.json(
        {
            "alert_id": detail.get("alert_id", "unavailable"),
            "window_index": detail.get("window_index", "unavailable"),
            "event_timestamp": detail.get("event_timestamp", "unavailable"),
            "layer1_flagged": detail.get("layer1_flagged", "unavailable"),
            "drift_metadata": detail.get("drift_metadata", "unavailable"),
            "top_features": detail.get("top_features", "unavailable"),
        }
    )


def _render_historical(snapshot: dict[str, Any]) -> None:
    st.header("Historical model performance")
    layer1, layer1_error = snapshot["reports"]["metrics.json"]
    layer2, layer2_error = snapshot["reports"]["classifier_metrics.json"]
    if layer1 is None:
        st.info(f"Phase 5 report unavailable or invalid: {layer1_error}")
    else:
        st.caption("Historical synthetic Layer 1 evaluation (not live API data).")
        st.write("Research acceptance:", layer1.get("research_acceptance", "unavailable"))
        scenarios = layer1.get("scenarios")
        if isinstance(scenarios, dict):
            for scenario_name, models in scenarios.items():
                st.subheader(scenario_name.replace("_", " ").title())
                summary = {}
                if isinstance(models, dict):
                    for model_name, values in models.items():
                        mean_std = values.get("mean_std", {})
                        summary[model_name] = {
                            metric: (
                                mean_std[metric].get("mean")
                                if isinstance(mean_std.get(metric), dict)
                                else None
                            )
                            for metric in ("false_alarm_rate", "recall", "f1")
                        }
                if summary:
                    st.dataframe(
                        pd.DataFrame.from_dict(summary, orient="index"),
                        use_container_width=True,
                    )
        plot_path = ROOT / "reports" / "adaptive_vs_static.png"
        if plot_path.is_file():
            st.image(str(plot_path), caption="Saved Layer 1 comparison plot")

    if layer2 is None:
        st.info(f"Phase 6 report unavailable or invalid: {layer2_error}")
        return
    st.caption("Historical synthetic Layer 2 classifier evaluation.")
    st.write(
        "Layer 1 research acceptance:",
        layer2.get("layer1_research_acceptance", "unavailable"),
    )
    st.write("Selected model:", layer2.get("selected_model", "unavailable"))
    for model_name, model in layer2.get("models", {}).items():
        st.subheader(model_name.replace("_", " ").title())
        per_class = model.get("per_class_mean_std", {})
        rows = {}
        for class_name, metrics in per_class.items():
            rows[class_name] = {
                metric: metrics.get(metric, {}).get("mean")
                for metric in ("precision", "recall", "f1")
            }
        if rows:
            st.dataframe(pd.DataFrame.from_dict(rows, orient="index"))
        confusion_path = ROOT / "reports" / f"confusion_matrix_{model_name}.png"
        if confusion_path.is_file():
            st.image(str(confusion_path), caption=f"{model_name} confusion matrix")
    shap_path = ROOT / "reports" / "shap_summary.png"
    if shap_path.is_file():
        st.image(str(shap_path), caption="Saved SHAP global summary")


def main() -> None:
    st.set_page_config(page_title="ATA dashboard", layout="wide")
    st.title("Adaptive Network Traffic Analyzer")
    st.caption("Local streaming and historical evaluation dashboard")
    st.sidebar.header("Connection")
    default_url = os.environ.get("ATA_API_URL", DEFAULT_API_URL)
    api_url = st.sidebar.text_input("ATA API base URL", value=default_url).strip()
    refresh = st.sidebar.button("Refresh now")
    if refresh:
        fetch_json.clear()
        load_report.clear()
    if not valid_api_url(api_url):
        st.error("Enter a valid API URL using http:// or https://.")
        return
    snapshot = load_dashboard_data(api_url)
    _render_live(snapshot, api_url)
    _render_alerts(snapshot, api_url)
    _render_historical(snapshot)
    st.sidebar.caption(
        "The API has no authentication. Keep it bound to localhost; do not "
        "expose it on a public network."
    )


if __name__ == "__main__":
    main()
