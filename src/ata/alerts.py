"""Alert record contract

An alert is a plain JSON-serializable dictionary.
It contains an `alert_id` and source `window_index`.
`layer1_score` records the unsupervised Layer 1 score for that window.
`predicted_class` and `class_probability` come from the selected Layer 2 model.
`top_features` contains exactly three feature/value/SHAP-value objects.
`model_name` and `model_version` identify the classifier used.
`created_at` is an ISO-8601 UTC timestamp.
Numeric values are converted to built-in Python scalars, not NumPy scalar types.
Labels, timestamps, and schedule metadata are never included as model features.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def build_alert_record(
    *,
    alert_id: str,
    window_index: int,
    layer1_score: float | None,
    predicted_class: str,
    class_probability: float | None,
    top_features: list[dict] | None = None,
    model_name: str,
    model_version: str | None,
    layer1_flagged: bool | None = None,
) -> dict[str, Any]:
    classified = model_name != "unavailable"
    feature_list = [] if top_features is None else top_features
    if predicted_class == "unknown_anomaly":
        if feature_list:
            raise ValueError("unknown_anomaly alerts cannot contain SHAP feature values.")
    elif classified:
        if len(feature_list) != 3:
            raise ValueError("An alert must contain exactly three top features.")
    elif feature_list:
        raise ValueError("Unclassified alerts cannot contain SHAP feature values.")
    clean_features = [
        {
            "feature": str(item["feature"]),
            "value": float(item["value"]),
            "shap_value": (
                None
                if item.get("shap_value") is None
                else float(item["shap_value"])
            ),
        }
        for item in feature_list
    ]
    record = {
        "alert_id": str(alert_id),
        "window_index": int(window_index),
        "layer1_score": None if layer1_score is None else float(layer1_score),
        "predicted_class": str(predicted_class),
        "class_probability": (
            None if class_probability is None else float(class_probability)
        ),
        "top_features": clean_features,
        "model_name": str(model_name),
        "model_version": None if model_version is None else str(model_version),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if layer1_flagged is not None:
        record["layer1_flagged"] = bool(layer1_flagged)
    return record


def describe_alert(alert: dict[str, Any]) -> str:
    """Summarize only the prediction and attributions already present in an alert."""
    predicted = alert.get("predicted_class", "unknown")
    score = alert.get("layer1_score")
    probability = alert.get("class_probability")
    layer1_flagged = alert.get("layer1_flagged")
    if predicted == "unknown":
        return (
            f"Layer 1 flagged this window (score: {score}); no classifier "
            "or feature attribution was available."
        )
    if predicted == "unknown_anomaly":
        return (
            f"Layer 1 flagged this window as anomalous (score: {score}), "
            "but Layer 2 classified it as normal; no attack class or "
            "feature attribution was assigned."
        )
    if layer1_flagged is False:
        summary = (
            f"Layer 2 predicted {predicted} "
            f"(probability: {probability}); Layer 1 did not flag the window "
            f"(score: {score})."
        )
    else:
        summary = (
            f"Layer 2 predicted {predicted} "
            f"(probability: {probability}) after Layer 1 flagged the window "
            f"(score: {score})."
        )
    features = alert.get("top_features") or []
    if features:
        contributions = ", ".join(
            f"{item['feature']}={item['value']} (SHAP {item['shap_value']:+.4g})"
            for item in features
        )
        summary += (
            " Supplied SHAP contributions for this class: "
            f"{contributions}. Positive values push toward the prediction; "
            "negative values push away."
        )
    else:
        summary += " No SHAP attributions were supplied."
    return summary
