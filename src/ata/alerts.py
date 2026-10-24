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
    class_probability: float,
    top_features: list[dict],
    model_name: str,
    model_version: str,
) -> dict[str, Any]:
    if len(top_features) != 3:
        raise ValueError("An alert must contain exactly three top features.")
    clean_features = [
        {
            "feature": str(item["feature"]),
            "value": float(item["value"]),
            "shap_value": float(item["shap_value"]),
        }
        for item in top_features
    ]
    return {
        "alert_id": str(alert_id),
        "window_index": int(window_index),
        "layer1_score": None if layer1_score is None else float(layer1_score),
        "predicted_class": str(predicted_class),
        "class_probability": float(class_probability),
        "top_features": clean_features,
        "model_name": str(model_name),
        "model_version": str(model_version),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
