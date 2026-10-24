"""SHAP shape compatibility and compact feature-attribution helpers."""

from __future__ import annotations

import numpy as np


def shap_for_predicted_class(
    values,
    *,
    n_samples: int,
    n_features: int,
    n_classes: int,
    predicted_classes,
) -> np.ndarray:
    """Normalize SHAP multiclass outputs to rows x features for each prediction."""
    if hasattr(values, "values"):
        values = values.values
    predictions = np.asarray(predicted_classes, dtype=int)
    if predictions.shape != (n_samples,):
        raise ValueError("predicted_classes must contain one class index per sample.")
    if np.any(predictions < 0) or np.any(predictions >= n_classes):
        raise ValueError("predicted class index is outside the SHAP class dimension.")

    if isinstance(values, list):
        if len(values) != n_classes:
            raise ValueError(
                f"Expected {n_classes} SHAP class arrays; received {len(values)}."
            )
        class_values = np.stack([np.asarray(item) for item in values], axis=-1)
    else:
        class_values = np.asarray(values)

    if class_values.shape == (n_samples, n_features, n_classes):
        return class_values[np.arange(n_samples), :, predictions]
    if class_values.shape == (n_classes, n_samples, n_features):
        class_values = np.moveaxis(class_values, 0, -1)
        return class_values[np.arange(n_samples), :, predictions]
    if class_values.shape == (n_samples, n_classes, n_features):
        class_values = np.moveaxis(class_values, 1, -1)
        return class_values[np.arange(n_samples), :, predictions]
    raise ValueError(
        "Unsupported multiclass SHAP shape "
        f"{class_values.shape}; expected ({n_samples}, {n_features}, {n_classes})."
    )


def top_features(
    feature_names,
    feature_values,
    shap_row,
    *,
    limit: int = 3,
) -> list[dict]:
    """Return the largest absolute attributions with JSON-safe scalar values."""
    names = list(feature_names)
    values = np.asarray(feature_values, dtype=float)
    attributions = np.asarray(shap_row, dtype=float)
    if values.shape != attributions.shape or values.shape != (len(names),):
        raise ValueError("Feature values, SHAP values, and feature names must align.")
    if limit < 1 or limit > len(names):
        raise ValueError("limit must be between 1 and the number of features.")
    selected = np.argsort(-np.abs(attributions), kind="stable")[:limit]
    return [
        {
            "feature": str(names[index]),
            "value": float(values[index]),
            "shap_value": float(attributions[index]),
        }
        for index in selected
    ]
