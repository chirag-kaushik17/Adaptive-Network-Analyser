"""Supervised Layer 2 classifiers and chronological synthetic datasets."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .features import FEATURE_COLUMNS, window_features
from .synth import generate_synthetic_data

CLASS_NAMES = ("normal", "portscan", "ddos", "exfiltration")
MODEL_NAMES = ("xgboost", "random_forest")


def make_dataset(
    seed: int,
    *,
    warmup: int = 120,
    n_cycles: int = 5,
    train_fraction: float = 0.65,
    gap_windows: int = 5,
) -> tuple[pd.DataFrame, list[int], list[int]]:
    """Build shuffled synthetic windows and a chronological train/test split."""
    from .evaluate import time_ordered_split

    flows = generate_synthetic_data(
        seed=seed,
        scenario="stable_attacks",
        attack_order="shuffled",
        n_cycles=n_cycles,
        warmup=warmup,
        window=1.0,
    )
    windows = window_features(flows, window=1.0)
    features = windows.loc[:, FEATURE_COLUMNS]
    if list(features.columns) != FEATURE_COLUMNS:
        raise AssertionError("Classifier input must match FEATURE_COLUMNS exactly.")
    train_indices, test_indices = time_ordered_split(
        windows.index.tolist(),
        train_fraction=train_fraction,
        gap=gap_windows,
    )
    missing = {
        partition: sorted(set(CLASS_NAMES) - set(windows.loc[indices, "label"]))
        for partition, indices in (
            ("train", train_indices),
            ("test", test_indices),
        )
    }
    if any(missing.values()):
        raise ValueError(
            "Synthetic classifier split is missing classes; increase n_cycles "
            f"rather than changing the chronological split: {missing}"
        )
    return windows, train_indices, test_indices


def class_counts(labels) -> dict[str, int]:
    values = pd.Series(labels, dtype="string")
    return {
        name: int(values.eq(name).sum())
        for name in CLASS_NAMES
    }


def build_model(
    model_name: str,
    *,
    seed: int,
    parameters: Mapping[str, object],
):
    """Construct one fixed-seed classifier from recorded configuration."""
    if model_name == "xgboost":
        return XGBClassifier(
            objective="multi:softprob",
            num_class=len(CLASS_NAMES),
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=1,
            random_state=seed,
            **dict(parameters),
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            class_weight="balanced",
            n_jobs=1,
            random_state=seed,
            **dict(parameters),
        )
    raise ValueError(f"Unsupported classifier model: {model_name!r}")


def fit_model(model, model_name: str, features, labels):
    """Fit with class weights calculated from training labels only."""
    encoded = np.asarray(
        [CLASS_NAMES.index(str(label)) for label in labels],
        dtype=np.int32,
    )
    if model_name == "xgboost":
        weights = compute_sample_weight(class_weight="balanced", y=encoded)
        model.fit(features, encoded, sample_weight=weights)
    else:
        model.fit(features, encoded)
    return model


def classification_metrics(y_true, y_pred) -> dict:
    """Return fixed-vocabulary per-class and macro precision/recall/F1."""
    true_values = np.asarray(y_true, dtype=np.int32)
    predicted_values = np.asarray(y_pred, dtype=np.int32)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_values,
        predicted_values,
        labels=np.arange(len(CLASS_NAMES)),
        zero_division=0,
    )
    matrix = confusion_matrix(
        true_values,
        predicted_values,
        labels=np.arange(len(CLASS_NAMES)),
    )
    by_class = {
        name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, name in enumerate(CLASS_NAMES)
    }
    return {
        "per_class": by_class,
        "macro": {
            "precision": float(np.mean(precision)),
            "recall": float(np.mean(recall)),
            "f1": float(np.mean(f1)),
        },
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def score_candidate(
    model_name: str,
    parameters: Mapping[str, object],
    datasets: Mapping[int, tuple[pd.DataFrame, list[int], list[int]]],
) -> float:
    """Average macro F1 across development-seed chronological test splits."""
    scores = []
    for seed, (windows, train_indices, test_indices) in datasets.items():
        model = build_model(model_name, seed=0, parameters=parameters)
        fit_model(
            model,
            model_name,
            windows.loc[train_indices, FEATURE_COLUMNS].to_numpy(dtype=float),
            windows.loc[train_indices, "label"].tolist(),
        )
        predicted = model.predict(
            windows.loc[test_indices, FEATURE_COLUMNS].to_numpy(dtype=float)
        )
        truth = np.asarray(
            [CLASS_NAMES.index(value) for value in windows.loc[test_indices, "label"]],
            dtype=np.int32,
        )
        scores.append(classification_metrics(truth, predicted)["macro"]["f1"])
    return float(np.mean(scores))
