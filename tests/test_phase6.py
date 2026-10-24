import json

import numpy as np
import pytest
import shap

from ata.alerts import build_alert_record
from ata.classifier import (
    CLASS_NAMES,
    build_model,
    fit_model,
    make_dataset,
)
from ata.explain import shap_for_predicted_class, top_features
from ata.features import FEATURE_COLUMNS


def test_classifier_split_is_chronological_and_all_classes_are_present():
    windows, train_indices, test_indices = make_dataset(
        200,
        warmup=120,
        n_cycles=5,
        train_fraction=0.65,
        gap_windows=5,
    )
    assert max(train_indices) < min(test_indices)
    assert set(windows.loc[train_indices, "label"]) == set(CLASS_NAMES)
    assert set(windows.loc[test_indices, "label"]) == set(CLASS_NAMES)


def test_classifier_features_are_exactly_feature_columns_without_label_or_time():
    assert len(FEATURE_COLUMNS) == 11
    assert "label" not in FEATURE_COLUMNS
    assert "window" not in FEATURE_COLUMNS
    assert "timestamp" not in FEATURE_COLUMNS

    windows, _, _ = make_dataset(201, n_cycles=5)
    classifier_features = windows.loc[:, FEATURE_COLUMNS]
    assert classifier_features.columns.tolist() == FEATURE_COLUMNS
    assert "label" not in classifier_features.columns
    assert "window" not in classifier_features.columns


def test_shap_adapter_supports_list_and_sample_feature_class_array():
    n_samples, n_features, n_classes = 2, 3, 4
    class_arrays = [
        np.full((n_samples, n_features), class_index, dtype=float)
        for class_index in range(n_classes)
    ]
    predictions = np.array([2, 1])
    expected = np.vstack(
        (
            np.full(n_features, 2.0),
            np.full(n_features, 1.0),
        )
    )
    from_list = shap_for_predicted_class(
        class_arrays,
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        predicted_classes=predictions,
    )
    assert from_list.shape == (n_samples, n_features)
    np.testing.assert_array_equal(from_list, expected)

    class_array = np.stack(class_arrays, axis=-1)
    from_array = shap_for_predicted_class(
        class_array,
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        predicted_classes=predictions,
    )
    assert from_array.shape == (n_samples, n_features)
    np.testing.assert_array_equal(from_array, expected)


def test_shap_adapter_rejects_unexpected_multiclass_shape():
    with pytest.raises(ValueError, match="Unsupported multiclass SHAP shape"):
        shap_for_predicted_class(
            np.zeros((2, 4)),
            n_samples=2,
            n_features=3,
            n_classes=4,
            predicted_classes=[0, 1],
        )


@pytest.mark.parametrize(
    "model_name,parameters",
    [
        ("xgboost", {"n_estimators": 3, "max_depth": 2, "learning_rate": 0.1}),
        ("random_forest", {"n_estimators": 5, "max_depth": 3}),
    ],
)
def test_shap_adapter_normalizes_real_tree_explainer_outputs(model_name, parameters):
    rng = np.random.default_rng(11)
    features = rng.normal(size=(48, 3))
    labels = [CLASS_NAMES[index % len(CLASS_NAMES)] for index in range(len(features))]
    model = build_model(model_name, seed=7, parameters=parameters)
    fit_model(model, model_name, features, labels)
    samples = features[:6]
    predictions = np.asarray(model.predict(samples), dtype=int)
    raw_values = shap.TreeExplainer(model).shap_values(samples)

    normalized = shap_for_predicted_class(
        raw_values,
        n_samples=len(samples),
        n_features=samples.shape[1],
        n_classes=len(CLASS_NAMES),
        predicted_classes=predictions,
    )

    assert normalized.shape == samples.shape
    assert np.isfinite(normalized).all()


def test_alert_is_json_serializable_and_has_exactly_three_shap_features():
    record = build_alert_record(
        alert_id="ata-xgboost-200-900",
        window_index=np.int64(900),
        layer1_score=np.float64(0.12),
        predicted_class="portscan",
        class_probability=np.float64(0.98),
        top_features=top_features(
            FEATURE_COLUMNS[:4],
            np.array([1, 2, 3, 4], dtype=float),
            np.array([0.1, -0.7, 0.3, 0.2], dtype=float),
        ),
        model_name="xgboost",
        model_version="3.2.0",
    )
    assert len(record["top_features"]) == 3
    assert record["top_features"][0]["feature"] == FEATURE_COLUMNS[1]
    assert json.loads(json.dumps(record)) == record
    assert set(record) == {
        "alert_id",
        "window_index",
        "layer1_score",
        "predicted_class",
        "class_probability",
        "top_features",
        "model_name",
        "model_version",
        "created_at",
    }


def test_alert_rejects_a_top_feature_list_of_the_wrong_size():
    with pytest.raises(ValueError, match="exactly three"):
        build_alert_record(
            alert_id="bad",
            window_index=1,
            layer1_score=0.1,
            predicted_class="ddos",
            class_probability=0.9,
            top_features=[],
            model_name="random_forest",
            model_version="1.0",
        )
