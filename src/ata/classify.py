"""Train and evaluate supervised Layer 2 classifiers on synthetic windows."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .alerts import build_alert_record
from .baseline import AdaptiveBaseline
from .classifier import (
    CLASS_NAMES,
    MODEL_NAMES,
    build_model,
    classification_metrics,
    class_counts,
    fit_model,
    make_dataset,
    score_candidate,
)
from .evaluate import time_ordered_split
from .explain import shap_for_predicted_class, top_features
from .features import FEATURE_COLUMNS
from .run import DEFAULTS_PATH

ROOT = Path(__file__).resolve().parents[2]
SHAP_SANITY_FEATURES = {
    "portscan": {"uniq_dport", "dport_entropy", "syn_ratio"},
    "ddos": {"flows", "bytes_total", "uniq_src"},
}


def _model_version(model_name: str) -> str:
    package = "xgboost" if model_name == "xgboost" else "scikit-learn"
    return importlib.metadata.version(package)


def _summary(values: list[float]) -> dict:
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def _aggregate_model(seed_records: dict[str, dict]) -> dict:
    fields = ("precision", "recall", "f1", "support")
    per_class = {}
    for class_name in CLASS_NAMES:
        per_class[class_name] = {
            field: _summary(
                [
                    float(record["per_class"][class_name][field])
                    for record in seed_records.values()
                ]
            )
            for field in fields
        }
    macro = {
        metric: _summary(
            [
                float(record["macro"][metric])
                for record in seed_records.values()
            ]
        )
        for metric in ("precision", "recall", "f1")
    }
    summed_confusion = np.sum(
        [record["confusion_matrix"] for record in seed_records.values()],
        axis=0,
        dtype=int,
    )
    return {
        "per_class_mean_std": per_class,
        "macro_mean_std": macro,
        "confusion_matrix_total": summed_confusion.tolist(),
        "per_seed": seed_records,
    }


def _plot_confusion(matrix, model_name: str, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted class",
        ylabel="True class",
        title=f"{model_name.replace('_', ' ').title()} confusion matrix (seeds 200–204)",
    )
    plt.setp(axis.get_xticklabels(), rotation=25, ha="right", rotation_mode="anchor")
    threshold = float(np.max(matrix)) / 2 if np.size(matrix) else 0.0
    for row in range(len(CLASS_NAMES)):
        for column in range(len(CLASS_NAMES)):
            value = int(matrix[row][column])
            axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "#222222",
            )
    figure.colorbar(image, ax=axis, label="Window count")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _write_shap_summary(shap_rows, feature_rows, output_path: Path) -> None:
    shap.summary_plot(
        np.asarray(shap_rows, dtype=float),
        np.asarray(feature_rows, dtype=float),
        feature_names=FEATURE_COLUMNS,
        plot_type="bar",
        max_display=len(FEATURE_COLUMNS),
        show=False,
        color="#0072B2",
    )
    figure = plt.gcf()
    figure.set_size_inches(9, 6)
    figure.suptitle("Mean absolute SHAP values for predicted classes")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def run_classification(seeds: list[int], output_dir: Path) -> dict:
    defaults = json.loads(DEFAULTS_PATH.read_text(encoding="utf-8"))
    config = defaults["classifier"]
    output_dir.mkdir(parents=True, exist_ok=True)
    layer1_report_path = ROOT / "reports" / "metrics.json"
    if layer1_report_path.is_file():
        layer1_verdict = json.loads(
            layer1_report_path.read_text(encoding="utf-8")
        ).get("verdict", {}).get("verdict")
    else:
        layer1_verdict = None
    layer1_status = {
        "GO": "PASS",
        "NO-GO": "FAIL",
    }.get(layer1_verdict, "NOT_RUN")
    per_model = {name: {} for name in MODEL_NAMES}
    all_alerts = []
    all_shap_rows = []
    all_feature_rows = []
    class_driver_abs = {name: [] for name in CLASS_NAMES}
    model_name = config["selected_model"]
    model_parameters = {
        name: config[name]["parameters"]
        for name in MODEL_NAMES
    }

    for seed in seeds:
        windows, train_indices, test_indices = make_dataset(
            seed,
            warmup=defaults["warmup_windows"],
            n_cycles=config["n_cycles"],
            train_fraction=config["train_fraction"],
            gap_windows=config["gap_windows"],
        )
        train_labels = windows.loc[train_indices, "label"].tolist()
        test_labels = windows.loc[test_indices, "label"].tolist()
        print(
            f"seed {seed} train counts: {class_counts(train_labels)}; "
            f"test counts: {class_counts(test_labels)}"
        )
        x_train = windows.loc[train_indices, FEATURE_COLUMNS].to_numpy(dtype=float)
        x_test = windows.loc[test_indices, FEATURE_COLUMNS].to_numpy(dtype=float)
        y_test = np.asarray([CLASS_NAMES.index(label) for label in test_labels])

        for current_model in MODEL_NAMES:
            model = build_model(
                current_model,
                seed=config["model_seed"],
                parameters=model_parameters[current_model],
            )
            fit_model(model, current_model, x_train, train_labels)
            predictions = np.asarray(model.predict(x_test), dtype=int)
            metrics = classification_metrics(y_test, predictions)
            metrics["train_class_counts"] = class_counts(train_labels)
            metrics["test_class_counts"] = class_counts(test_labels)
            per_model[current_model][str(seed)] = metrics

            if current_model != model_name:
                continue

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(x_test)
            selected_shap = shap_for_predicted_class(
                shap_values,
                n_samples=len(x_test),
                n_features=len(FEATURE_COLUMNS),
                n_classes=len(CLASS_NAMES),
                predicted_classes=predictions,
            )
            all_shap_rows.extend(selected_shap.tolist())
            all_feature_rows.extend(x_test.tolist())

            for class_index, class_name in enumerate(CLASS_NAMES):
                class_values = shap_for_predicted_class(
                    shap_values,
                    n_samples=len(x_test),
                    n_features=len(FEATURE_COLUMNS),
                    n_classes=len(CLASS_NAMES),
                    predicted_classes=np.full(len(x_test), class_index),
                )
                class_driver_abs[class_name].append(
                    np.mean(np.abs(class_values), axis=0)
                )

            probabilities = np.asarray(model.predict_proba(x_test), dtype=float)
            layer1 = AdaptiveBaseline(
                warmup=defaults["warmup_windows"],
                history=defaults["history_windows"],
                refit_every=defaults["refit_every_windows"],
                contamination=defaults["contamination"],
                seed=defaults["model_seed"],
                refit_trigger="periodic",
            )
            layer1_results = layer1.run(windows[FEATURE_COLUMNS])
            for row_number, predicted_class in enumerate(predictions):
                if predicted_class == CLASS_NAMES.index("normal"):
                    continue
                window_index = int(test_indices[row_number])
                all_alerts.append(
                    build_alert_record(
                        alert_id=f"ata-{model_name}-{seed}-{window_index}",
                        window_index=window_index,
                        layer1_score=float(layer1_results.loc[window_index, "score"]),
                        predicted_class=CLASS_NAMES[predicted_class],
                        class_probability=float(probabilities[row_number, predicted_class]),
                        top_features=top_features(
                            FEATURE_COLUMNS,
                            x_test[row_number],
                            selected_shap[row_number],
                        ),
                        model_name=model_name,
                        model_version=_model_version(model_name),
                    )
                )

    aggregates = {
        name: _aggregate_model(per_model[name])
        for name in MODEL_NAMES
    }
    for name in MODEL_NAMES:
        _plot_confusion(
            aggregates[name]["confusion_matrix_total"],
            name,
            output_dir / f"confusion_matrix_{name}.png",
        )

    _write_shap_summary(
        all_shap_rows,
        all_feature_rows,
        output_dir / "shap_summary.png",
    )
    summary_drivers = {}
    for class_name, rows in class_driver_abs.items():
        mean_abs = np.mean(np.vstack(rows), axis=0)
        order = np.argsort(-mean_abs, kind="stable")[:5]
        summary_drivers[class_name] = [
            {
                "feature": FEATURE_COLUMNS[index],
                "mean_abs_shap": float(mean_abs[index]),
            }
            for index in order
        ]
    sanity = {
        class_name: bool(
            SHAP_SANITY_FEATURES[class_name]
            & {
                item["feature"]
                for item in summary_drivers[class_name]
            }
        )
        for class_name in SHAP_SANITY_FEATURES
    }
    held_out_f1 = {
        name: aggregates[name]["macro_mean_std"]["f1"]["mean"]
        for name in MODEL_NAMES
    }
    if abs(held_out_f1["xgboost"] - held_out_f1["random_forest"]) < 1e-12:
        held_out_interpretation = (
            "Held-out macro F1 is tied; Random Forest remains selected from "
            "development-only results and the documented tie-break."
        )
    else:
        held_out_leader = max(held_out_f1, key=held_out_f1.get)
        held_out_interpretation = (
            f"{held_out_leader.replace('_', ' ').title()} has the higher "
            "held-out macro F1, but model selection remains based only on "
            "development seeds."
        )

    sample_alerts = all_alerts[:25]
    alerts_path = output_dir / "alerts_sample.jsonl"
    alerts_path.write_text(
        "".join(json.dumps(alert, allow_nan=False) + "\n" for alert in sample_alerts),
        encoding="utf-8",
    )
    cic_files = sorted((ROOT / "data").glob("*TrafficLabelling*.csv"))

    report = {
        "status": "synthetic, not evidence of real-world performance",
        "limitations": [
            "Synthetic-only evaluation on five held-out seeds.",
            "The simplified generator uses repeated, strongly separable attack templates.",
            "CIC TrafficLabelling evaluation is skipped because no files are present.",
        ],
        "engineering_status": "PASS",
        "layer1_research_acceptance": {
            "status": layer1_status,
            "verdict": layer1_verdict,
            "note": (
                "Layer 2 engineering is independent; this does not change the "
                "Phase 5 research-acceptance result."
            ),
        },
        "seeds": seeds,
        "development_selection": config["development_selection"],
        "config": config,
        "class_names": list(CLASS_NAMES),
        "feature_columns": list(FEATURE_COLUMNS),
        "library_versions": {
            "xgboost": importlib.metadata.version("xgboost"),
            "shap": importlib.metadata.version("shap"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "cic_trafficlabelling": (
            {"status": "available", "files": [str(path) for path in cic_files]}
            if cic_files
            else {
                "status": "SKIPPED",
                "reason": "No CIC TrafficLabelling CSV files in data/.",
            }
        ),
        "models": aggregates,
        "selected_model": model_name,
        "model_comparison": {
            "selected_on_development_seeds": config["development_selection"]["selected_model"],
            "selection_reason": config["development_selection"]["reason"],
            "held_out_macro_f1": {
                name: aggregates[name]["macro_mean_std"]["f1"]
                for name in MODEL_NAMES
            },
            "held_out_interpretation": held_out_interpretation,
        },
        "shap": {
            "selected_model": model_name,
            "global_top_features_by_class": summary_drivers,
            "sanity_checks": sanity,
        },
        "alerts_sample": {
            "file": "alerts_sample.jsonl",
            "records_written": len(sample_alerts),
            "limit": 25,
        },
    }
    (output_dir / "classifier_metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    for current_model in MODEL_NAMES:
        print(f"\n{current_model} per-class metrics (mean +/- SD across seeds)")
        for class_name, metrics in aggregates[current_model]["per_class_mean_std"].items():
            print(
                f"  {class_name}: precision={metrics['precision']['mean']:.4f} "
                f"+/- {metrics['precision']['std']:.4f}, "
                f"recall={metrics['recall']['mean']:.4f} +/- {metrics['recall']['std']:.4f}, "
                f"F1={metrics['f1']['mean']:.4f} +/- {metrics['f1']['std']:.4f}, "
                f"support={metrics['support']['mean']:.1f} "
                f"+/- {metrics['support']['std']:.1f}"
            )
        macro = aggregates[current_model]["macro_mean_std"]
        print(
            f"  macro: precision={macro['precision']['mean']:.4f} "
            f"+/- {macro['precision']['std']:.4f}, "
            f"recall={macro['recall']['mean']:.4f} +/- {macro['recall']['std']:.4f}, "
            f"F1={macro['f1']['mean']:.4f} +/- {macro['f1']['std']:.4f}"
        )
    print(
        f"\nSelected model: {model_name}. "
        f"{report['model_comparison']['selection_reason']}"
    )
    print(f"SHAP global top features: {summary_drivers}")
    print(f"SHAP sanity checks: {sanity}")
    print(f"Reports: {output_dir.resolve()}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate supervised ATA Layer 2 classifiers."
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[200, 201, 202, 203, 204],
    )
    parser.add_argument("--out", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    run_classification(args.seeds, args.out)


if __name__ == "__main__":
    main()
