"""Incremental ATA pipeline using the batch feature and detector interfaces."""

from __future__ import annotations

import importlib.metadata
import math
import time
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from .alerts import build_alert_record
from .baseline import AdaptiveBaseline, StaticBaseline
from .classifier import CLASS_NAMES, build_model, fit_model, make_dataset
from .explain import shap_for_predicted_class, top_features
from .features import FEATURE_COLUMNS, WindowAggregator
from .run import DEFAULTS_PATH

FLOW_REQUIRED = (
    "ts",
    "src",
    "dst",
    "dport",
    "proto",
    "bytes",
    "packets",
    "duration",
    "syn",
)
PROTOCOLS = {"tcp", "udp", "icmp", "other"}


def validate_flow_batch(
    flows: Sequence[Mapping[str, Any]],
    *,
    previous_ts: float | None = None,
) -> list[dict[str, Any]]:
    """Validate a complete batch before the caller mutates pipeline state."""
    validated: list[dict[str, Any]] = []
    last_ts = previous_ts
    for position, source in enumerate(flows):
        if not isinstance(source, Mapping):
            raise ValueError(f"Flow {position} must be a JSON object.")
        missing = [name for name in FLOW_REQUIRED if name not in source]
        if missing:
            raise ValueError(
                f"Flow {position} is missing required fields: {', '.join(missing)}."
            )
        flow = dict(source)
        try:
            timestamp = float(flow["ts"])
            port = int(flow["dport"])
            byte_count = float(flow["bytes"])
            packet_count = float(flow["packets"])
            duration = float(flow["duration"])
            syn = int(flow["syn"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Flow {position} has invalid numeric fields.") from exc
        numeric = (timestamp, byte_count, packet_count, duration)
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError(f"Flow {position} contains a non-finite numeric value.")
        if timestamp < 0:
            raise ValueError(f"Flow {position} timestamp cannot be negative.")
        if last_ts is not None and timestamp < last_ts:
            raise ValueError(f"Flow {position} timestamp is out of order.")
        if port < 0 or port > 65535:
            raise ValueError(f"Flow {position} destination port must be in 0..65535.")
        if min(byte_count, packet_count, duration) < 0:
            raise ValueError(
                f"Flow {position} bytes, packets, and duration must be non-negative."
            )
        if syn not in (0, 1):
            raise ValueError(f"Flow {position} syn must be zero or one.")
        protocol = str(flow["proto"]).lower()
        if protocol not in PROTOCOLS:
            raise ValueError(
                f"Flow {position} protocol must be one of "
                f"{', '.join(sorted(PROTOCOLS))}."
            )
        for address in ("src", "dst"):
            if not isinstance(flow[address], str) or not flow[address].strip():
                raise ValueError(f"Flow {position} {address} must be a non-empty string.")
            flow[address] = flow[address].strip()
        if "label" in flow and flow["label"] is not None:
            if not isinstance(flow["label"], str):
                raise ValueError(f"Flow {position} label must be a string or null.")
            flow["label"] = flow["label"].lower()
        else:
            flow.pop("label", None)
        if "label_raw" in flow and flow["label_raw"] is None:
            flow.pop("label_raw")
        flow.update(
            ts=timestamp,
            dport=port,
            bytes=byte_count,
            packets=packet_count,
            duration=duration,
            syn=syn,
            proto=protocol,
        )
        validated.append(flow)
        last_ts = timestamp
    return validated


class AtaPipeline:
    """Process flows and emit completed-window scores and persisted alert records."""

    def __init__(
        self,
        *,
        window: float = 1.0,
        warmup: int = 120,
        history: int = 300,
        refit_every: int | None = 30,
        contamination: float = 0.005,
        model_seed: int = 0,
        refit_trigger: str = "periodic",
        drift: Mapping[str, Any] | None = None,
        classifier=None,
        explainer=None,
        model_name: str | None = None,
        telemetry_limit: int = 2000,
        adaptation_policy: str = "strict",
        layer1_mode: str = "static",
    ):
        defaults = self._defaults()
        drift_config = dict(drift or defaults["drift"])
        self.window = float(window)
        self.warmup = int(warmup)
        self.history = int(history)
        self.refit_every = refit_every
        self.contamination = float(contamination)
        self.model_seed = int(model_seed)
        self.layer1_mode = str(layer1_mode)
        if self.layer1_mode == "static":
            self.baseline = StaticBaseline(
                warmup=self.warmup,
                history=self.history,
                refit_every=None,
                contamination=self.contamination,
                seed=self.model_seed,
                max_state_rows=telemetry_limit,
            )
        elif self.layer1_mode == "adaptive":
            self.baseline = AdaptiveBaseline(
                warmup=self.warmup,
                history=self.history,
                refit_every=self.refit_every,
                contamination=self.contamination,
                seed=self.model_seed,
                refit_trigger="periodic",
                drift_method=drift_config["method"],
                psi_threshold=drift_config["psi_threshold"],
                min_drift_features=drift_config["min_features"],
                drift_persistence=drift_config["persistence"],
                test_windows=drift_config["test_windows"],
                check_every=drift_config["check_every"],
                drift_cooldown=drift_config["cooldown"],
                max_state_rows=telemetry_limit,
            )
        elif self.layer1_mode == "drift":
            self.baseline = AdaptiveBaseline(
                warmup=self.warmup,
                history=self.history,
                refit_every=self.refit_every,
                contamination=self.contamination,
                seed=self.model_seed,
                refit_trigger="drift",
                drift_method=drift_config["method"],
                psi_threshold=drift_config["psi_threshold"],
                min_drift_features=drift_config["min_features"],
                drift_persistence=drift_config["persistence"],
                test_windows=drift_config["test_windows"],
                check_every=drift_config["check_every"],
                drift_cooldown=drift_config["cooldown"],
                max_state_rows=telemetry_limit,
            )
        else:
            raise ValueError(
                f"layer1_mode must be 'static', 'adaptive', or 'drift'; received {layer1_mode!r}."
            )
        self.aggregator = WindowAggregator(window=self.window)
        self.classifier = classifier
        self.explainer = explainer
        self.model_name = model_name
        self.model_version = (
            importlib.metadata.version(
                "xgboost" if model_name == "xgboost" else "scikit-learn"
            )
            if classifier is not None and model_name is not None
            else None
        )
        self.flows_processed = 0
        self.windows_processed = 0
        self.flags = 0
        self.alerts_generated = 0
        self._last_flow_ts: float | None = None
        self._flushed = False
        self._started_at = time.perf_counter()
        self._telemetry: deque[dict[str, Any]] = deque(maxlen=telemetry_limit)
        self._observed_drift_events = 0

    @staticmethod
    def _defaults() -> dict[str, Any]:
        import json

        return json.loads(DEFAULTS_PATH.read_text(encoding="utf-8"))

    @property
    def ready(self) -> bool:
        return self.windows_processed >= self.warmup

    @property
    def last_flow_ts(self) -> float | None:
        return self._last_flow_ts

    @property
    def recent_windows(self) -> list[dict[str, Any]]:
        return list(self._telemetry)

    def process_flows(
        self, flows: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        """Validate the entire batch, then process each flow incrementally."""
        if self._flushed:
            raise ValueError("Pipeline is closed after flush.")
        validated = validate_flow_batch(flows, previous_ts=self._last_flow_ts)
        alerts: list[dict[str, Any]] = []
        for flow in validated:
            self.aggregator.add(flow)
            self.flows_processed += 1
            self._last_flow_ts = float(flow["ts"])
            ready = self.aggregator.pop_ready()
            alerts.extend(self._process_windows(ready))
        return alerts

    def flush(self) -> list[dict[str, Any]]:
        """Finalize the last partial window; repeated calls are safe."""
        if self._flushed:
            return []
        self._flushed = True
        return self._process_windows(self.aggregator.flush())

    def _process_windows(self, windows: pd.DataFrame) -> list[dict[str, Any]]:
        alerts: list[dict[str, Any]] = []
        for item in windows.to_dict(orient="records"):
            index = int(item.pop("window"))
            item.pop("label", None)
            feature_vector = np.asarray(
                [item[column] for column in FEATURE_COLUMNS], dtype=float
            )
            score, flagged = self.baseline.step(feature_vector, index=index)
            self.windows_processed += 1
            self.flags += int(flagged)
            drift_events = self.baseline.drift_events
            newly_observed = drift_events[self._observed_drift_events :]
            self._observed_drift_events = len(drift_events)
            is_warmup = not math.isfinite(float(score))
            telemetry = {
                "window_index": index,
                "event_timestamp": float(index * self.window),
                "flow_count": int(item["flows"]),
                "score": None if is_warmup else float(score),
                "flagged": bool(flagged),
                "predicted_class": None,
                "drift_event": bool(newly_observed),
                "features": {
                    name: float(item[name]) for name in FEATURE_COLUMNS
                },
            }
            self._telemetry.append(telemetry)
            if not is_warmup:
                drift_metadata = newly_observed[-1] if newly_observed else None
                if self.classifier is None:
                    if flagged:
                        prediction = self._classify(feature_vector)
                        alert = build_alert_record(
                            alert_id=f"ata-{uuid.uuid4().hex}",
                            window_index=index,
                            layer1_score=float(score),
                            predicted_class="unknown",
                            class_probability=None,
                            top_features=[],
                            model_name="unavailable",
                            model_version=None,
                            layer1_flagged=True,
                        )
                        alert.update(
                            event_timestamp=float(index * self.window),
                            drift_metadata=drift_metadata,
                            layer1_flagged=True,
                        )
                        self.alerts_generated += 1
                        telemetry["predicted_class"] = "unknown"
                        alerts.append(alert)
                else:
                    prediction = self._classify(feature_vector)
                    predicted_class = prediction["predicted_class"]
                    telemetry["predicted_class"] = predicted_class

                    if predicted_class != "normal":
                        alert = build_alert_record(
                            alert_id=f"ata-{uuid.uuid4().hex}",
                            window_index=index,
                            layer1_score=float(score),
                            predicted_class=predicted_class,
                            class_probability=prediction["class_probability"],
                            top_features=prediction["top_features"],
                            model_name=prediction["model_name"],
                            model_version=prediction["model_version"],
                            layer1_flagged=bool(flagged),
                        )
                        alert.update(
                            event_timestamp=float(index * self.window),
                            drift_metadata=drift_metadata,
                            layer1_flagged=bool(flagged),
                        )
                        self.alerts_generated += 1
                        alerts.append(alert)
                    elif flagged:
                        alert = build_alert_record(
                            alert_id=f"ata-{uuid.uuid4().hex}",
                            window_index=index,
                            layer1_score=float(score),
                            predicted_class="unknown_anomaly",
                            class_probability=None,
                            top_features=[],
                            model_name=prediction["model_name"],
                            model_version=prediction["model_version"],
                            layer1_flagged=True,
                        )
                        alert.update(
                            event_timestamp=float(index * self.window),
                            drift_metadata=drift_metadata,
                            layer1_flagged=True,
                        )
                        self.alerts_generated += 1
                        telemetry["predicted_class"] = "unknown_anomaly"
                        alerts.append(alert)
        return alerts

    def _classify(self, feature_vector: np.ndarray) -> dict[str, Any]:
        if self.classifier is None:
            return {
                "predicted_class": "unknown",
                "class_probability": None,
                "top_features": [],
                "model_name": "unavailable",
                "model_version": None,
            }
        predicted = int(self.classifier.predict(feature_vector.reshape(1, -1))[0])
        probabilities = np.asarray(
            self.classifier.predict_proba(feature_vector.reshape(1, -1))[0],
            dtype=float,
        )
        top = []
        if self.explainer is not None:
            raw_values = self.explainer.shap_values(feature_vector.reshape(1, -1))
            shap_row = shap_for_predicted_class(
                raw_values,
                n_samples=1,
                n_features=len(FEATURE_COLUMNS),
                n_classes=len(CLASS_NAMES),
                predicted_classes=[predicted],
            )[0]
            top = top_features(FEATURE_COLUMNS, feature_vector, shap_row)
        return {
            "predicted_class": CLASS_NAMES[predicted],
            "class_probability": float(probabilities[predicted]),
            "top_features": top,
            "model_name": self.model_name or "unavailable",
            "model_version": self.model_version,
        }

    def score_features(self, features: Mapping[str, Any]) -> dict[str, Any]:
        """Score a feature mapping without changing detector or stream state."""
        if set(features) != set(FEATURE_COLUMNS):
            missing = sorted(set(FEATURE_COLUMNS) - set(features))
            extra = sorted(set(features) - set(FEATURE_COLUMNS))
            raise ValueError(f"Feature columns mismatch; missing={missing}, extra={extra}.")
        try:
            vector = np.asarray([features[name] for name in FEATURE_COLUMNS], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Feature values must be numeric.") from exc
        if vector.shape != (len(FEATURE_COLUMNS),) or not np.isfinite(vector).all():
            raise ValueError("Feature vector must contain finite numeric values.")
        score, flagged = self.baseline.score_only(vector)
        if self.classifier is not None:
            prediction = self._classify(vector)
        elif flagged:
            prediction = self._classify(vector)
        else:
            prediction = {
                "predicted_class": "normal",
                "class_probability": None,
                "top_features": [],
                "model_name": self.model_name or "unavailable",
                "model_version": self.model_version,
            }
        return {
            "score": float(score),
            "flagged": bool(flagged),
            "classification": prediction,
        }

    def stats(self) -> dict[str, Any]:
        elapsed = max(time.perf_counter() - self._started_at, 1e-9)
        return {
            "flows_processed": self.flows_processed,
            "windows_processed": self.windows_processed,
            "flags": self.flags,
            "alerts_generated": self.alerts_generated,
            "drift_events": len(self.baseline.drift_events),
            "refits": self.baseline.n_refits,
            "skipped_refits": self.baseline.skipped_refits,
            "warmup_complete": self.ready,
            "classifier_loaded": self.classifier is not None,
            "pipeline_ready": self.ready,
            "elapsed_seconds": elapsed,
            "flows_per_second": self.flows_processed / elapsed,
            "windows_per_second": self.windows_processed / elapsed,
            "recent_windows": self.recent_windows,
            "recent_drift_events": self.baseline.drift_events[-100:],
        }


def train_demo_classifier(seed: int = 1):
    """Train an optional in-memory demo classifier on the development split."""
    defaults = AtaPipeline._defaults()
    config = defaults["classifier"]
    windows, train_indices, _ = make_dataset(
        seed,
        warmup=defaults["warmup_windows"],
        n_cycles=config["n_cycles"],
        train_fraction=config["train_fraction"],
        gap_windows=config["gap_windows"],
    )
    model_name = config["selected_model"]
    model = build_model(
        model_name,
        seed=config["model_seed"],
        parameters=config[model_name]["parameters"],
    )
    fit_model(
        model,
        model_name,
        windows.loc[train_indices, FEATURE_COLUMNS].to_numpy(dtype=float),
        windows.loc[train_indices, "label"].tolist(),
    )
    import shap

    return model, shap.TreeExplainer(model), model_name
