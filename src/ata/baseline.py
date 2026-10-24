"""Adaptive and static unsupervised Isolation Forest baselines."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .drift import DriftMonitor, n_shifted_features
from .features import FEATURE_COLUMNS


class BaseBaseline:
    """Shared warm-up and scoring logic for baseline variants."""

    def __init__(
        self,
        warmup: int,
        history: int,
        refit_every: int | None,
        contamination: float,
        seed: int = 0,
        min_fit_rows: int | None = None,
        policy: str = "strict",
        severity_beta: float = 1.5,
        promote_after: int = 46,
        min_flag_frac: float = 0.30,
        promotion_cooldown: int = 30,
        drift_monitor: DriftMonitor | None = None,
        refit_trigger: str = "periodic",
        drift_method: str = "psi",
        psi_threshold: float = 0.25,
        min_drift_features: int = 3,
        drift_persistence: int = 2,
        test_windows: int = 60,
        check_every: int = 10,
        drift_cooldown: int = 30,
        max_state_rows: int | None = None,
    ):
        if warmup < 2:
            raise ValueError("warmup must be at least 2 windows.")
        if history < 2:
            raise ValueError("history must be at least 2 windows.")
        if refit_every is not None and refit_every < 1:
            raise ValueError("refit_every must be at least 1 or None.")
        if not 0 < contamination < 0.5:
            raise ValueError("contamination must be strictly between 0 and 0.5.")
        if policy not in {"strict", "quarantine"}:
            raise ValueError("policy must be 'strict' or 'quarantine'.")
        if refit_trigger not in {"periodic", "drift"}:
            raise ValueError("refit_trigger must be 'periodic' or 'drift'.")
        if severity_beta <= 0:
            raise ValueError("severity_beta must be greater than zero.")
        if promote_after <= 0:
            raise ValueError("promote_after must be positive.")
        if not 0 <= min_flag_frac <= 1:
            raise ValueError("min_flag_frac must lie in [0, 1].")
        if promotion_cooldown < 0:
            raise ValueError("promotion_cooldown cannot be negative.")
        if max_state_rows is not None and max_state_rows < 1:
            raise ValueError("max_state_rows must be positive or None.")
        if max_state_rows is not None and max_state_rows < warmup:
            raise ValueError("max_state_rows cannot be smaller than warmup.")
        if min_fit_rows is None:
            min_fit_rows = min(warmup, history)
        if min_fit_rows < 2:
            raise ValueError("min_fit_rows must be at least 2.")
        if min_fit_rows > history:
            raise ValueError("min_fit_rows cannot be greater than history.")

        self.warmup = int(warmup)
        self.history = int(history)
        self.refit_every = refit_every
        self.contamination = float(contamination)
        self.seed = int(seed)
        self.min_fit_rows = int(min_fit_rows)
        self.policy = policy
        self.severity_beta = float(severity_beta)
        self.promote_after = int(promote_after)
        self.min_flag_frac = float(min_flag_frac)
        self.promotion_cooldown = int(promotion_cooldown)
        self.max_state_rows = (
            None if max_state_rows is None else int(max_state_rows)
        )
        self.refit_trigger = refit_trigger
        self._default_drift_monitor = drift_monitor is None
        self.drift_monitor = drift_monitor if drift_monitor is not None else DriftMonitor(
            min_features=min_drift_features,
            test_windows=test_windows,
            method=drift_method,
            psi_threshold=psi_threshold,
            persistence=drift_persistence,
            check_every=check_every,
            cooldown=drift_cooldown,
        )
        self.scaler = StandardScaler()
        self.forest = IsolationForest(
            contamination=self.contamination, random_state=self.seed
        )
        self.buffer: list[np.ndarray] = []
        self.buffer_indices: list[int | str] = []
        self.quarantine: list[dict] = []
        self.n_quarantined = 0
        self.n_promotions = 0
        self.n_dropped_severe = 0
        self.degenerate_cap = 0
        self.last_cap: float | None = None
        self.last_promotion_position: int | None = None
        self.drift_events: list[dict] = []
        self.drift_refit_events: list[dict] = []
        self.promotion_events: list[dict] = []
        self._post_warmup_seen = 0
        self._refit_calls = 0
        self.skipped_refits = 0
        self._fitted = False
        self._step_count = 0
        self._n_features: int | None = None
        self._window_log: list[dict] = []
        self._recent_stream_rows: list[np.ndarray] = []

    @property
    def n_refits(self) -> int:
        """Number of adaptive refits performed after warm-up."""
        return self._refit_calls

    def _prepare_vector(self, x) -> np.ndarray:
        array = np.asarray(x, dtype=float)
        if array.ndim == 0:
            array = array.reshape(1)
        if array.ndim == 2 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 1:
            raise ValueError("Each baseline input must be a one-dimensional feature vector.")
        if not np.all(np.isfinite(array)):
            raise ValueError("Input contains non-finite values; all feature values must be finite.")
        if self._n_features is not None and len(array) != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} features but received {len(array)}."
            )
        return array.astype(float, copy=False)

    def _fit_model(self, rows: Sequence[np.ndarray], *, initial: bool = False) -> None:
        required_rows = self.warmup if initial else self.min_fit_rows
        if len(rows) < required_rows:
            if initial:
                raise ValueError(
                    f"Insufficient warm-up data: need at least {required_rows} rows."
                )
            self.skipped_refits += 1
            return
        matrix = np.vstack(rows)
        if (
            self._default_drift_monitor
            and matrix.shape[1] == len(FEATURE_COLUMNS)
        ):
            self.drift_monitor.feature_names = tuple(FEATURE_COLUMNS)
        self.scaler.fit(matrix)
        self.forest = IsolationForest(
            contamination=self.contamination, random_state=self.seed
        )
        self.forest.fit(self.scaler.transform(matrix))
        self._n_features = matrix.shape[1]
        self._fitted = True
        if not initial:
            self._refit_calls += 1
        self.drift_monitor.reset_reference(matrix)

    def _score_flag(self, row: np.ndarray) -> tuple[float, bool]:
        if not self._fitted:
            raise ValueError("The baseline must be fitted before scoring rows.")
        scaled = self.scaler.transform(row.reshape(1, -1))
        score = float(-self.forest.decision_function(scaled)[0])
        return score, score > 0

    def _bound_state(self) -> None:
        if self.max_state_rows is None:
            return
        del self.buffer[:-self.max_state_rows]
        del self.buffer_indices[:-self.max_state_rows]
        del self._window_log[:-self.max_state_rows]
        del self.quarantine[:-self.max_state_rows]

    def _cap_for_span(self, scores: Sequence[float]) -> float:
        if not scores:
            return 0.0
        median_score = float(np.median(np.asarray(scores, dtype=float)))
        if not np.isfinite(median_score) or median_score >= 0 or abs(median_score) < 1e-3:
            self.degenerate_cap += 1
            return 0.0
        cap = float(median_score * (1.0 - self.severity_beta))
        return cap

    def _refit_if_needed(self, *, force: bool = False) -> None:
        if self.refit_trigger == "drift":
            return
        if self.refit_every is None and not force:
            return
        if not force and self._post_warmup_seen % self.refit_every != 0:
            return
        fit_rows = self.buffer[-self.history :]
        if len(fit_rows) < self.min_fit_rows:
            self.skipped_refits += 1
            return
        self._fit_model(fit_rows)

    def _observe_drift(self, row: np.ndarray, position: int) -> bool:
        drifted = self.drift_monitor.update(row, position=position)
        if drifted:
            result = self.drift_monitor.last_result
            self.drift_events.append(
                {
                    "position": position,
                    "shift_count": result.n_drifted,
                    "n_drifted": result.n_drifted,
                    "drifted_features": [
                        name
                        for name, value in result.statistics.items()
                        if value is not None
                        and value > self.drift_monitor.psi_threshold
                    ],
                    "statistics": {
                        name: value
                        for name, value in result.statistics.items()
                        if value is not None
                    },
                }
            )
        return drifted

    def _refit_on_drift(self, position: int) -> None:
        fit_rows = self._recent_stream_rows[-self.history :]
        if len(fit_rows) < self.min_fit_rows:
            self.skipped_refits += 1
            return
        features = list(self.drift_events[-1]["drifted_features"])
        self._fit_model(fit_rows)
        if self.n_refits > 0:
            self.drift_refit_events.append(
                {
                    "position": position,
                    "trigger": "drift",
                    "rows_used": len(fit_rows),
                    "drifted_features": features,
                }
            )

    def _maybe_promote(self, current_index: int | str | None = None) -> None:
        if self.policy != "quarantine":
            return
        if len(self._window_log) < self.promote_after:
            return
        if self.last_promotion_position is not None and (
            self._post_warmup_seen - self.last_promotion_position < self.promotion_cooldown
        ):
            return
        span = self._window_log[-self.promote_after :]
        if len(span) != self.promote_after:
            return
        scores = [float(item["score"]) for item in span]
        flagged = [bool(item["flag"]) for item in span]
        if not any(flagged):
            return
        flagged_frac = sum(flagged) / len(span)
        if flagged_frac < self.min_flag_frac:
            return
        cap = self._cap_for_span(scores)
        if cap == 0.0 and np.median(scores) >= 0:
            return
        flagged_window_scores = [float(item["score"]) for item in span if item["flag"]]
        if not flagged_window_scores:
            return
        mild_ratio = sum(1 for score in flagged_window_scores if 0.0 < score <= cap) / len(flagged_window_scores)
        if mild_ratio < 0.9:
            return
        median_score = float(np.median(scores))
        if median_score > cap:
            return

        if self.drift_monitor is not None:
            first_half = [item["row"] for item in span[: len(span) // 2]]
            second_half = [item["row"] for item in span[len(span) // 2 :]]
            if first_half and second_half:
                shift_count = n_shifted_features(
                    np.vstack(first_half),
                    np.vstack(second_half),
                    self.drift_monitor.reference if self.drift_monitor.reference is not None else np.vstack(first_half + second_half),
                )
                if shift_count >= self.drift_monitor.min_features:
                    return

        quarantined = [item for item in self.quarantine if item["stream_position"] >= self._post_warmup_seen - self.promote_after + 1]
        if not quarantined:
            return
        promoted = [item for item in quarantined if item["flag"] and 0.0 < item["score"] <= cap]
        dropped = [item for item in quarantined if item["flag"] and item["score"] > cap]
        if not promoted:
            return
        self.buffer.extend(np.asarray([item["row"] for item in promoted], dtype=float))
        self.buffer_indices.extend(item["original_index"] for item in promoted)
        self._bound_state()
        self.last_cap = cap
        self.last_promotion_position = self._post_warmup_seen
        self.n_promotions += 1
        self.n_dropped_severe += len(dropped)
        self.quarantine.clear()
        self.n_quarantined = 0
        self.drift_monitor.reset_reference(np.vstack(self.buffer[-self.history :]))
        self.promotion_events.append(
            {
                "position": self._post_warmup_seen,
                "cap": cap,
                "span_length": self.promote_after,
                "flagged_fraction": flagged_frac,
                "mild_ratio": mild_ratio,
                "promoted_count": len(promoted),
                "dropped_severe": len(dropped),
                "new_regime_fraction": len(promoted) / max(1, len(self.buffer[-self.history :])),
            }
        )
        self._refit_if_needed(force=True)

    def score_only(self, x) -> tuple[float, bool]:
        """Score a feature vector without changing any baseline state."""
        row = self._prepare_vector(x)
        return self._score_flag(row)

    def _reset(self) -> None:
        self.scaler = StandardScaler()
        self.forest = IsolationForest(
            contamination=self.contamination, random_state=self.seed
        )
        self.buffer = []
        self.buffer_indices = []
        self.quarantine = []
        self.n_quarantined = 0
        self.n_promotions = 0
        self.n_dropped_severe = 0
        self.degenerate_cap = 0
        self.last_cap = None
        self.last_promotion_position = None
        self.drift_events = []
        self.drift_refit_events = []
        self.promotion_events = []
        self._post_warmup_seen = 0
        self._refit_calls = 0
        self.skipped_refits = 0
        self._fitted = False
        self._step_count = 0
        self._n_features = None
        self._window_log = []
        self._recent_stream_rows = []
        if self.drift_monitor is not None:
            self.drift_monitor.reset_reference(None)

    def run(self, X) -> pd.DataFrame:
        """Reset the detector and process rows through the streaming step API."""
        self._reset()
        if isinstance(X, pd.DataFrame):
            rows = X.to_numpy(dtype=float)
            index = X.index
        else:
            rows = np.asarray(X, dtype=float)
            if rows.ndim == 1:
                rows = rows.reshape(-1, 1)
            index = pd.RangeIndex(len(rows))
        if len(rows) < self.warmup:
            raise ValueError(
                f"Insufficient warm-up data: received {len(rows)} rows, "
                f"need {self.warmup}."
            )
        output_rows = [
            self.step(row, index=current_index)
            for current_index, row in zip(index, rows)
        ]
        return pd.DataFrame(output_rows, index=index, columns=["score", "flag"])


class AdaptiveBaseline(BaseBaseline):
    """Refit on eligible rows only; false negatives can still poison history."""

    def step(self, x, index=None) -> tuple[float, bool]:
        row = self._prepare_vector(x)
        if index is None:
            index = self._step_count
        self._step_count += 1

        if self._step_count <= self.warmup:
            self.buffer.append(row.copy())
            self.buffer_indices.append(index)
            if self._step_count == self.warmup:
                self._fit_model(self.buffer, initial=True)
                self._recent_stream_rows = [
                    row.copy() for row in self.buffer[-self.history :]
                ]
            return np.nan, False

        if not self._fitted:
            raise ValueError("Adaptive baseline has not been warmed up.")

        score, flag = self._score_flag(row)
        self._post_warmup_seen += 1
        self._window_log.append({
            "row": row.copy(),
            "index": index,
            "score": float(score),
            "flag": bool(flag),
            "stream_position": self._post_warmup_seen,
        })
        self._recent_stream_rows.append(row.copy())
        if len(self._recent_stream_rows) > self.history:
            self._recent_stream_rows.pop(0)

        drifted = self._observe_drift(row, self._step_count - 1)

        if self.policy == "quarantine":
            if flag:
                entry = {
                    "stream_position": self._post_warmup_seen,
                    "original_index": index,
                    "row": row.copy(),
                    "score": float(score),
                    "flag": True,
                }
                self.quarantine.append(entry)
                self.n_quarantined += 1
            else:
                self.buffer.append(row.copy())
                self.buffer_indices.append(index)
            self._bound_state()
            self._maybe_promote(current_index=index)
            if drifted and self.refit_trigger == "drift":
                self._refit_on_drift(self._step_count - 1)
            else:
                self._refit_if_needed()
            return float(score), bool(flag)

        if not flag:
            self.buffer.append(row.copy())
            self.buffer_indices.append(index)
        self._bound_state()

        if drifted and self.refit_trigger == "drift":
            self._refit_on_drift(self._step_count - 1)
        else:
            self._refit_if_needed()
        return float(score), bool(flag)


class StaticBaseline(BaseBaseline):
    """Keep only warm-up rows and never refit the initial model."""

    def step(self, x, index=None) -> tuple[float, bool]:
        row = self._prepare_vector(x)
        if index is None:
            index = self._step_count
        self._step_count += 1

        if self._step_count <= self.warmup:
            self.buffer.append(row.copy())
            self.buffer_indices.append(index)
            if self._step_count == self.warmup:
                self._fit_model(self.buffer, initial=True)
            return np.nan, False

        if not self._fitted:
            raise ValueError("Static baseline has not been warmed up.")
        return self._score_flag(row)
