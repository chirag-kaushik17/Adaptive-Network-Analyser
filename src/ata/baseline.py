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
        if refit_trigger not in {"periodic", "drift"}:
            raise ValueError("refit_trigger must be 'periodic' or 'drift'.")
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
        self.drift_events: list[dict] = []
        self.drift_refit_events: list[dict] = []
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

    def _fit_model(self, rows: Sequence[np.ndarray], *, initial: bool = False, drift_triggered: bool = False) -> None:
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
        if initial or drift_triggered:
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
        self._fit_model(fit_rows, drift_triggered=True)
        if self.n_refits > 0:
            self.drift_refit_events.append(
                {
                    "position": position,
                    "trigger": "drift",
                    "rows_used": len(fit_rows),
                    "drifted_features": features,
                }
            )

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
        self.drift_events = []
        self.drift_refit_events = []
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
