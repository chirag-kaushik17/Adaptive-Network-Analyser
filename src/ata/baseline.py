"""Adaptive and static unsupervised Isolation Forest baselines."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


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
    ):
        if warmup < 2:
            raise ValueError("warmup must be at least 2 windows.")
        if history < 2:
            raise ValueError("history must be at least 2 windows.")
        if refit_every is not None and refit_every < 1:
            raise ValueError("refit_every must be at least 1 or None.")
        if not 0 < contamination < 0.5:
            raise ValueError("contamination must be strictly between 0 and 0.5.")
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
        self.scaler = StandardScaler()
        self.forest = IsolationForest(
            contamination=self.contamination, random_state=self.seed
        )
        self.buffer: list[np.ndarray] = []
        self.buffer_indices: list[int | str] = []
        self._post_warmup_seen = 0
        self._refit_calls = 0
        self.skipped_refits = 0
        self._fitted = False
        self._step_count = 0
        self._n_features: int | None = None

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
        self.scaler.fit(matrix)
        self.forest = IsolationForest(
            contamination=self.contamination, random_state=self.seed
        )
        self.forest.fit(self.scaler.transform(matrix))
        self._n_features = matrix.shape[1]
        self._fitted = True
        if not initial:
            self._refit_calls += 1

    def _score_flag(self, row: np.ndarray) -> tuple[float, bool]:
        if not self._fitted:
            raise ValueError("The baseline must be fitted before scoring rows.")
        scaled = self.scaler.transform(row.reshape(1, -1))
        score = float(-self.forest.decision_function(scaled)[0])
        return score, score > 0

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
        self._post_warmup_seen = 0
        self._refit_calls = 0
        self.skipped_refits = 0
        self._fitted = False
        self._step_count = 0
        self._n_features = None

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
            return np.nan, False

        if not self._fitted:
            raise ValueError("Adaptive baseline has not been warmed up.")

        score, flag = self._score_flag(row)
        self._post_warmup_seen += 1
        if not flag:
            self.buffer.append(row.copy())
            self.buffer_indices.append(index)

        if (
            self.refit_every is not None
            and self._post_warmup_seen % self.refit_every == 0
        ):
            fit_rows = self.buffer[-self.history :]
            if len(fit_rows) < self.min_fit_rows:
                self.skipped_refits += 1
            else:
                self._fit_model(fit_rows)
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
