from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class BaseBaseline:
    """Shared baseline logic for adaptive and static isolation-forest baselines."""

    def __init__(self, warmup: int, history: int, refit_every: int, contamination: float, seed: int):
        if warmup < 2:
            raise ValueError("warmup must be at least 2 windows.")
        if history < 1:
            raise ValueError("history must be at least 1 window.")
        if refit_every < 1:
            raise ValueError("refit_every must be at least 1.")
        if not 0 < contamination < 0.5:
            raise ValueError("contamination must be strictly between 0 and 0.5.")
        self.warmup = int(warmup)
        self.history = int(history)
        self.refit_every = int(refit_every)
        self.contamination = float(contamination)
        self.seed = int(seed)

        self.scaler = StandardScaler()
        self.forest = IsolationForest(contamination=self.contamination, random_state=self.seed)
        self.buffer: list[np.ndarray] = []
        self.buffer_indices: list[int | str] = []
        self._post_warmup_seen = 0
        self._refit_calls = 0
        self._fitted = False
        self._step_count = 0

    def _prepare_vector(self, x) -> np.ndarray:
        array = np.asarray(x, dtype=float)
        if array.ndim == 0:
            array = array.reshape(1)
        if array.ndim == 2 and array.shape[0] == 1:
            array = array[0]
        if not np.all(np.isfinite(array)):
            raise ValueError("Input contains non-finite values; all feature values must be finite.")
        return array.astype(float, copy=False)

    def _fit_model(self, rows: Sequence[np.ndarray]) -> None:
        if len(rows) < self.warmup:
            raise ValueError("Insufficient warm-up data to fit the baseline model.")
        matrix = np.vstack(rows)
        self.scaler.fit(matrix)
        scaled = self.scaler.transform(matrix)
        self.forest.fit(scaled)
        self._fitted = True

    def _refit_model(self) -> None:
        if not self.buffer:
            return
        sample = self.buffer[-self.history :]
        self._fit_model(sample)
        self._refit_calls += 1

    def _score_flag(self, row: np.ndarray) -> tuple[float, bool]:
        if not self._fitted:
            raise ValueError("The baseline must be fitted before scoring rows.")
        scaled = self.scaler.transform(row.reshape(1, -1))
        decision = self.forest.decision_function(scaled)[0]
        score = float(-decision)
        flag = bool(self.forest.predict(scaled)[0] == -1)
        return score, flag

    def _reset(self) -> None:
        self.buffer = []
        self.buffer_indices = []
        self._post_warmup_seen = 0
        self._refit_calls = 0
        self._fitted = False
        self._step_count = 0

    def run(self, X):
        self._reset()
        if isinstance(X, pd.DataFrame):
            rows = X.to_numpy(dtype=float)
            index = X.index
        else:
            rows = np.asarray(X, dtype=float)
            index = pd.RangeIndex(len(rows))
        output_rows = []
        for current_index, row in zip(index, rows):
            score, flag = self.step(row, index=current_index)
            output_rows.append({"score": score, "flag": flag})
        return pd.DataFrame(output_rows, index=index)


class AdaptiveBaseline(BaseBaseline):
    """Isolation-forest baseline that relearns from unflagged windows only."""

    def step(self, x, index=None):
        row = self._prepare_vector(x)
        if index is None:
            index = self._step_count
        self._step_count += 1

        if self._step_count <= self.warmup:
            self.buffer.append(row.copy())
            self.buffer_indices.append(index)
            if self._step_count == self.warmup:
                self._fit_model(self.buffer)
            return np.nan, False

        if not self._fitted:
            raise ValueError("Adaptive baseline has not been warmed up.")

        score, flag = self._score_flag(row)
        self._post_warmup_seen += 1

        if not flag:
            self.buffer.append(row.copy())
            self.buffer_indices.append(index)

        if self._post_warmup_seen % self.refit_every == 0:
            self._refit_model()

        return float(score), bool(flag)


class StaticBaseline(BaseBaseline):
    """A fixed isolation-forest baseline that never refits after warm-up."""

    def step(self, x, index=None):
        row = self._prepare_vector(x)
        if index is None:
            index = self._step_count
        self._step_count += 1

        if self._step_count <= self.warmup:
            self.buffer.append(row.copy())
            self.buffer_indices.append(index)
            if self._step_count == self.warmup:
                self._fit_model(self.buffer)
            return np.nan, False

        if not self._fitted:
            raise ValueError("Static baseline has not been warmed up.")

        score, flag = self._score_flag(row)
        if not flag:
            self.buffer.append(row.copy())
            self.buffer_indices.append(index)
        return float(score), bool(flag)
