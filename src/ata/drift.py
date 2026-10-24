"""Streaming feature drift statistics and compatibility helpers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


def _as_matrix(values, *, name: str = "values") -> np.ndarray:
    """Convert a 1D or 2D array-like object into a 2D float matrix."""
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array.")
    if arr.size == 0:
        return arr.reshape(0, arr.shape[-1] if arr.ndim == 2 else 0)
    return arr.astype(float, copy=False)


def n_shifted_features(current, previous, reference=None) -> int:
    """Count features whose mean has shifted materially between two windows."""
    current_arr = _as_matrix(current, name="current")
    previous_arr = _as_matrix(previous, name="previous")
    if current_arr.shape[1] != previous_arr.shape[1]:
        raise ValueError(
            "current and previous windows must have the same number of features."
        )

    if reference is None:
        reference_arr = np.vstack([previous_arr, current_arr])
    else:
        reference_arr = _as_matrix(reference, name="reference")
        if reference_arr.shape[1] != current_arr.shape[1]:
            raise ValueError("reference and window feature counts must match.")

    finite_reference = np.where(np.isfinite(reference_arr), reference_arr, np.nan)
    ref_std = np.nanstd(finite_reference, axis=0)
    ref_std = np.where(np.isfinite(ref_std) & (ref_std > 0), ref_std, 1.0)
    with np.errstate(invalid="ignore"):
        delta = np.abs(
            np.nanmean(current_arr, axis=0) - np.nanmean(previous_arr, axis=0)
        )
    shifted = np.isfinite(delta) & (delta > np.maximum(0.25 * ref_std, 1e-6))
    return int(np.count_nonzero(shifted))


def population_stability_index(
    reference,
    current,
    *,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float | None:
    """Return PSI using reference quantiles, or None for a constant reference."""
    ref = np.asarray(reference, dtype=float)
    test = np.asarray(current, dtype=float)
    ref = ref[np.isfinite(ref)]
    test = test[np.isfinite(test)]
    if ref.size < 2 or test.size == 0 or np.ptp(ref) == 0:
        return None
    if bins < 2:
        raise ValueError("bins must be at least 2.")
    if epsilon <= 0:
        raise ValueError("epsilon must be greater than zero.")

    quantiles = np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1])
    interior = np.unique(quantiles)
    edges = np.concatenate(([-np.inf], interior, [np.inf]))
    ref_counts = np.histogram(ref, bins=edges)[0].astype(float)
    test_counts = np.histogram(test, bins=edges)[0].astype(float)
    ref_pct = np.maximum(ref_counts / ref.size, epsilon)
    test_pct = np.maximum(test_counts / test.size, epsilon)
    return float(np.sum((test_pct - ref_pct) * np.log(test_pct / ref_pct)))


def _ks_statistic(reference: np.ndarray, current: np.ndarray) -> float | None:
    ref = np.sort(reference[np.isfinite(reference)])
    test = np.sort(current[np.isfinite(current)])
    if ref.size < 2 or test.size < 1 or np.ptp(ref) == 0:
        return None
    points = np.sort(np.concatenate((ref, test)))
    ref_cdf = np.searchsorted(ref, points, side="right") / ref.size
    test_cdf = np.searchsorted(test, points, side="right") / test.size
    return float(np.max(np.abs(ref_cdf - test_cdf)))


@dataclass(frozen=True)
class DriftResult:
    """Statistics from one drift check."""

    statistics: dict[str, float | None]
    drifted: bool
    n_drifted: int
    position: int | str | None = None


class DriftMonitor:
    """Compare recent admitted feature rows with a model-fit reference."""

    def __init__(
        self,
        reference=None,
        min_features: int = 3,
        window_size: int = 12,
        *,
        test_windows: int = 60,
        method: str = "psi",
        psi_threshold: float = 0.25,
        persistence: int = 2,
        check_every: int = 10,
        cooldown: int = 30,
        feature_names=None,
    ):
        if min_features < 1:
            raise ValueError("min_features must be positive.")
        if test_windows < 2:
            raise ValueError("test_windows must be at least 2.")
        if method not in {"psi", "ks"}:
            raise ValueError("method must be 'psi' or 'ks'.")
        if psi_threshold <= 0:
            raise ValueError("psi_threshold must be positive.")
        if persistence < 1 or check_every < 1 or cooldown < 0:
            raise ValueError("persistence and check_every must be positive; cooldown cannot be negative.")
        if feature_names is not None:
            feature_names = tuple(feature_names)
        self.reference: np.ndarray | None = None
        self.min_features = int(min_features)
        self.window_size = int(window_size)
        self.test_windows = int(test_windows)
        self.method = method
        self.psi_threshold = float(psi_threshold)
        self.persistence = int(persistence)
        self.check_every = int(check_every)
        self.cooldown = int(cooldown)
        self.feature_names = feature_names
        self._recent: deque[np.ndarray] = deque(maxlen=self.test_windows)
        self._updates_since_check = 0
        self._cooldown_remaining = 0
        self._consecutive = 0
        self._position: int | str | None = None
        self.events: list[dict] = []
        self.last_result = DriftResult({}, False, 0)
        self.last_shift_count = 0
        self.drifted = False
        if reference is not None:
            self.reset_reference(reference)

    def _names(self, n_features: int) -> tuple[str, ...]:
        if self.feature_names is not None:
            if len(self.feature_names) != n_features:
                raise ValueError("feature_names length must match the feature count.")
            return self.feature_names
        return tuple(f"feature_{index}" for index in range(n_features))

    def reset_reference(self, reference=None) -> None:
        """Replace the model-fit reference and restart checks and cooldown."""
        if reference is None:
            self.reference = None
        else:
            matrix = _as_matrix(reference, name="reference")
            self.reference = matrix.copy()
            self._names(matrix.shape[1])
        self._recent.clear()
        self._updates_since_check = 0
        self._consecutive = 0
        self._cooldown_remaining = self.cooldown if self.reference is not None else 0
        self.last_shift_count = 0
        self.drifted = False
        self.last_result = DriftResult({}, False, 0, self._position)

    def _statistics(self) -> dict[str, float | None]:
        if self.reference is None or len(self._recent) < self.test_windows:
            return {}
        recent = np.vstack(self._recent)
        names = self._names(self.reference.shape[1])
        result = {}
        for column, name in enumerate(names):
            ref = self.reference[:, column]
            test = recent[:, column]
            if self.method == "psi":
                statistic = population_stability_index(ref, test)
            else:
                statistic = _ks_statistic(ref, test)
            result[name] = statistic
        return result

    def check(self) -> DriftResult:
        """Evaluate the current test window and update persistence state."""
        statistics = self._statistics()
        threshold = self.psi_threshold
        shifted = [
            name
            for name, value in statistics.items()
            if value is not None and np.isfinite(value) and value > threshold
        ]
        if statistics and self._cooldown_remaining == 0:
            self._consecutive = self._consecutive + 1 if len(shifted) >= self.min_features else 0
        else:
            self._consecutive = 0
        fired = self._consecutive >= self.persistence
        if fired:
            values = {name: statistics[name] for name in shifted}
            event = {
                "position": self._position,
                "drifted_features": shifted,
                "statistics": values,
                "method": self.method,
            }
            if self.method == "psi":
                event["psi_values"] = values
            self.events.append(event)
            self._consecutive = 0
            self._cooldown_remaining = self.cooldown
        self.last_shift_count = len(shifted)
        self.drifted = fired
        self.last_result = DriftResult(
            statistics, fired, len(shifted), self._position
        )
        return self.last_result

    def update(self, row, position=None) -> bool:
        """Observe one feature row; return True only when a drift event fires."""
        row_arr = _as_matrix(row, name="row")
        if row_arr.shape[0] != 1:
            raise ValueError("update expects one feature row.")
        if self.reference is None:
            self.reference = row_arr.copy()
            self._names(row_arr.shape[1])
            self._cooldown_remaining = self.cooldown
        elif row_arr.shape[1] != self.reference.shape[1]:
            raise ValueError("row and reference feature counts must match.")
        self._position = position
        self.drifted = False
        self.last_shift_count = 0
        self._recent.append(row_arr[0].copy())
        self._updates_since_check += 1
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            self._updates_since_check = 0
            return False
        if (
            len(self._recent) >= self.test_windows
            and self._updates_since_check >= self.check_every
        ):
            self._updates_since_check = 0
            return self.check().drifted
        return False


class LegacyShiftMonitor:
    """The former half-window mean-shift detector for compatibility/testing."""

    def __init__(self, reference=None, min_features: int = 3, window_size: int = 12):
        self.reference = None if reference is None else _as_matrix(reference)
        self.min_features = int(min_features)
        self.window_size = int(window_size)
        self._recent: list[np.ndarray] = []
        self.last_shift_count = 0
        self.drifted = False

    def reset_reference(self, reference=None) -> None:
        self.reference = None if reference is None else _as_matrix(reference)
        self._recent.clear()
        self.last_shift_count = 0
        self.drifted = False

    def update(self, row, position=None) -> bool:
        row_arr = _as_matrix(row)
        if row_arr.size == 0:
            return False
        if self.reference is None:
            self.reference = row_arr.copy()
            self._recent = [row_arr.copy()]
            self.last_shift_count = 0
            self.drifted = False
            return False
        self._recent.append(row_arr.copy())
        recent = np.asarray(self._recent[-self.window_size :], dtype=float)
        if recent.shape[0] < 2:
            return False
        midpoint = max(1, recent.shape[0] // 2)
        self.last_shift_count = n_shifted_features(
            recent[midpoint:], recent[:midpoint], self.reference
        )
        self.drifted = self.last_shift_count >= self.min_features
        return self.drifted


__all__ = [
    "DriftMonitor",
    "DriftResult",
    "LegacyShiftMonitor",
    "n_shifted_features",
    "population_stability_index",
]
