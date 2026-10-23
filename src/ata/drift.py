"""Drift detection helpers for the adaptive baseline.

The project's baseline logic expects a small, deterministic drift monitor with a
feature-shift count and a boolean drift state. This module keeps the interface
lightweight while remaining usable in the streaming evaluation flow.
"""

from __future__ import annotations

import numpy as np


def _as_matrix(values, *, name: str = "values") -> np.ndarray:
    """Convert a 1D or 2D array-like object into a 2D float matrix."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.size == 0:
        return arr.reshape(0, 0)
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
            raise ValueError(
                "reference and window feature counts must match."
            )

    ref_mean = reference_arr.mean(axis=0)
    ref_std = reference_arr.std(axis=0)
    ref_std = np.where(np.isfinite(ref_std) & (ref_std > 0), ref_std, 1.0)

    delta = np.abs(current_arr.mean(axis=0) - previous_arr.mean(axis=0))
    tolerance = np.maximum(0.25 * ref_std, 1e-6)
    shifted = delta > tolerance
    return int(np.count_nonzero(shifted))


class DriftMonitor:
    """Track streaming feature drift relative to a stable reference window."""

    def __init__(self, reference=None, min_features: int = 3, window_size: int = 12):
        self.reference = None if reference is None else _as_matrix(reference)
        self.min_features = int(min_features)
        self.window_size = int(window_size)
        self._recent: list[np.ndarray] = []
        self.last_shift_count = 0
        self.drifted = False

    def reset_reference(self, reference=None) -> None:
        """Replace the baseline reference and clear any recent-window state."""
        self.reference = None if reference is None else _as_matrix(reference)
        self._recent.clear()
        self.last_shift_count = 0
        self.drifted = False

    def update(self, row, position=None) -> bool:
        """Update the drift monitor with a new row and return whether drift is detected."""
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
        if len(self._recent) < 2:
            return False

        start = max(0, len(self._recent) - self.window_size)
        recent = np.asarray(self._recent[start:], dtype=float)
        if recent.ndim == 1:
            recent = recent.reshape(1, -1)
        if recent.shape[0] < 2:
            return False

        first_half = recent[: max(1, recent.shape[0] // 2)]
        second_half = recent[max(1, recent.shape[0] // 2) :]
        if second_half.size == 0:
            second_half = recent[-1:].copy()
        if first_half.size == 0:
            first_half = recent[:1].copy()

        self.last_shift_count = n_shifted_features(
            second_half, first_half, self.reference
        )
        self.drifted = self.last_shift_count >= self.min_features
        return self.drifted


__all__ = ["DriftMonitor", "n_shifted_features"]
