from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "flows",
    "bytes_total",
    "bytes_mean",
    "pkts_mean",
    "dur_mean",
    "syn_ratio",
    "uniq_src",
    "uniq_dst",
    "uniq_dport",
    "dport_entropy",
    "top_src_share",
]


def _validate_ordered(df: pd.DataFrame) -> None:
    if df.empty:
        return
    if "ts" not in df.columns:
        raise ValueError("Window feature generation requires a 'ts' column.")
    if not df["ts"].is_monotonic_increasing:
        raise ValueError("Input flows must be chronological; out-of-order timestamps are not allowed.")


def _window_label(group: pd.DataFrame, min_attack_flows: int, min_attack_frac: float) -> str:
    total = len(group)
    if total == 0:
        return "normal"
    if "label" not in group.columns or group["label"].isna().all():
        return "unknown"
    attack_mask = group["label"].astype(str).str.lower().ne("normal")
    attack_flows = int(attack_mask.sum())
    if attack_flows == 0:
        return "normal"
    if attack_flows >= min_attack_flows or (attack_flows / total) >= min_attack_frac:
        attacks = group.loc[attack_mask, "label"].astype(str).str.lower()
        counts = attacks.value_counts()
        winner = sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))[0][0]
        return str(winner).lower()
    return "normal"


def _entropy(values: pd.Series) -> float:
    counts = values.value_counts(normalize=True)
    if counts.empty:
        return 0.0
    probs = counts.to_numpy(dtype=float)
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def window_features(
    df: pd.DataFrame,
    window: float = 1.0,
    min_attack_flows: int = 3,
    min_attack_frac: float = 0.02,
) -> pd.DataFrame:
    """Aggregate a chronological flow table into window features."""
    if window <= 0:
        raise ValueError("Window size must be positive.")
    ordered = df.copy()
    _validate_ordered(ordered)
    if ordered.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS + ["label"])

    ordered["window_index"] = np.floor(ordered["ts"] / window).astype(int)
    rows = []
    for key, group in ordered.groupby("window_index", sort=True):
        total_flows = len(group)
        bytes_total = float(group["bytes"].sum())
        label = _window_label(group, min_attack_flows=min_attack_flows, min_attack_frac=min_attack_frac)
        if total_flows == 0:
            continue
        row = {
            "flows": int(total_flows),
            "bytes_total": bytes_total,
            "bytes_mean": float(group["bytes"].mean()),
            "pkts_mean": float(group["packets"].mean()),
            "dur_mean": float(group["duration"].mean()),
            "syn_ratio": float(group["syn"].mean()),
            "uniq_src": int(group["src"].nunique()),
            "uniq_dst": int(group["dst"].nunique()),
            "uniq_dport": int(group["dport"].nunique()),
            "dport_entropy": _entropy(group["dport"]),
            "top_src_share": float(group["src"].value_counts(normalize=True).max()),
            "label": str(label),
        }
        row["window"] = key
        rows.append(row)

    result = pd.DataFrame(rows, columns=["window", *FEATURE_COLUMNS, "label"]) if rows else pd.DataFrame(columns=["window", *FEATURE_COLUMNS, "label"])
    if not result.empty:
        result = result.set_index("window")
        result = result[FEATURE_COLUMNS + ["label"]]
    return result


class WindowAggregator:
    """Stream windows in chronological order and emit completed window summaries."""

    def __init__(self, window: float = 1.0, min_attack_flows: int = 3, min_attack_frac: float = 0.02):
        if window <= 0:
            raise ValueError("Window size must be positive.")
        self.window = float(window)
        self.min_attack_flows = int(min_attack_flows)
        self.min_attack_frac = float(min_attack_frac)
        self._seen_ts = None
        self._pending: dict[int, list[dict]] = defaultdict(list)
        self._flushed = False

    def _key_for(self, ts: float) -> int:
        return int(np.floor(float(ts) / self.window))

    def add(self, flow: dict) -> None:
        if self._flushed:
            raise ValueError("The aggregator has already been flushed; create a new instance to continue.")
        if "ts" not in flow:
            raise ValueError("A flow must include a 'ts' key.")
        flow_ts = float(flow["ts"])
        if self._seen_ts is not None and flow_ts < self._seen_ts:
            raise ValueError("Input flows must be chronological; out-of-order timestamps are not allowed.")
        self._seen_ts = flow_ts
        kp = self._key_for(flow_ts)
        self._pending[kp].append(flow)

    def add_many(self, flows) -> None:
        for flow in flows:
            self.add(flow)

    def pop_ready(self) -> pd.DataFrame:
        if self._seen_ts is None:
            return pd.DataFrame(columns=["window", *FEATURE_COLUMNS, "label"])
        current_key = self._key_for(float(self._seen_ts))
        ready_keys = sorted(key for key in self._pending if key < current_key)
        if not ready_keys:
            return pd.DataFrame(columns=["window", *FEATURE_COLUMNS, "label"])
        frames = []
        for key in ready_keys:
            batch = pd.DataFrame(self._pending.pop(key))
            batch = batch.sort_values("ts").reset_index(drop=True)
            if batch.empty:
                continue
            features = window_features(batch, window=self.window, min_attack_flows=self.min_attack_flows, min_attack_frac=self.min_attack_frac)
            features = features.reset_index().rename(columns={"index": "window"})
            frames.append(features)
        if not frames:
            return pd.DataFrame(columns=["window", *FEATURE_COLUMNS, "label"])
        result = pd.concat(frames, ignore_index=True)
        result = result[["window", *FEATURE_COLUMNS, "label"]]
        return result

    def flush(self) -> pd.DataFrame:
        if self._flushed:
            return pd.DataFrame(columns=["window", *FEATURE_COLUMNS, "label"])
        self._flushed = True
        if not self._pending:
            return pd.DataFrame(columns=["window", *FEATURE_COLUMNS, "label"])
        frames = []
        for key in sorted(self._pending):
            batch = pd.DataFrame(self._pending[key])
            if batch.empty:
                continue
            features = window_features(batch, window=self.window, min_attack_flows=self.min_attack_flows, min_attack_frac=self.min_attack_frac)
            features = features.reset_index().rename(columns={"index": "window"})
            frames.append(features)
        self._pending.clear()
        if not frames:
            return pd.DataFrame(columns=["window", *FEATURE_COLUMNS, "label"])
        result = pd.concat(frames, ignore_index=True)
        return result[["window", *FEATURE_COLUMNS, "label"]]
