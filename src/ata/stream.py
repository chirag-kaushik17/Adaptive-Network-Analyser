"""Replay canonical ATA flow CSVs in timestamp order."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .loader import load


class FlowStream:
    """Iterate rows returned by the shared flow loader without another copy."""

    def __init__(
        self,
        path: str | Path,
        *,
        dataset: str = "synth",
        speed: float = 0.0,
        date_order: str = "auto",
        allow_any_dates: bool = False,
    ):
        if speed < 0:
            raise ValueError("Replay speed must be zero or a positive multiplier.")
        self.path = Path(path)
        self.dataset = dataset
        self.speed = float(speed)
        self.date_order = date_order
        self.allow_any_dates = allow_any_dates
        self.load_stats: dict[str, Any] = {}

    def __iter__(self) -> Iterator[dict[str, Any]]:
        frame, self.load_stats = load(
            self.path,
            self.dataset,
            return_stats=True,
            date_order=self.date_order,
            allow_any_dates=self.allow_any_dates,
        )
        previous_ts: float | None = None
        for row in frame.itertuples(index=False, name=None):
            flow = dict(zip(frame.columns, row))
            timestamp = float(flow["ts"])
            if previous_ts is not None:
                if timestamp < previous_ts:
                    raise ValueError("Flow source returned out-of-order timestamps.")
                if self.speed > 0:
                    time.sleep((timestamp - previous_ts) / self.speed)
            previous_ts = timestamp
            yield flow


def iter_flow_batches(
    flows: Iterator[dict[str, Any]], batch_size: int
) -> Iterator[list[dict[str, Any]]]:
    """Group a flow iterator into bounded request batches."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least one.")
    batch: list[dict[str, Any]] = []
    for flow in flows:
        batch.append(flow)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
