from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd

from . import synth


def _clean_label(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if not text:
        return "normal"
    text = text.replace("\xa0", " ")
    text = re.sub(r"[^A-Za-z0-9]+", "", text)
    lowered = text.lower()
    return "normal" if lowered in {"benign", "normal"} else lowered


def _coerce_timestamp(value: object) -> pd.Timestamp | pd.NaT:
    if pd.isna(value):
        return pd.NaT
    candidate = str(value).strip().replace("T", " ")
    if not candidate:
        return pd.NaT

    if re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?", candidate):
        parts = candidate.split(":")
        hour = int(parts[0])
        minute = int(parts[1])
        second = int(parts[2]) if len(parts) == 3 else 0
        dt = pd.Timestamp("2000-01-01 00:00:00") + pd.Timedelta(
            hours=hour, minutes=minute, seconds=second
        )
        if hour < 8:
            dt += pd.Timedelta(hours=12)
        return dt

    parsed = pd.to_datetime(candidate, errors="coerce", dayfirst=True)
    if pd.isna(parsed):
        return pd.NaT
    if parsed.hour < 8:
        # CIC-IDS2017 capture times are assumed to be roughly 08:00-17:00.
        # Under that assumption, hours before 8 are PM values from the same day.
        parsed = parsed + pd.Timedelta(hours=12)
    return parsed


def _cicids2017_to_flows(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    cleaned = frame.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]

    if "Timestamp" not in cleaned.columns:
        raise ValueError("CIC-IDS2017 file is missing the 'Timestamp' column.")

    cleaned["_parsed_ts"] = cleaned["Timestamp"].map(_coerce_timestamp)
    valid = cleaned[cleaned["_parsed_ts"].notna()].copy()
    invalid_count = int(len(cleaned) - len(valid))

    if valid.empty:
        raise ValueError("No valid timestamps were found in the CIC-IDS2017 dataset.")

    start = valid["_parsed_ts"].min()
    valid["ts"] = (valid["_parsed_ts"] - start).dt.total_seconds()

    def get_value(name: str, default: object = None):
        return valid[name].iloc[0] if name in valid.columns else default

    if "Source IP" in valid.columns:
        valid["src"] = valid["Source IP"].astype(str).str.strip()
    else:
        valid["src"] = "unknown"
    if "Destination IP" in valid.columns:
        valid["dst"] = valid["Destination IP"].astype(str).str.strip()
    else:
        valid["dst"] = "unknown"

    valid["dport"] = pd.to_numeric(valid["Destination Port"], errors="coerce")
    valid["proto"] = valid["Protocol"].map(lambda v: {6: "tcp", 17: "udp"}.get(int(v), "other") if pd.notna(v) else "other")

    valid["packets"] = pd.to_numeric(valid["Total Fwd Packets"] + valid["Total Bwd Packets"], errors="coerce")
    valid["bytes"] = pd.to_numeric(valid["Total Length of Fwd Packets"] + valid["Total Length of Bwd Packets"], errors="coerce")
    valid["duration"] = pd.to_numeric(valid["Flow Duration"], errors="coerce") / 1_000_000.0
    valid["syn"] = valid["SYN Flag Count"].fillna(0)
    valid["syn"] = valid["syn"].map(lambda v: 1 if pd.notna(v) and float(v) > 0 else 0)

    valid["label"] = valid["Label"].map(_clean_label)
    valid["label"] = valid["label"].replace({"benign": "normal"})

    for column in ["bytes", "packets", "duration", "syn", "dport"]:
        valid[column] = pd.to_numeric(valid[column], errors="coerce").fillna(0)
    valid["bytes"] = valid["bytes"].astype(float)
    valid["packets"] = valid["packets"].astype(float)
    valid["duration"] = valid["duration"].astype(float)
    valid["dport"] = valid["dport"].astype(int)

    required = ["ts", "src", "dst", "dport", "proto", "bytes", "packets", "duration", "syn", "label"]
    output = valid[required].copy()
    output = output.sort_values("ts").reset_index(drop=True)
    output["dport"] = output["dport"].astype(int)
    output["proto"] = output["proto"].astype(str)
    output["label"] = output["label"].astype(str)
    return output, invalid_count


def load(path: str | Path, dataset: str, nrows: int | None = None, return_stats: bool = False):
    """Load flows from a synthetic CSV or a CIC-IDS2017 CSV."""
    path = Path(path)
    if dataset == "synth":
        frame = pd.read_csv(path, nrows=nrows)
        frame = frame.copy()
        if "ts" not in frame.columns:
            raise ValueError("Synthetic CSV is missing the 'ts' column.")
        frame["dport"] = pd.to_numeric(frame["dport"], errors="coerce").fillna(0).astype(int)
        frame["bytes"] = pd.to_numeric(frame["bytes"], errors="coerce").fillna(0.0)
        frame["packets"] = pd.to_numeric(frame["packets"], errors="coerce").fillna(0.0)
        frame["duration"] = pd.to_numeric(frame["duration"], errors="coerce").fillna(0.0)
        frame["syn"] = pd.to_numeric(frame["syn"], errors="coerce").fillna(0).astype(int)
        frame["label"] = frame["label"].fillna("normal").astype(str)
        frame["proto"] = frame["proto"].fillna("other").astype(str)
        frame = frame.sort_values("ts").reset_index(drop=True)
        stats = {"discarded_invalid_timestamps": 0}
        return (frame, stats) if return_stats else frame

    if dataset == "cicids2017":
        frame = pd.read_csv(path, encoding="latin-1", nrows=nrows)
        frame.columns = [str(col).strip() for col in frame.columns]
        output, invalid_count = _cicids2017_to_flows(frame)
        stats = {"discarded_invalid_timestamps": invalid_count}
        return (output, stats) if return_stats else output

    raise ValueError(f"Unsupported dataset name: {dataset!r}")
