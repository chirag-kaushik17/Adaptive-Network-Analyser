"""Load CIC-IDS2017 and synthetic flow CSVs into a common schema."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .labels import map_raw_labels

FLOW_COLUMNS = [
    "ts",
    "src",
    "dst",
    "dport",
    "proto",
    "bytes",
    "packets",
    "duration",
    "syn",
    "label",
    "label_raw",
]

_CIC_REQUIRED = [
    "Timestamp",
    "Source IP",
    "Destination IP",
    "Destination Port",
    "Protocol",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "SYN Flag Count",
    "Label",
]
_DATE_MIN = pd.Timestamp("2017-07-02")
_DATE_MAX = pd.Timestamp("2017-07-08 23:59:59")


class InvalidCICHeaderError(ValueError):
    """Raised when a CSV does not have the CIC TrafficLabelling schema."""


def _read_cic_csv(path: Path, nrows: int | None) -> pd.DataFrame:
    """Read UTF-8 first, falling back only when decoding fails."""
    try:
        return pd.read_csv(path, encoding="utf-8", nrows=nrows)
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, encoding="cp1252", nrows=nrows)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin-1", nrows=nrows)


def _parse_dates(values: pd.Series, dayfirst: bool) -> pd.Series:
    cleaned = values.astype("string").str.strip()
    cleaned = cleaned.str.replace(r"^[A-Za-z]+-", "", regex=True)
    return pd.to_datetime(cleaned, format="mixed", dayfirst=dayfirst, errors="coerce")


def _date_order_candidates(values: pd.Series, date_order: str, allow_any_dates: bool):
    options = {"dmy": True, "mdy": False}
    if date_order == "auto":
        parsed = {name: _parse_dates(values, dayfirst=dayfirst) for name, dayfirst in options.items()}
        if allow_any_dates:
            valid = {name: dates.notna().any() for name, dates in parsed.items()}
        else:
            valid = {
                name: dates.notna().any()
                and dates.dropna().between(_DATE_MIN, _DATE_MAX).all()
                for name, dates in parsed.items()
            }
        candidates = [name for name, is_valid in valid.items() if is_valid]
        if len(candidates) == 1:
            selected = candidates[0]
        elif len(candidates) == 2:
            dmy_dates = parsed["dmy"].dropna().dt.normalize()
            mdy_dates = parsed["mdy"].dropna().dt.normalize()
            if dmy_dates.equals(mdy_dates):
                selected = "dmy"
            else:
                raise ValueError(
                    "CIC timestamp date order is ambiguous; pass --date-order dmy or "
                    "--date-order mdy (or --allow-any-dates for dates outside the "
                    "CIC-IDS2017 capture period)."
                )
        elif allow_any_dates and parsed["dmy"].notna().any():
            selected = "dmy"
        else:
            hint = " Pass --allow-any-dates if these dates are intentional."
            raise ValueError(
                "Neither DMY nor MDY timestamps fall within the CIC-IDS2017 capture "
                "period (2017-07-02 through 2017-07-08); pass an explicit "
                "--date-order or --allow-any-dates." + hint
            )
        return parsed[selected], selected

    if date_order not in options:
        raise ValueError("date_order must be 'auto', 'dmy', or 'mdy'.")
    return _parse_dates(values, dayfirst=options[date_order]), date_order


def _required_columns(columns: list[str]) -> list[str]:
    available = set(columns)
    missing = [name for name in _CIC_REQUIRED if name not in available]
    if (
        "Total Backward Packets" not in available
        and "Total Bwd Packets" not in available
    ):
        missing.append("Total Backward Packets (or Total Bwd Packets)")
    return missing


def _cicids2017_to_flows(
    frame: pd.DataFrame,
    date_order: str = "auto",
    allow_any_dates: bool = False,
) -> tuple[pd.DataFrame, dict]:
    cleaned = frame.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    missing = _required_columns(list(cleaned.columns))
    if missing:
        raise InvalidCICHeaderError(
            "Missing required CIC-IDS2017 columns: "
            + ", ".join(missing)
            + ". Use the CIC-IDS2017 TrafficLabelling CSVs (not MachineLearningCVE)."
        )

    parsed, date_order_used = _date_order_candidates(
        cleaned["Timestamp"], date_order=date_order, allow_any_dates=allow_any_dates
    )
    valid_timestamp = parsed.notna()
    invalid_count = int((~valid_timestamp).sum())
    if not valid_timestamp.any():
        raise ValueError("No valid CIC-IDS2017 timestamps were found.")

    parsed = parsed.loc[valid_timestamp].copy()
    # CIC capture hours are assumed to be 08:00-17:00; pre-08:00 values are PM.
    before_8 = parsed.dt.hour < 8
    parsed.loc[before_8] = parsed.loc[before_8] + pd.Timedelta(hours=12)
    valid = cleaned.loc[valid_timestamp].copy()
    start = parsed.min()
    valid["ts"] = (parsed - start).dt.total_seconds().to_numpy()

    backward_packets = (
        "Total Backward Packets"
        if "Total Backward Packets" in valid.columns
        else "Total Bwd Packets"
    )
    packet_count = (
        pd.to_numeric(valid["Total Fwd Packets"], errors="coerce")
        + pd.to_numeric(valid[backward_packets], errors="coerce")
    )
    byte_count = (
        pd.to_numeric(valid["Total Length of Fwd Packets"], errors="coerce")
        + pd.to_numeric(valid["Total Length of Bwd Packets"], errors="coerce")
    )
    protocol = pd.to_numeric(valid["Protocol"], errors="coerce")

    valid["src"] = valid["Source IP"].astype("string").str.strip().fillna("unknown")
    valid["dst"] = valid["Destination IP"].astype("string").str.strip().fillna("unknown")
    valid["dport"] = pd.to_numeric(valid["Destination Port"], errors="coerce")
    valid["proto"] = np.select(
        [protocol.eq(6), protocol.eq(17)], ["tcp", "udp"], default="other"
    )
    valid["packets"] = packet_count
    valid["bytes"] = byte_count
    valid["duration"] = pd.to_numeric(valid["Flow Duration"], errors="coerce") / 1_000_000
    syn = pd.to_numeric(valid["SYN Flag Count"], errors="coerce").fillna(0)
    valid["syn"] = syn.gt(0).astype("int8")
    raw_labels, families, unknown = map_raw_labels(valid["Label"])
    valid["label_raw"] = raw_labels.to_numpy()
    valid["label"] = families.to_numpy()

    for column in ["dport", "bytes", "packets", "duration"]:
        valid[column] = pd.to_numeric(valid[column], errors="coerce")
        valid[column] = valid[column].replace([np.inf, -np.inf], np.nan).fillna(0)
    valid["dport"] = valid["dport"].clip(0, 65535).astype("int32")
    valid["bytes"] = valid["bytes"].astype(float)
    valid["packets"] = valid["packets"].astype(float)
    valid["duration"] = valid["duration"].astype(float)
    valid = valid.sort_values("ts", kind="stable")
    output = valid[FLOW_COLUMNS].reset_index(drop=True)

    stats = {
        "discarded_invalid_timestamps": invalid_count,
        "date_order_used": date_order_used,
        "date_range": (
            parsed.min().date().isoformat(),
            parsed.max().date().isoformat(),
        ),
        "label_counts": valid["label"].value_counts().sort_index().to_dict(),
        "unknown_raw_labels": unknown,
    }
    if unknown:
        warnings.warn(
            "Unrecognized CIC labels mapped to 'other': " + ", ".join(unknown),
            UserWarning,
            stacklevel=2,
        )
    return output, stats


def load(
    path: str | Path,
    dataset: str,
    nrows: int | None = None,
    return_stats: bool = False,
    date_order: str = "auto",
    allow_any_dates: bool = False,
):
    """Load flow rows and optional parsing statistics from a CSV file."""
    path = Path(path)
    if dataset == "synth":
        frame = pd.read_csv(path, nrows=nrows)
        required = {
            "ts", "src", "dst", "dport", "proto", "bytes", "packets",
            "duration", "syn", "label",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError("Synthetic CSV is missing required columns: " + ", ".join(missing))
        frame = frame.copy()
        for column in ["ts", "dport", "bytes", "packets", "duration", "syn"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        numeric = ["ts", "dport", "bytes", "packets", "duration", "syn"]
        frame[numeric] = frame[numeric].replace([np.inf, -np.inf], np.nan).fillna(0)
        frame["dport"] = frame["dport"].astype(int)
        frame["syn"] = frame["syn"].clip(0, 1).astype(int)
        frame["label"] = frame["label"].fillna("normal").astype(str).str.lower()
        frame["proto"] = frame["proto"].fillna("other").astype(str).str.lower()
        frame = frame.sort_values("ts", kind="stable").reset_index(drop=True)
        stats = {
            "discarded_invalid_timestamps": 0,
            "date_order_used": None,
            "date_range": None,
            "label_counts": frame["label"].value_counts().sort_index().to_dict(),
            "unknown_raw_labels": [],
        }
        return (frame, stats) if return_stats else frame

    if dataset == "cicids2017":
        frame = _read_cic_csv(path, nrows=nrows)
        output, stats = _cicids2017_to_flows(
            frame, date_order=date_order, allow_any_dates=allow_any_dates
        )
        return (output, stats) if return_stats else output

    raise ValueError(f"Unsupported dataset name: {dataset!r}")
