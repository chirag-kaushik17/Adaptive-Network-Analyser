"""Canonical CIC-IDS2017 label families."""

from __future__ import annotations

import re

LABEL_FAMILY_ROWS = (
    ("benign", "normal"),
    ("portscan", "portscan"),
    ("ddos", "ddos"),
    ("doshulk / dosgoldeneye / dosslowloris / dosslowhttptest", "dos"),
    ("ftppatator / sshpatator", "bruteforce"),
    ("webattack*", "webattack"),
    ("bot", "botnet"),
    ("infiltration", "infiltration"),
    ("heartbleed", "other"),
    ("unrecognized", "other"),
)


def clean_raw_label(value: object) -> str:
    """Return a lowercase alphanumeric label without encoding artifacts."""
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def label_family(raw_label: str) -> str:
    """Map a cleaned CIC label to the project's attack-family vocabulary."""
    if raw_label in {"benign", "normal"}:
        return "normal"
    if raw_label == "portscan":
        return "portscan"
    if raw_label == "ddos":
        return "ddos"
    if raw_label.startswith("dos"):
        return "dos"
    if raw_label in {"ftppatator", "sshpatator"}:
        return "bruteforce"
    if raw_label.startswith("webattack"):
        return "webattack"
    if raw_label == "bot":
        return "botnet"
    if raw_label == "infiltration":
        return "infiltration"
    return "other"


def map_raw_labels(values):
    """Return cleaned raw labels, canonical families, and unknown raw names."""
    raw_labels = values.map(clean_raw_label)
    families = raw_labels.map(label_family)
    known = {
        "benign", "normal", "portscan", "ddos", "doshulk", "dosgoldeneye",
        "dosslowloris", "dosslowhttptest", "ftppatator", "sshpatator",
        "webattackbruteforce", "webattackxss", "webattacksqlinjection",
        "bot", "infiltration", "heartbleed",
    }
    unknown = sorted(set(raw_labels) - known)
    unknown = [name for name in unknown if name and label_family(name) == "other"]
    return raw_labels, families, unknown
