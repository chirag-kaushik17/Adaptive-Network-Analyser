import time
import platform
import numpy as np
import pandas as pd
import pytest
import sklearn

from ata.labels import label_family
from ata.loader import InvalidCICHeaderError, load
from ata.machine import core_count, cpu_model


def _row(timestamp, label="BENIGN", backward_name="Total Backward Packets"):
    row = {
        " Timestamp ": timestamp,
        " Source IP ": "192.0.2.1",
        " Destination IP ": "198.51.100.2",
        " Destination Port ": 443,
        " Protocol ": 17,
        " Flow Duration ": 2_500_000,
        " Total Fwd Packets ": 3,
        " Total Length of Fwd Packets ": 120,
        " Total Length of Bwd Packets ": 80,
        " SYN Flag Count ": 2,
        " Label ": label,
    }
    row[f" {backward_name} "] = 4
    return row


def test_auto_and_forced_date_orders_preserve_relative_seconds(tmp_path):
    path = tmp_path / "dates.csv"
    pd.DataFrame(
        [
            _row("3/7/2017 8:55"),
            _row("3/7/2017 8:56"),
            _row("3/7/2017 12:01"),
            _row("3/7/2017 1:02"),
            _row("3/7/2017 2:03"),
            _row("3/7/2017 3:04"),
            _row("3/7/2017 4:05"),
        ]
    ).to_csv(path, index=False)

    auto, auto_stats = load(path, "cicids2017", return_stats=True)
    forced, forced_stats = load(
        path, "cicids2017", return_stats=True, date_order="mdy"
    )
    assert auto_stats["date_order_used"] == "dmy"
    assert forced_stats["date_order_used"] == "mdy"
    assert auto_stats["date_range"] == ("2017-07-03", "2017-07-03")
    assert forced_stats["date_range"] == ("2017-03-07", "2017-03-07")
    assert auto["ts"].tolist() == forced["ts"].tolist()
    assert auto["ts"].is_monotonic_increasing


def test_ambiguous_identical_dates_and_neither_valid_behaviors(tmp_path):
    same = tmp_path / "same.csv"
    pd.DataFrame([_row("7/7/2017 8:55"), _row("7/7/2017 8:56")]).to_csv(
        same, index=False
    )
    _, stats = load(same, "cicids2017", return_stats=True)
    assert stats["date_order_used"] == "dmy"

    outside = tmp_path / "outside.csv"
    pd.DataFrame([_row("3/8/2017 8:55")]).to_csv(outside, index=False)
    with pytest.raises(ValueError, match="pass an explicit --date-order"):
        load(outside, "cicids2017")
    forced, stats = load(
        outside, "cicids2017", date_order="mdy", return_stats=True
    )
    assert len(forced) == 1
    assert stats["date_range"] == ("2017-03-08", "2017-03-08")
    unambiguous = tmp_path / "unambiguous_outside.csv"
    pd.DataFrame([_row("13/8/2017 8:55")]).to_csv(
        unambiguous, index=False
    )
    any_date, _ = load(
        unambiguous, "cicids2017", allow_any_dates=True, return_stats=True
    )
    assert len(any_date) == 1


def test_missing_columns_are_reported_together_with_trafficlabelling_hint(tmp_path):
    path = tmp_path / "bad_header.csv"
    pd.DataFrame({" Timestamp ": ["7/3/2017 8:55"]}).to_csv(path, index=False)
    with pytest.raises(InvalidCICHeaderError) as error:
        load(path, "cicids2017")
    message = str(error.value)
    for missing in [
        "Source IP", "Destination IP", "Destination Port", "Protocol",
        "Flow Duration", "Total Fwd Packets",
        "Total Length of Fwd Packets", "Total Length of Bwd Packets",
        "SYN Flag Count", "Label",
    ]:
        assert missing in message
    assert "Total Backward Packets (or Total Bwd Packets)" in message
    assert "TrafficLabelling" in message


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("BENIGN", "normal"),
        ("PortScan", "portscan"),
        ("DDoS", "ddos"),
        ("DoS Hulk", "dos"),
        ("DoS GoldenEye", "dos"),
        ("DoS Slowloris", "dos"),
        ("DoS Slowhttptest", "dos"),
        ("FTP-Patator", "bruteforce"),
        ("SSH-Patator", "bruteforce"),
        ("Web Attack \x96 XSS", "webattack"),
        ("Bot", "botnet"),
        ("Infiltration", "infiltration"),
        ("Heartbleed", "other"),
        ("unlisted family", "other"),
    ],
)
def test_cic_label_family_mapping(raw, expected):
    cleaned = "".join(character.lower() for character in raw if character.isalnum())
    assert label_family(cleaned) == expected


def test_unknown_label_warns_and_maps_to_other(tmp_path):
    path = tmp_path / "unknown.csv"
    pd.DataFrame([_row("7/3/2017 8:55", "Novel Attack")]).to_csv(
        path, index=False
    )
    with pytest.warns(UserWarning, match="novelattack"):
        flows, stats = load(path, "cicids2017", return_stats=True)
    assert flows.loc[0, "label"] == "other"
    assert flows.loc[0, "label_raw"] == "novelattack"
    assert stats["unknown_raw_labels"] == ["novelattack"]


def test_cic_loader_maps_every_canonical_family(tmp_path):
    names = [
        "BENIGN", "PortScan", "DDoS", "DoS Hulk", "DoS GoldenEye",
        "DoS Slowloris", "DoS Slowhttptest", "FTP-Patator", "SSH-Patator",
        "Web Attack – XSS", "Bot", "Infiltration", "Heartbleed",
    ]
    expected = [
        "normal", "portscan", "ddos", "dos", "dos", "dos", "dos",
        "bruteforce", "bruteforce", "webattack", "botnet", "infiltration",
        "other",
    ]
    path = tmp_path / "families.csv"
    pd.DataFrame(
        [_row("7/7/2017 8:55", label) for label in names]
    ).to_csv(path, index=False, encoding="cp1252")
    flows = load(path, "cicids2017")
    assert flows["label"].tolist() == expected


def test_utf8_and_latin1_files_are_read_and_cleaned(tmp_path):
    utf8_path = tmp_path / "utf8.csv"
    pd.DataFrame([_row("7/3/2017 8:55", "Web Attack – XSS")]).to_csv(
        utf8_path, index=False, encoding="utf-8"
    )
    utf8_flows = load(utf8_path, "cicids2017")
    assert utf8_flows.loc[0, "label_raw"] == "webattackxss"
    assert utf8_flows.loc[0, "label"] == "webattack"

    latin1_path = tmp_path / "latin1.csv"
    content = pd.DataFrame([_row("7/3/2017 8:55", "Web Attack – XSS")]).to_csv(
        index=False
    )
    latin1_path.write_bytes(content.encode("cp1252"))
    latin1_flows = load(latin1_path, "cicids2017")
    assert latin1_flows.loc[0, "label_raw"] == "webattackxss"
    assert latin1_flows.loc[0, "label"] == "webattack"


def test_backward_packet_alias_and_safe_numeric_coercion(tmp_path):
    path = tmp_path / "alias.csv"
    row = _row("7/3/2017 8:55", backward_name="Total Bwd Packets")
    row[" Flow Duration "] = np.inf
    row[" Destination Port "] = np.nan
    pd.DataFrame([row]).to_csv(path, index=False)
    flows = load(path, "cicids2017")
    assert flows.loc[0, "packets"] == 7
    assert flows.loc[0, "dport"] == 0
    assert flows.loc[0, "duration"] == 0


def test_100k_trafficlabelling_rows_load_under_ten_seconds(tmp_path):
    path = tmp_path / "large_trafficlabelling.csv"
    timestamps = [
        "3/7/2017 8:55",
        "7/7/2017 8:59",
        "3/7/2017 12:01",
        "3/7/2017 1:02",
        "3/7/2017 2:03",
        "3/7/2017 3:04",
        "3/7/2017 4:05",
    ]
    rows = [
        _row(timestamps[index % len(timestamps)])
        for index in range(100_000)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)

    started = time.perf_counter()
    flows = load(path, "cicids2017")
    elapsed = time.perf_counter() - started
    print(
        "100k loader performance: "
        f"{elapsed:.3f}s; CPU={cpu_model()}; "
        f"logical_cores={core_count()}; Python={platform.python_version()}; "
        f"numpy={np.__version__}; pandas={pd.__version__}; "
        f"scikit-learn={sklearn.__version__}"
    )
    assert len(flows) == 100_000
    assert elapsed < 10.0
