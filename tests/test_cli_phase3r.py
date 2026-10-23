import sys

import pandas as pd
import pytest

from ata.run import main


def test_missing_cic_file_is_skipped_with_download_instructions(
    monkeypatch, tmp_path, capsys
):
    missing = tmp_path / "Monday-WorkingHours.pcap_ISCX.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        ["ata.run", "--dataset", "cicids2017", "--csv", str(missing)],
    )
    main()
    output = capsys.readouterr().out
    assert "https://www.unb.ca/cic/datasets/ids-2017.html" in output
    assert "CHECK: SKIPPED (CIC CSV is missing)." in output


def test_invalid_cic_header_is_skipped_but_invalid_data_is_failed(
    monkeypatch, tmp_path, capsys
):
    invalid_header = tmp_path / "invalid_header.csv"
    pd.DataFrame({"Timestamp": ["7/3/2017 8:55"]}).to_csv(
        invalid_header, index=False
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ata.run", "--dataset", "cicids2017", "--csv",
            str(invalid_header),
        ],
    )
    main()
    output = capsys.readouterr().out
    assert "CHECK: SKIPPED (invalid CIC header:" in output

    invalid_data = tmp_path / "invalid_data.csv"
    row = {
        "Timestamp": "invalid timestamp",
        "Source IP": "192.0.2.1",
        "Destination IP": "198.51.100.2",
        "Destination Port": 80,
        "Protocol": 6,
        "Flow Duration": 1,
        "Total Fwd Packets": 1,
        "Total Backward Packets": 1,
        "Total Length of Fwd Packets": 10,
        "Total Length of Bwd Packets": 10,
        "SYN Flag Count": 0,
        "Label": "BENIGN",
    }
    pd.DataFrame([row]).to_csv(invalid_data, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ata.run", "--dataset", "cicids2017", "--csv",
            str(invalid_data),
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    assert "CHECK: FAILED (ValueError:" in capsys.readouterr().err
