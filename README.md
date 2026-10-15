# Adaptive Network Traffic Analyzer

This project implements the first three phases of an adaptive network traffic analyzer: environment setup, a data-loading layer, feature-window generation, and an unsupervised layer-1 detector based on an isolation forest.

## Setup

1. Create and activate the local virtual environment:
   ```bash
   python -m venv .venv
   . .venv/Scripts/Activate.ps1   # PowerShell
   ```
2. Install the pinned requirements:
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   python -m pip install -e .
   ```
3. Run the test suite:
   ```bash
   python -m pytest -q
   ```

## Data assumptions

- CIC-IDS2017 timestamps are minute resolution and use a 12-hour clock without explicit AM/PM markers.
- The project follows the documented capture-window assumption that the capture runs roughly from 08:00 to 17:00. In that setting, a timestamp with an hour below 8 is treated as PM (+12h) to preserve the intended date and ordering.
- All dataset CSV files are intentionally kept out of Git via `.gitignore`.

## Manual CIC data note

If the CIC-IDS2017 `TrafficLabelling` CSV files are not downloaded locally, the project continues with synthetic data only. The loader expects files such as:

```text
data/Monday-WorkingHours.pcap_ISCX.csv
data/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
```

Download the data from: https://www.unb.ca/cic/datasets/ids-2017.html

## Synthetic generation and evaluation

Generate a synthetic dataset:

```bash
python -m ata.synth --out data/synth.csv --seconds 600 --seed 7 --attack-start 420
```

Run the baseline evaluation against that dataset:

```bash
python -m ata.run --dataset synth --csv data/synth.csv --window 1 --warmup 3 --baseline adaptive --seed 7
```

## Known limitations

- Poison protection is defensive but not a guarantee that the training buffer is totally attack-free, because false negatives can still let abnormal windows enter the adaptive history buffer.
- The synthetic attacker timing and traffic rates are intentionally simplified for deterministic tests and demos; they are not a substitute for evaluation on the full CIC-IDS2017 dataset.
- The layer-1 detector is intentionally unsupervised and is only one part of the larger adaptive anomaly system planned for later phases.

## Synthetic-performance caveat

Synthetic metrics are included for reproducible local validation only. They are not evidence of real-world detection quality on CIC-IDS2017 and should be reported separately from real-dataset measurements.
