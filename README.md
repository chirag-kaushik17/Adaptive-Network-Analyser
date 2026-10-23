# Adaptive Network Traffic Analyzer

ATA builds a flow-level anomaly detector in phases. The implemented scope in
this repository is Phases 0–3R: the CIC/synthetic data layer, time-window
features, and the unsupervised Layer 1 Isolation Forest baselines. Drift
detection, a supervised classifier, explanations, and a serving API are not
implemented in this phase.

## Environment and tests

Python 3.11 or newer is required. From PowerShell at the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
```

The runtime dependencies used by this phase are pinned in
[requirements.txt](requirements.txt).

## CIC-IDS2017 TrafficLabelling data

Use the **TrafficLabelling** flow CSVs, not MachineLearningCVE. Download from
<https://www.unb.ca/cic/datasets/ids-2017.html> and place files in `data/`,
for example:

```text
data/Monday-WorkingHours.pcap_ISCX.csv
data/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
```

CSV headers are stripped of surrounding spaces. The loader accepts both
`Total Backward Packets` and `Total Bwd Packets`. It tries UTF-8 first and
falls back to cp1252, then latin-1, on decoding errors. Timestamps are
minute-resolution 12-hour values without AM/PM. The documented capture-time
assumption is that the capture runs approximately 08:00–17:00; parsed hours
below 8 are interpreted as PM by adding 12 hours to the same date. Timestamps
are stored as seconds relative to the earliest valid timestamp and sorted
chronologically. Invalid timestamp rows are counted and discarded.

Automatic date-order detection accepts DMY or MDY only when the parsed dates
fit the CIC capture dates, 2017-07-02 through 2017-07-08. If the order cannot be
resolved, pass `--date-order dmy`, `--date-order mdy`, or
`--allow-any-dates`. For the intended single-day CIC files, date-order
selection does not change relative seconds; it matters when combining files
or comparing absolute dates.

Ground-truth label families:

| Cleaned raw label | Canonical `label` |
| --- | --- |
| `benign` | `normal` |
| `portscan` | `portscan` |
| `ddos` | `ddos` |
| `doshulk`, `dosgoldeneye`, `dosslowloris`, `dosslowhttptest` | `dos` |
| `ftppatator`, `sshpatator` | `bruteforce` |
| `webattack...` | `webattack` |
| `bot` | `botnet` |
| `infiltration` | `infiltration` |
| `heartbleed` | `other` |
| unknown | `other` (with a warning listing the raw names) |

The loader also preserves `label_raw`, the cleaned raw label. CIC contains no
`exfiltration` family; that class is synthetic-only. Labels are used for
evaluation and window labels, never as Layer 1 model features or for baseline
admission.

A window label is an evaluation target: it is an attack when it contains at
least three attack flows **or** attack flows make up at least 2% of that
window. The dominant attack family wins, with lexical tie-breaking. As a
result, windows with a small attack fraction can still be labeled as attacks;
this is a known source of label noise for CIC evaluation.

If CIC files are absent, real-data checks are marked **SKIPPED** with download
instructions. CSV files under `data/`, logs, model artifacts, and generated
reports are excluded from Git.

## Synthetic scenarios

Generate the default three-cycle attack scenario (900 seconds total):

```powershell
python -m ata.synth --out data/synth.csv --data-seed 100
```

The default clean warm-up is 120 windows of 1 second each. Attacks start at
120 seconds, with portscan, DDoS, and exfiltration episodes in each cycle,
30 seconds per episode, 40 seconds of normal-only traffic between episodes,
and a 190-second normal tail. The schedule and generation settings are saved
to `data/synth.csv.meta.json`; this file is only a consistency check and is
never model input. `--data-seed` drives generated data (`--seed` is an alias).
Drift is independent of attacks and controlled by
`--drift-start`, `--drift-end`, and `--drift-strength`; `--drift` is shorthand
for a unit-strength drift. `--no-attacks` prevents attack and stealth flows.
`--attack-order shuffled` deterministically shuffles each cycle for later
classifier evaluation.

Every duration parameter is in seconds unless its name explicitly says
`windows`. Warm-up and history values are windows. Defaults assume a 1-second
window. CIC commands use 60-second windows, so an equal number of windows
represents 60 times as much wall-clock traffic.

## Layer 1 evaluation

Run the adaptive baseline against an existing synthetic CSV:

```powershell
python -m ata.run --dataset synth --csv data/synth.csv
```

The CLI only evaluates the CSV; it does not generate or edit it. It defaults
to a 1-second window, 120 warm-up windows, 300 history windows, periodic
refits every 30 post-warm-up windows, and model seed 0. The selected
contamination value and the development-seed evidence are in
`configs/defaults.json`. Compare with a static warm-up-only baseline using
`--baseline static`. Synthetic output is **not** evidence of real-world
detection quality.

For a CIC TrafficLabelling file, use settings appropriate to 60-second
windows, for example:

```powershell
python -m ata.run --dataset cicids2017 --csv data/Monday-WorkingHours.pcap_ISCX.csv --window 60 --warmup 60 --history 200 --date-order auto
```

The baseline runs scaler and Isolation Forest inference using only the 11
numeric window features. It scores before admission, and flagged windows are
never admitted to the adaptive buffer. This poison-protection rule does **not**
guarantee an attack-free buffer: false negatives can still enter it. Layer 1
is unsupervised; attack labels do not determine inference flags or buffer
admission.

Synthetic and real CIC results must always be reported separately. The
synthetic generator is simplified and deterministic, and does not establish
real-world performance.
