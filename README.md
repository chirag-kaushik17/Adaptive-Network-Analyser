# Adaptive Network Traffic Analyzer

ATA builds a flow-level network traffic analyzer in phases. This repository
contains the CIC/synthetic data layer, time-window features, unsupervised
Layer 1 Isolation Forest baselines, PSI/KS drift detection, supervised Layer 2
classifiers with SHAP explanations, a replayable streaming API, SQLite alert
storage, and a local Streamlit dashboard. The frozen Phase 5 Layer 1 research
result remains **FAIL / NO-GO**; this does not prevent the later engineering
phases from running.

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

Named deterministic scenarios are available with `--scenario stable`,
`--scenario stable_attacks`, `--scenario drift`, and
`--scenario drift_attacks`. The default remains the existing attack schedule
without drift. `stable` contains normal traffic only; `stable_attacks` is the
hard negative for false drift events; `drift` contains normal traffic with a
gradual change after warm-up; and `drift_attacks` combines the change and
attack schedule. A named scenario takes precedence over `--no-attacks` and
`--drift`. The sidecar records drift start/end and is used only for evaluation
delay, never as model input.

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

Drift detection can be selected with `--drift-method psi|ks`; use
`--refit-trigger drift` to refit after a persistent drift event instead of on
the periodic interval. The default is `--refit-trigger periodic`, preserving
Phase 3R behavior. The CLI reports drift event count, first event window,
detection delay when sidecar metadata is available, and refit count. The
monitor receives feature rows only. Drift-triggered refits use recent stream
history, including flagged rows, so a rejected new regime can be learned; this
differs intentionally from ordinary unflagged-buffer admission and can admit
attack contamination.

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

Compare static, periodic-adaptive, and drift-adaptive Layer 1 baselines on
the fixed evaluation seeds and generate JSON/PNG reports with:

```powershell
python -m ata.evaluate --seeds 200 201 202 203 204 --out reports
```

The evaluation reports synthetic results only unless CIC data is separately
provided. Its frozen GO/NO-GO verdict is a **research acceptance result**, not
an engineering test gate; Phase 6 can be implemented and tested while the
Layer 1 result remains NO-GO. Never represent a NO-GO as a PASS.

## Layer 2 classifier and explainability

Layer 2 compares a class-weighted XGBoost classifier with a class-weighted
Random Forest. Hyperparameters and model choice are selected on development
seeds 1–3 only and recorded under `classifier` in
`configs/defaults.json`. The reported run uses seeds 200–204. For each seed,
synthetic attack order is shuffled and five cycles ensure every target class
appears on both sides of the chronological split. The first 65% of windows
are training data, followed by a five-window gap and the test segment. The
classifier input is exactly the 11 columns in `FEATURE_COLUMNS`; labels are
targets only, and window indices are used only to preserve order.

Run the classifier evaluation and generate reports:

```powershell
python -m ata.classify --seeds 200 201 202 203 204 --out reports
```

The CLI prints per-class precision, recall, F1, and support for both models,
plus macro averages. It writes `reports/classifier_metrics.json`,
`reports/confusion_matrix_xgboost.png`,
`reports/confusion_matrix_random_forest.png`, `reports/shap_summary.png`, and
`reports/alerts_sample.jsonl`. Each sample alert has exactly three
class-specific SHAP features. The JSON report keeps Layer 1 research
acceptance separate from Phase 6 engineering status. These are synthetic
demonstration results, not evidence of real-world performance; CIC remains
SKIPPED until TrafficLabelling data is provided.

## Streaming API and dashboard

The streaming source reuses the existing flow loader and schema. It orders
flows by timestamp, feeds them incrementally into the same window aggregator
and Layer 1 baseline, and flushes the final partial window. Alerts are stored
as JSON-backed records in SQLite; raw packet contents are not stored.

Start the local API from PowerShell:

```powershell
python -m ata.serve --host 127.0.0.1 --port 8000 --database data/ata_alerts.sqlite3
```

The service exposes `GET /health`, `POST /flows`, `POST /score`,
`POST /flush`, `GET /alerts`, `GET /alerts/{alert_id}`, and `GET /stats`.
Replay an existing synthetic CSV into it from a second terminal:

```powershell
python -m ata.replay --csv data/synth.csv --dataset synth --url http://127.0.0.1:8000 --batch 2000 --speed 0
```

`--speed 0` replays as quickly as possible; a positive speed is a timing
multiplier. The replay command finalizes the last incomplete window and writes
`reports/phase7_replay_summary.json`. CIC CSVs can be replayed with
`--dataset cicids2017` and the loader's date-order options.

By default there is no saved Phase 6 model artifact, so the server emits
explicit `unknown` classifications and no SHAP features for Layer 1 alerts.
For an optional in-memory classifier demonstration, start the server with
`--with-classifier`; this trains from development seed 1 and does not load or
create a persistent model artifact.

Run the Streamlit dashboard in another terminal:

```powershell
python -m streamlit run src/ata/dashboard.py
```

It connects to `http://127.0.0.1:8000` by default. Set `ATA_API_URL` to change
the endpoint. The dashboard separates current API telemetry and alerts from
the saved historical Phase 5/6 evaluation reports; a Phase 5 NO-GO remains
visible and is not converted into an engineering status.

Measure local throughput with:

```powershell
python -m ata.bench --csv data/synth.csv --dataset synth --batch 2000 --out reports/throughput.json
```

The benchmark reports aggregation, full-pipeline, and in-process FastAPI
`TestClient` timings. The API timing excludes Uvicorn socket/process overhead;
all such throughput values are local engineering measurements, not capacity
guarantees or evidence of real-world detection performance.

**Security and data limitations:** the API has no authentication or TLS and
must stay bound to localhost; do not expose it to an untrusted network. The
existing CSV loader materializes its input before replay, so replay is
incremental after loading but is not a chunked large-file reader. Demonstration
results are synthetic; real CIC validation is **SKIPPED** until
TrafficLabelling CSVs are supplied.
