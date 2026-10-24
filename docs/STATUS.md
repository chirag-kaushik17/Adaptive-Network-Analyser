# ATA phase status

## Phase 7/8 starting audit (2026-10-24)

- The workspace contains reusable flow loading (`ata.loader.load`), batch and
  incremental window feature extraction (`window_features` and
  `WindowAggregator`), Layer 1 `AdaptiveBaseline`/`StaticBaseline` scoring and
  drift/adaptation counters, plus Phase 6 model construction, SHAP adaptation,
  and JSON alert construction.
- No streaming source/replay, unified online pipeline, alert database, FastAPI
  app, server entry point, API benchmark, or dashboard existed at audit time.
- Phase 6 currently trains models inside the evaluation CLI; it does not save
  a deployable model artifact. `models/` is absent. The serving pipeline must
  therefore make classifier use optional and must not claim a persisted model
  was loaded.
- Existing Phase 5/6 reports and plots are present under `reports/`. The
  Phase 5 research verdict is **FAIL / NO-GO** and will remain separate from
  Phase 7/8 engineering readiness.
- The active interpreter is Python 3.14.6. FastAPI 0.141.1, Uvicorn 0.52.4,
  HTTPX 0.28.1 and Pydantic 2.13.5 are installed; Streamlit is not installed.
- Initial Phase 7/8 status: **NOT IMPLEMENTED**. No feature/schema or frozen
  research criteria changes are needed for the planned serving work.

## Completed before Phase 3R

- Phase 0 foundation: commit `d127e52`.
- Phase 1 flow loading and synthetic generator: commit `17ea44f`.
- Phase 2 feature windows: commit `4f0e4f7`.
- Phase 3 initial Layer 1 baseline: commit `05e2d4b`.
- Phase 3R baseline tag: `ata-before-3R` at `05e2d4b`.

## Phase 3R (completed)

- Current branch: `main`.
- Baseline suite before changes: **12 passed** (`logs/phase3R_baseline_pytest.txt`).
- CIC loader: vectorized timestamp parsing and conversion; supports both
  backward-packet header spellings, UTF-8 with cp1252/latin-1 fallback, family
  labels, automatic DMY/MDY selection and explicit overrides.
- Loader 100k performance measured at **0.803 seconds** in the final full
  suite on Windows 11, Intel Core Ultra 5 125H, 14 physical/18 logical cores,
  Python 3.13.5, NumPy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1.
- Default synthetic schedule: three cycles, nine attacks, 30 seconds each,
  eight 40-second normal gaps, a 190-second normal tail and 120-second clean
  warm-up; total schedule duration 900 seconds.
- Layer 1 baseline supports a `history` shorter than warm-up, `min_fit_rows`,
  skipped-refit accounting, `score_only`, static-only warm-up storage and
  disabled periodic refits using `refit_every=None`.
- Development contamination sweep (dev seeds 1–3, model seed 0): selected
  contamination **0.005**, with mean scenario-A recall 0.6062 and mean false
  alarm rate across scenarios A+B of 0.0070. Full output is in
  `logs/phase3R_contamination_sweep.txt`.
- Frozen smoke-seed acceptance at contamination 0.005: mean attack recall
  **0.7716**, attack FAR **0.0052**, and normal-only FAR **0.0162**; all fixed
  aggregate acceptance thresholds passed. Seed 101 recall was 0.3259, so the
  aggregate result does not imply stable per-seed recall.
- Full suite: **49 passed in 193.94 seconds**; output is in
  `logs/phase3R_full_pytest.txt`.
- Default synthetic CLI: **PASSED**, 900 windows, TP=269, FP=3, FN=1,
  recall=0.9963, FAR=0.0059; output is in
  `logs/phase3R_default_run_cli.txt`.
- Real CIC dataset: not present; real-data CLI check is **SKIPPED** with
  manual-download instructions; output is in `logs/phase3R_cic_cli.txt`.
- The final development sweep is recorded in
  `logs/phase3R_contamination_sweep.txt`; the contamination default was chosen
  only from development seeds 1–3. Smoke seeds 100–102 were reserved for the
  fixed acceptance check.
- Final test output includes the actual loader timing, smoke metrics, and
  runtime versions. Temporary logs and generated CSVs remain excluded from Git.

## Frozen defaults (Phase 3R)

- Warm-up: 120 windows.
- History: 300 windows.
- Periodic refit: every 30 post-warm-up windows.
- Contamination: 0.005.
- Model seed: 0.
- Default window: 1 second; CIC recommendation uses 60-second windows.

## Open issues and unmet criteria

- Real CIC metrics cannot be measured until TrafficLabelling files are supplied.
- The acceptance tests measure synthetic behavior only and are not evidence of
  real-world performance.
- Smoke-seed recall varies substantially: seed 101 had recall 0.3259 while
  seeds 100 and 102 had recall above 0.99. The required aggregate thresholds
  pass, but stronger per-seed robustness is not established.
- Poison protection cannot prevent false negatives from entering history.

## Phase 3R verdict

| Gate | Result | Evidence |
| --- | --- | --- |
| Frozen synthetic acceptance | PASS | Mean recall 0.7716; attack FAR 0.0052; normal-only FAR 0.0162 |
| Full regression suite | PASS | 49 passed |
| Default synthetic CLI | PASS | 900 windows; TP=269, FP=3, FN=1 |
| Real CIC CLI | SKIPPED | TrafficLabelling CSV is not present |

## Phase 4 (implemented; acceptance gates not all met)

- Added a PSI monitor with optional NumPy KS statistics, reference-quantile
  bins, constant-feature skipping, non-finite filtering, persistence,
  check-spacing, and cooldown. No dependency was added.
- Added drift-triggered refitting as an opt-in mode. The default remains
  periodic. The monitor observes every post-warm-up feature row; drift refits
  use the most recent observed rows, including flagged rows. This can adapt
  to a rejected new regime, but exposes refits to attack contamination.
- Added deterministic `stable`, `stable_attacks`, `drift`, and `drift_attacks`
  scenarios, plus development tuning and a held-out acceptance runner.
- Dev-only sweep (seeds 1–3, 32 candidates): **no candidate had zero events on
  both negative scenarios**. The selected fallback settings are PSI 0.50,
  four shifted features, persistence 2, 40 test windows, check every 20
  windows, cooldown 30. The median first-event delay on development drift
  runs was 59 windows. Full evidence is in
  `logs/phase4_drift_sweep.txt`; the selection criterion was not met.
- Focused Phase 4 tests: **8 passed** (`logs/phase4_unit.txt`), including a
  fixed seed-100 periodic compatibility fingerprint.
- Related synthesis/baseline regressions: **18 passed**
  (`logs/phase4_regression_targeted.txt`).
- Full suite: **57 passed in 127.50 seconds**
  (`logs/phase4_full_pytest.txt`).
- The periodic CLI on seed 100 matched the recorded Phase 3R metrics exactly:
  TP=269, FP=3, FN=1, with 26 periodic refits
  (`logs/phase4_periodic_compatibility.txt`).
- CLI smoke check on seed 100: generated the named normal-only drift scenario,
  then `ata.run --refit-trigger drift` reported 7 events, first at window 189,
  39-window delay, and 7 refits. It also flagged 160/520 post-warm-up windows
  (normal-flow FAR 0.3077)
  (`logs/phase4_cli_smoke100.txt`).
- Held-out acceptance (`logs/phase4_acceptance.txt`): drift fired on all three
  smoke `drift` runs, first at windows 189, 209, and 189; delays were 39, 59,
  and 39 windows, with 7 refits per run. `drift_attacks` produced 11 events
  and 11 refits per smoke seed, first at window 189 (39-window delay).
- Stable negative results **FAILED**: 5/10 `stable` runs drifted (50%); all
  10/10 `stable_attacks` runs drifted (100%). Combined false-drift rate was
  15/20 runs (75%). The acceptance command exited nonzero for these 15 rows.
  No smoke-seed values were used to retune the selected configuration.
- Informational preview only, not the Phase 5 result: on `drift_attacks`, the
  drift-triggered adaptive recall was 0.6259, 0.3852, and 0.4778 on seeds
  100–102, with FAR 0.0059, 0.0039, and 0.0490. Periodic and static recall
  were 1.0 on all three, but their FARs were approximately 0.90. This
  scenario is a severe distribution shift; these preview numbers do not
  establish that any baseline is preferable.
- Open issues remain: real CIC TrafficLabelling files are absent; synthetic
  evidence is not real-world performance evidence; Phase 3R per-seed recall
  variability (including seed 101) is unchanged; and the stable-traffic
  false-drift rate is unacceptable.

| Gate | Result | Evidence |
| --- | --- | --- |
| Drift detection on smoke `drift` | PASS | Events on 3/3; delay 39–59 windows |
| Quiet on stable traffic | FAIL | 5/10 stable and 10/10 stable-attacks runs had events |
| Dev tuning criterion | FAIL | No zero-false candidate among 32; fallback is explicitly recorded |
| Phase 4 unit tests | PASS | 8 passed |
| Related synthesis/baseline regressions | PASS | 18 passed |
| Periodic Phase 3R compatibility | PASS | Seed 100: TP=269, FP=3, FN=1; 26 refits |
| Full regression suite | PASS | 57 passed in 127.50 seconds |
| Real CIC validation | SKIPPED | TrafficLabelling files are not present |

## Phase 5 (evaluation harness; NO-GO)

All Phase 5 measurements below use synthetic traffic only and are not evidence
of real-world performance. Seeds 200–204 were reserved for reporting. The
synthetic datasets are generated once per scenario/seed and reused across the
static, periodic-adaptive, and drift-adaptive baselines.

- Added an ordered split helper, null-safe binary metrics, per-attack recall,
  and per-episode detection delay.
- `python -m pytest -q tests/test_phase5.py`: **6 passed**
  (`logs/phase5_unit.txt`).
- Final full regression suite: **63 passed in 122.34 seconds**
  (`logs/phase5_full_pytest.txt`).
- `python -m ata.evaluate --seeds 200 201 202 203 204 --out reports`:
  **NO-GO**. Full configuration, library versions, per-seed values, means,
  standard deviations, paired differences, poisoning results, diagnosis, and
  machine information are in `reports/metrics.json`.
- On `drift_attacks`, drift-adaptive FAR was **0.0255 ± 0.0185**, versus
  static **0.8592 ± 0.0922**, and won FAR on **5/5** seeds. Its recall was
  **0.4741 ± 0.1393**, versus static **0.9970 ± 0.0066**: a **52.29 point**
  mean recall deficit, failing the 5-point tolerance.
- On `stable_attacks`, drift-adaptive FAR was **0.0004 ± 0.0009** versus
  static **0.0086 ± 0.0051**, but recall was **0.2430 ± 0.0460** versus
  **0.8252 ± 0.0763**, failing the 5-point recall tolerance.
- Poisoning test on `stable_attacks`, seed 200: static recall dropped from
  0.8407 clean to 0.1667 at 5% warm-up poisoning and 0.3296 at 15%;
  periodic-adaptive dropped from 0.5519 to 0.1556 and 0.4074; drift-adaptive
  dropped from 0.3074 to 0.1296 and 0.2000. The non-monotonic 5%/15% results
  are reported as observed, not generalized. Poison protection still cannot
  stop false negatives from entering the buffer.
- Reports saved: `reports/metrics.json`, `reports/score_timeline.png`, and
  `reports/adaptive_vs_static.png`. Both plots were opened and visually
  checked for readable labels, legends, axes, and markers.
- CIC TrafficLabelling evaluation: **SKIPPED**; no CIC CSV is present in
  `data/`.

**Diagnosis:** drift refitting greatly reduces false alarms on `drift_attacks`
but sacrifices too much recall, and its recall is also poor on
`stable_attacks`. Likely contributors are false drift during attack episodes
and drift refits including recent flagged attack rows.

The research acceptance result is **FAIL / NO-GO**; it does not gate Phase 6
engineering. The drift protocol was restored to the documented Phase 4
behavior before this rerun: the monitor observes post-warm-up feature rows and
drift-triggered refits use recent stream history, including flagged rows.
This preserves the explicit attack-contamination tradeoff rather than
silently substituting an admitted-buffer-only experiment.

| Gate | Result | Evidence |
| --- | --- | --- |
| Chronological split and metric tests | PASS | 6 tests passed |
| Full regression suite | PASS | 63 passed in 122.34 seconds |
| `drift_attacks` lower mean FAR | PASS | 0.0255 vs 0.8592 |
| `drift_attacks` recall within 5 points | FAIL | 0.4741 vs 0.9970 |
| FAR win on at least 4/5 seeds | PASS | 5/5 |
| `stable_attacks` FAR within tolerance | PASS | 0.0004 vs 0.0086 |
| `stable_attacks` recall within 5 points | FAIL | 0.2430 vs 0.8252 |
| Overall research acceptance | **FAIL / NO-GO** | Drift-attacks recall and stable-attacks recall failed |

## Phase 6 (implemented; synthetic engineering validation)

- Added supervised XGBoost and Random Forest models using only the 11
  `FEATURE_COLUMNS`, with chronological 65% train / five-window gap / test
  partitions and shuffled attack order across five cycles. All four classes
  were present in train and test for each reported seed.
- The Python 3.11-compatible dependency pins include NumPy 2.3.5, XGBoost
  3.2.0, and SHAP 0.51.0. Pip's binary-wheel resolver verified the complete
  requirements set for both Python 3.11 and 3.14. Candidate/model selection
  was rechecked using development seeds 1–3 with
  these final pins. Both models tied at mean development macro F1 1.0000;
  Random Forest was selected by the documented interpretability tie-break.
  Candidate scores and parameters are in `configs/defaults.json`.
- Held-out seeds 200–204, per-class mean ± SD:

  | Model | Class | Precision | Recall | F1 |
  | --- | --- | ---: | ---: | ---: |
  | XGBoost | normal | 0.9988 ± 0.0027 | 1.0000 ± 0.0000 | 0.9994 ± 0.0014 |
  | XGBoost | portscan | 1.0000 ± 0.0000 | 0.9934 ± 0.0147 | 0.9967 ± 0.0075 |
  | XGBoost | ddos | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
  | XGBoost | exfiltration | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
  | Random Forest | normal | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
  | Random Forest | portscan | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
  | Random Forest | ddos | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
  | Random Forest | exfiltration | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |

  Macro F1 was 0.9990 ± 0.0022 for XGBoost and 1.0000 ± 0.0000 for Random
  Forest. Random Forest is the selected model because the dev-set macro F1
  tied and it is simpler to interpret; held-out metrics are not used for model
  selection. These very high scores reflect strongly separable synthetic
  templates and are not evidence of real-world performance.
- SHAP uses `TreeExplainer` on the development-selected Random Forest.
  Multiclass shape handling supports both class-array lists and 3D arrays; the
  portscan and DDoS driver sanity checks passed.
- Reports: `reports/classifier_metrics.json`, both model confusion matrices,
  `reports/shap_summary.png`, and `reports/alerts_sample.jsonl`. The sample
  alert records have exactly three JSON-serializable SHAP feature entries.
  The generated confusion, SHAP, score-timeline, and adaptive-comparison plots
  were opened and visually checked.
- Layer 1 research acceptance remains **FAIL / NO-GO** in the classifier
  report, separately from Phase 6 engineering validation. This implementation
  does not claim or imply that the Phase 5 gate passed.
- Focused Phase 6 tests: **8 passed** with three upstream SHAP deprecation
  warnings; `pip check` reports no broken requirements.
- Final full regression suite: **72 passed in 107.62 seconds** on Python 3.14.6
  (`python -m pytest -q`); only the same three upstream SHAP deprecation
  warnings were reported.
- CIC TrafficLabelling remains **SKIPPED** because no real CIC data is present.

## Phase 7 (streaming and serving; implemented)

- Added `FlowStream`, which reuses `ata.loader.load`, yields chronologically
  ordered canonical flows, supports speed-controlled replay and bounded
  batches, and surfaces loader validation errors. The existing loader still
  materializes the CSV before iteration; replay is incremental after loading,
  not a chunked large-file reader.
- Added `AtaPipeline` using the existing `WindowAggregator`, `AdaptiveBaseline`,
  optional Phase 6 classifier/SHAP functions, and alert schema. It validates a
  full batch before changing state, emits final partial windows on `flush()`,
  makes repeated flushes safe, and exposes bounded telemetry/counters.
- Added a FastAPI app factory and localhost service with health, flow ingestion,
  read-only score, flush, alert list/detail, and stats endpoints. SQLite alert
  storage uses parameterized SQL and JSON serialization; raw packet contents
  are not stored. The API has no authentication or TLS and must remain local.
- Added API replay and benchmark CLIs. The benchmark checks batch aggregation,
  pipeline, and API window counts against the same loaded CSV.
- Batch-versus-stream equivalence and API/store/replay/benchmark tests are in
  `tests/test_phase7.py`: **9 passed**. The combined focused Phase 7/8 command
  passed **14 tests**; it emitted two upstream Starlette deprecation warnings.
- Synthetic end-to-end replay used one generated `stable_attacks` CSV,
  seed 230, 29,908 flows and 190 windows. Replay processed all flows and
  windows, persisted 21 alerts, and completed in **29.17 s** at
  **1,025.4 flows/s** over local HTTP. No drift events or promotions occurred;
  the baseline recorded two refits.
- No trained model artifact is available. This run used Layer 1-only mode:
  all 21 alerts explicitly used `predicted_class: unknown`, a null class
  probability, and an empty SHAP feature list. This is an intentional
  unavailable-classifier state, not a fabricated explanation.
- `reports/phase7_replay_summary.json` and `reports/phase7_api_smoke.json`
  record the replay and endpoint checks. Health, stats, alert-list, and alert
  detail requests returned HTTP 200; the stored count and list total both equal
  21.
- `python -m ata.bench` measured aggregation at **87,485 flows/s**, the full
  pipeline at **1,188.8 flows/s**, and in-process FastAPI `TestClient`
  ingestion at **1,175.6 flows/s** on 29,908 synthetic flows. The API timing
  excludes Uvicorn socket/process overhead. Machine and configuration details
  are in `reports/throughput.json`; these are engineering timings, not
  production capacity claims.

## Phase 8 (interactive dashboard; implemented)

- Added a Streamlit dashboard for API health/counters, recent traffic and
  anomaly telemetry, alert filtering/detail/explanation, and the saved Phase 5
  and Phase 6 reports and plots. It keeps live API observations separate from
  historical evaluation data and displays the Layer 1 NO-GO.
- Browser inspection verified the system overview, traffic/score charts, alert
  explorer, a persisted Layer 1-only alert, and historical Phase 5/6 content.
  The dashboard root and Streamlit health endpoint returned HTTP 200; the API
  was healthy and reported 190 processed windows. Evidence is in
  `reports/phase8_dashboard_smoke.json`.
- The browser console emitted transient Vega "infinite extent" warnings while
  charts rendered; the final visible telemetry charts contained finite data.
  No blocking render failure was observed.
- The API is unauthenticated and has no TLS. Keep it on localhost; exposing it
  to an untrusted network is unsupported.
- Real CIC TrafficLabelling validation remains **SKIPPED** because the data is
  absent. All replay and throughput results above are synthetic and are not
  evidence of real-world detection performance. The Phase 5 research result
  remains **FAIL / NO-GO**, independent of Phase 7/8 engineering readiness.

| Gate | Result | Evidence |
| --- | --- | --- |
| Phase 7 focused tests | PASS | 9 passed |
| Phase 8 focused tests | PASS | Combined Phase 7/8 run: 14 passed |
| Batch/stream window equivalence | PASS | 190 windows from the same replay input |
| Synthetic API replay and SQLite persistence | PASS | 29,908 flows; 190 windows; 21 stored alerts |
| API endpoint smoke checks | PASS | Health, stats, alerts, and detail returned HTTP 200 |
| Throughput benchmark | PASS | Aggregation, pipeline, and in-process API counts agree |
| Dashboard browser smoke check | PASS | Live telemetry, alert detail, historical report and NO-GO visible |
| Layer 1 research acceptance | **FAIL / NO-GO** | Phase 5 recall criteria remain unmet |
| CIC TrafficLabelling validation | SKIPPED | No CIC CSV files are present |
| Final full regression suite | PASS | 86 passed in 119.61 seconds on Python 3.14.6 |

- Final verification: `python -m pytest -q` completed once with **86 passed**
  in **119.61 seconds**. Five upstream SHAP/Starlette/AnyIO deprecation
  warnings were reported; no test failed.
