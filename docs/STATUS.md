# ATA phase status

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

No Phase 4 work was started. Phase 4 should not begin until real CIC
TrafficLabelling files are available and the per-seed synthetic recall
variability has been reviewed.
