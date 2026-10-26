# ATA design decisions

## Phase 3R decisions

### CIC date-order inference

Automatic mode parses timestamps as both DMY and MDY. A candidate is accepted
only when all parseable values fall within the documented CIC capture window,
2017-07-02 through 2017-07-08. A single valid candidate is chosen; two
identical date sequences choose DMY; two different valid sequences require an
explicit user choice. `--allow-any-dates` supports out-of-range, unambiguous
data without silently guessing between two plausible dates. For one date-file,
relative seconds remain based on that file's earliest timestamp.

### Label families

CIC labels are cleaned to lowercase alphanumeric text in `ata.labels`. The
loader preserves that cleaned token as `label_raw`, and maps known CIC attack
families to a compact family label. Unknown family names are warned and mapped
to `other`. Synthetic `exfiltration` is intentionally not a CIC family.

### Synthetic schedule

Attack times are scheduled as fixed, non-overlapping episodes with normal-only
gaps; attack generation is bounded to those intervals. Default schedule:
120 s clean warm-up + 9 x 30 s attack + 8 x 40 s gap + 190 s normal tail =
900 s. Portscan, DDoS, and exfiltration emit 50, 60, and 5 attack flows per
second respectively. Each attack type therefore has at least 25 windows at
the default 1-second window. The final scenario end is stored in the sidecar.
The sidecar is used only to warn about an onset before warm-up; it is never
passed to a model.

Development-only diagnostics showed mean portscan recall of 0.0481 at the
original rate of 10 flows per second, versus 0.8148 for DDoS and 0.9556 for
exfiltration. The portscan episode rate was raised to 50 one-source SYN-only
probes per second. This makes the synthetic scanning pattern more distinct; it
does not consult labels during detector inference. The contamination sweep and
fixed smoke acceptance suite below were rerun against this calibration.

### Baseline lifecycle

Periodic refits require `min_fit_rows` eligible rows, defaulting to
`min(warmup, history)`. This permits a history shorter than the initial
warm-up while still fitting the initial detector on all warm-up rows. If
eligible rows are insufficient for a periodic refit, it is skipped and counted.
The static baseline retains only its initial warm-up rows. `score_only` has no
state-changing behavior.

### Contamination selection

The dev-only sweep used seeds 1, 2, 3, model seed 0, scenarios A (stable plus
the default attacks) and B (stable normal-only), warm-up 120, history 300,
refit interval 30, and candidate contamination values 0.005, 0.01, 0.02,
0.05. Selection is the lowest mean false-alarm rate among candidates whose
scenario-A mean recall is strictly greater than 60%; ties choose the smaller
contamination.

| Contamination | Mean recall A | Mean FAR A+B |
| ---: | ---: | ---: |
| 0.005 | 0.6062 | 0.0070 |
| 0.010 | 0.6852 | 0.0161 |
| 0.020 | 0.7383 | 0.0271 |
| 0.050 | 0.9136 | 0.0868 |

Chosen value: **0.005**, written to `configs/defaults.json`. Full precision
values by development seed and all candidates are recorded in that JSON file.
This tuning evidence is for development only; the independent smoke seed
acceptance run verifies fixed regression thresholds and does not select a
configuration.

The sweep took 201.34 seconds on Windows 11, Intel Core Ultra 5 125H
(14 physical / 18 logical cores), Python 3.13.5, NumPy 2.5.4, pandas 3.0.6,
scikit-learn 1.9.1.

## Phase 4 decisions

### Drift statistics and safeguards

PSI is the primary statistic. It uses ten reference-quantile bins, removes
duplicate quantile edges, and applies epsilon smoothing to empty bins. A
reference feature with zero variance is skipped and reports no statistic;
non-finite values are excluded per feature. The optional KS mode computes the
two-sample empirical-CDF distance in NumPy, so Phase 4 adds no dependency.
`psi_threshold` is the comparison threshold for either selected statistic.

Each check compares the most recent configured number of observed feature
rows with the current fit reference. A feature counts only when its statistic
exceeds the threshold; at least `min_features` must exceed it at
`persistence` consecutive checks. Checks are separated by `check_every` rows.
The monitor enters cooldown after an event or after a model refit, including
periodic refits. Tuned values and their development-only evidence are recorded
in `configs/defaults.json` and `logs/phase4_drift_sweep.txt`.

### Monitor input and drift refit set

The monitor sees every post-warm-up feature row, including rows flagged by
Layer 1. This avoids the deadlock where the old detector flags every
new-regime window, the buffer rejects them, and an admitted-only monitor can
never accumulate evidence. It receives only the numeric feature vector:
ground-truth labels and sidecar metadata are not monitor inputs.

On a drift event, drift-trigger mode fits on the most recent `history`
observed rows, including flagged rows. This is an explicit exception to the
normal buffer-admission rule, limited to drift-trigger refits. It lets a
detector adapt when the old model rejects the new regime. The tradeoff is
that a sustained attack burst could be included if it also meets the
persistence and multi-feature drift guards; `stable_attacks` runs measure
that failure mode. Periodic mode continues to refit on the unflagged buffer
only and retains Phase 3R score, flag, and refit behavior.
The seed-100 default periodic CLI check retained the recorded Phase 3R
fingerprint: TP=269, FP=3, FN=1, and 26 periodic refits
(`logs/phase4_periodic_compatibility.txt`).

### Development tuning

The selector is fixed before held-out runs: require zero events on every
development `stable` and `stable_attacks` run, then minimize median detection
delay on `drift`. Only data seeds 1, 2, and 3 are used. The complete candidate
table, including per-seed event positions and false-run counts, is saved in
`logs/phase4_drift_sweep.txt`; the chosen parameter values and selection
verdict are recorded in the new `drift` configuration section. If no
zero-false candidate exists, the log reports that fact and records the
minimum measured false-run candidate without claiming the criterion passed.

The 32-candidate sweep found no zero-false candidate. No events occurred on
the three `stable` runs; every `stable_attacks` development run had at least
one event (3/3, a 100% per-run false-drift rate on this hard negative). The
fallback selected PSI threshold **0.50**, minimum **4** shifted features,
persistence **2**, **40** test windows, and a check interval of **20**
windows. The median first-event delay on `drift` was **59 windows**. These
values are a measured compromise, not a pass of the zero-false selection
criterion; attack-induced false drift remains an open failure mode.

| Candidate | Stable events | Stable-attack event runs | Median drift delay |
| --- | ---: | ---: | ---: |
| PSI 0.25; 3 features; persistence 2; 40 test; check 10 | 3/3 | 3/3 | 19 windows |
| PSI 0.25; 4 features; persistence 3; 60 test; check 20 | 1/3 | 3/3 | 69 windows |
| **PSI 0.50; 4 features; persistence 2; 40 test; check 20 (selected)** | **0/3** | **3/3** | **59 windows** |
| PSI 0.50; 4 features; persistence 3; 60 test; check 20 | 0/3 | 3/3 | 89 windows |

The complete per-seed result for every candidate is in
`logs/phase4_drift_sweep.txt`. The chosen candidate minimizes false-run count
first because the required zero-false set was empty; it does not minimize
detection delay across the full grid.

## Detection simplification decisions

### Static Layer 1 default and opt-in adaptive/drift refits

In Phase 5 research evaluation, adaptive periodic refitting consistently lowered
attack recall compared to the static baseline (and resulted in a Phase 5 NO-GO
verdict). When attacks or contaminated traffic occur, false negatives entering
the adaptation buffer can poison subsequent refits, and drift refits can adapt
away valid anomaly detections.

Consequently, the operational design simplifies Layer 1:
- `AtaPipeline`, API configuration (`create_app`), `serve.py`, and the CLI
  (`run.py`) default to a `static` baseline (fitted once on warm-up and never
  refit).
- Periodic adaptive refits and drift-triggered refits remain available as an
  explicit opt-in experiment via `--layer1-mode static|adaptive|drift`.
- `evaluate.py` (Phase 5) retains the ability to evaluate all three modes
  (`static`, `adaptive_periodic`, `adaptive_drift`) without changes, and the
  recorded Phase 5 research NO-GO verdict is preserved unchanged.

### Layer 2 evaluation on every window and alert rules

To maximize recall and provide robust defense-in-depth:
- After warm-up, Layer 2 evaluates the classifier on every window, rather than
  gating classifier inference only on Layer 1 anomaly flags.
- If Layer 2 predicts any attack class (other than `"normal"`), an alert is
  emitted. The alert records `layer1_flagged` (`True` or `False`) to indicate
  whether Layer 1 also detected the anomaly.
- If Layer 1 flags a window that Layer 2 calls `"normal"`, an alert is emitted
  with `predicted_class="unknown_anomaly"` and null SHAP features (`top_features=[]`),
  ensuring novel or unmodeled anomalies are not silently dropped.
- If no classifier is loaded, the pipeline behaves as before, generating alerts
  only when Layer 1 flags a window (`predicted_class="unknown"`).

### Dead code removal

- Deleted unused quarantine and promotion policies (`policy`, `promote_after`,
  `_maybe_promote`, and quarantine state variables) from `BaseBaseline` and
  `AdaptiveBaseline`.
- Deleted `LegacyShiftMonitor` from `ata.drift`.
- Removed the unused `"Promotions"` metric and always-empty `"severity"` column
  from API stats and the dashboard.
- Removed duplicate flow batch validation in the API's `/flows` endpoint.

