# Phase 3R viva notes

### 1. Why does the CIC loader parse timestamps as a vector?

`pandas.to_datetime` can parse the full column in optimized native code.
Applying a Python function to every row has much higher overhead, which
matters on files with hundreds of thousands of flows.

### 2. Why support both DMY and MDY timestamp order?

CIC mirrors can serialize the same capture date with different conventions.
The loader tests both and uses the known CIC capture period to reject an
impossible interpretation instead of silently guessing.

### 3. Why retain both `label_raw` and `label`?

`label_raw` preserves the cleaned source name for auditing, while `label`
groups related attacks into a stable vocabulary for evaluation and later
supervised tasks.

### 4. How is a synthetic attack episode kept from leaking into a normal gap?

The schedule provides explicit start/end times. Attack flows are emitted
only inside their assigned interval; the intervals between episodes contain
normal-labeled flows only.

### 5. What does the adaptive baseline do during warm-up?

It collects the first `warmup` feature rows without using labels, fits the
scaler and Isolation Forest, and outputs NaN scores and false flags for
those warm-up rows.

### 6. How does the poison-protection rule work?

A post-warm-up row is scored first. Only unflagged rows are admitted to the
adaptive buffer. This blocks flagged windows, but false negatives can still
enter the buffer, so the rule cannot guarantee attack-free history.

### 7. Why can history be shorter than warm-up?

Initial fitting uses every warm-up row; later refits use only the most
recent eligible history rows. `min_fit_rows` separately controls the
minimum usable sample size for a later refit.

### 8. What is the distinction between the data seed and model seed?

The data seed controls reproducible generated traffic. The model seed
controls the Isolation Forest randomness. Keeping them separate makes
experiments reproducible without confusing traffic variation with model
variation.

### 9. How should synthetic metrics be interpreted?

They measure behavior on deterministic, simplified generated traffic.
They are regression and demonstration evidence, not estimates of
real-world detection performance; real CIC results must be reported
separately.

### 10. Why is `score_only` separate from `step`?

It allows a read-only score query to inspect the current model without
changing counters, buffer contents, or the sequence state.

### 11. Why use PSI as the primary drift statistic?

PSI compares the reference and recent distributions, not just their means.
Reference-quantile bins work for the heterogeneous scales in ATA's 11 window
features. Empty bins are smoothed, duplicate edges are removed, and constant
reference features are skipped.

### 12. Why require multiple shifted features and persistent checks?

One feature can vary naturally, especially during a short attack burst. A
minimum shifted-feature count and consecutive qualifying checks reduce the
chance that one noisy check triggers a model refit. Check spacing and cooldown
also limit repeated alarms and immediate refits.

### 13. Why can attacks look like drift, and how is that tested?

An attack can change flow counts, byte statistics, source concentration, or
port diversity just like a new normal regime. The `stable_attacks` scenario is
the hard negative: its attack schedule is present but drift is disabled. The
measured false-event count is reported rather than hidden by changing smoke
thresholds.

### 14. Does the monitor see flagged windows?

Yes. It sees every post-warm-up numeric feature row, irrespective of the
detector flag. Otherwise, a new regime rejected by the old model might never
reach the monitor or the buffer.

### 15. Can a drift refit contain flagged rows?

Yes. Drift-trigger refits use the latest `history` observed rows, including
flagged rows, while ordinary admission and periodic refits still use only
unflagged rows. This permits recovery from a rejected new regime but means a
sustained attack burst that satisfies the drift guards could contaminate that
fit.

### 16. What are the limits of this drift detector?

It is an unsupervised distribution-change alarm, not proof of benign
behavior. Gradual changes may be detected late; feature dependencies are not
modeled; PSI/KS thresholds depend on the synthetic data distribution; and
real CIC TrafficLabelling files are not present. Synthetic results cannot
establish real-world performance.

### 17. How does Layer 2 avoid temporal leakage?

Each synthetic run shuffles attack order across multiple cycles, then uses
chronological window indices for the 65% training split, a five-window gap,
and the later test segment. The split is never shuffled, and all four classes
must occur on both sides.

### 18. Are labels or time included in classifier features?

No. The classifier receives only the 11 numeric columns in
`FEATURE_COLUMNS`. Labels are used as training targets and test references;
window indices are used only to create and identify the ordered split.

### 19. Why compare XGBoost and Random Forest with class weights?

The four target classes are not equally frequent. XGBoost receives balanced
per-row training weights and Random Forest uses `class_weight="balanced"`.
Hyperparameters and the selected model are chosen using development seeds
1–3, not the reported seeds.

### 20. How are multiclass SHAP values attached to alerts?

`TreeExplainer` returns a class-specific array list in some versions and a
three-dimensional array in others. The small adapter normalizes both to
samples-by-features, then selects the predicted class for each alert and sorts
features by absolute SHAP value.

### 21. What does the Phase 5 NO-GO mean for Layer 2?

It means the frozen Layer 1 research criteria were not met. It remains a
NO-GO in reports, but it does not prevent independent classifier engineering
and tests. Phase 6 status describes implementation readiness, not Layer 1
acceptance or real-world detection quality.

### 22. How should perfect classifier results be interpreted?

The reported five-seed synthetic test scored 1.0000 per-class metrics for
both tree models. Repeated synthetic attack templates are strongly separable;
this result validates the pipeline on its generator, not generalization to
real network traffic. CIC evaluation is SKIPPED until TrafficLabelling files
are supplied.
