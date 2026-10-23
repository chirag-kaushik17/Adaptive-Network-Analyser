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
