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
