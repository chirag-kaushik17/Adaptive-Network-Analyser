# Phase 6 blocker audit

Audit date: 2026-10-24

## Workspace and implementation status

- Project root: `D:\Open Source Contribution\Adaptive-Network-Analyser-copilot-ata-work`.
- The directory is not a Git repository (`git status --short --branch` exits 128 with “not a git repository”).
- Phases 0–3R are present in the loader, synthetic generator, feature, and baseline modules. Phase 4 drift monitoring and Phase 5 evaluation are implemented.
- No classifier, classifier CLI, explainability adapter, alert module, Phase 6 test file, or classifier report exists in the audited source tree.
- `data/` contains only its README; no CIC TrafficLabelling data is available.

## Reproduction and exact frozen-gate results

Command run against the required held-out seeds:

```powershell
python -m ata.evaluate --seeds 200 201 202 203 204 --out reports
```

Observed verdict: **NO-GO**. This is the evaluated research-acceptance result,
not a failed software test and not a missing-data status.

The frozen criteria are in `tests/thresholds.py`. `src/ata/evaluate.py::_go_no_go`
compares aggregate drift-adaptive FAR and recall with static, counts per-seed
FAR wins for `drift_attacks`, and applies FAR/recall tolerances for
`stable_attacks`.

| Condition | Expected | Observed | Result |
| --- | --- | --- | --- |
| `drift_attacks` mean FAR | Adaptive lower than static (0.8592) | Adaptive 0.9000 | FAIL |
| `drift_attacks` mean recall | No more than 0.05 below static (0.9970) | Adaptive 1.0000 | PASS |
| `drift_attacks` FAR wins | At least 4/5 seeds | 1/5 | FAIL |
| `stable_attacks` mean FAR | No more than 0.02 above static (0.0086) | Adaptive 0.0004 | PASS |
| `stable_attacks` mean recall | No more than 0.05 below static (0.8252) | Adaptive 0.5244 | FAIL |

For seed 200, drift-attacks FAR was 0.9078 static and 0.9196 adaptive; recall
was 1.0000 for both. Stable-attacks recall was 0.8407 static and 0.4407
adaptive. For seed 204, drift-attacks FAR was 0.9176 static and 0.9255
adaptive; recall was 1.0000 for both. Stable-attacks recall was 0.9000 static
and 0.4444 adaptive. These are two entries in the five-seed evaluation, not
independent per-seed pass/fail gates.

The values above are from `reports/metrics.json` after the reproduced command.
Older Phase 5 prose and diagnosis files describe a different measured result;
they are stale relative to this run and must be synchronized with the current
report rather than treated as the latest evidence.

## Cause assessment

1. **Research acceptance failure:** Confirmed. The current held-out aggregate
   values violate three frozen conditions. The failure must remain NO-GO unless
   a candidate chosen only on development data later passes a new held-out
   run. Seeds 200–204 and the thresholds must not be excluded or changed.
2. **Software/evaluation mismatch:** Found. The current
   `AdaptiveBaseline._refit_on_drift` fits from `buffer`, although Phase 4
   documentation describes the opt-in drift path as fitting the recent stream
   history (including flagged rows). `evaluate._new_model("adaptive_drift")`
   also selects `policy="quarantine"` while the documented Phase 5 comparison
   uses the standard adaptive baseline. This means the present run is not
   using the previously documented drift-refit protocol; restore the documented
   implementation before treating the new measurements as the final Phase 4/5
   comparison.
3. **Phase dependency/orchestration:** No code in the repository invokes the
   classifier conditionally on the research verdict because the classifier
   does not exist yet. README/STATUS describe the GO gate as a phase-progression
   policy. Engineering implementation and its valid tests can be separated
   from research acceptance without changing the thresholds or reporting a
   false PASS.
4. **Missing prerequisites:** No CIC data is present, but Phase 6 can run on
   the prescribed synthetic data. The feature schema, chronological split
   helper, scenario generator, and fixed evaluation seeds already exist.
   Classifier dependencies and the Phase 6 implementation/tests are absent.

## Safe resolution

- Restore the documented Phase 4 drift-refit path and default evaluation setup;
  preserve all frozen thresholds and label the re-evaluated result as PASS,
  FAIL, BLOCKED, or NOT_RUN without substituting seeds.
- Implement and test Phase 6 independently on synthetic data. Record Phase 6
  engineering-test status separately from Layer 1 research acceptance.
- Keep CIC evaluation marked SKIPPED until real TrafficLabelling files are
  supplied. Do not claim synthetic evidence is real-world performance.

## Post-audit resolution

The documented Phase 4 drift-refit protocol was restored and the required
held-out evaluation was rerun. It remains **FAIL / NO-GO**: drift-attacks
adaptive FAR is 0.0255 versus static 0.8592 and wins on 5/5 seeds, but recall
is 0.4741 versus 0.9970; stable-attacks recall is 0.2430 versus 0.8252.
Neither failure was hidden or used to alter the frozen criteria.

Phase 6 was implemented independently. The classifier CLI uses the ordered
split and synthetic data; its implementation and tests do not mark the Layer
1 research gate as accepted. The final test status and classifier metrics are
recorded in the project status document and `reports/classifier_metrics.json`.

## Final verification

After restoring the documented drift-refit protocol, the required held-out
command was rerun:

```powershell
python -m ata.evaluate --seeds 200 201 202 203 204 --out reports
```

The final Phase 5 result is still **FAIL / NO-GO**. On `drift_attacks`,
drift-adaptive FAR is 0.0255 ± 0.0185 versus static 0.8592 ± 0.0922 and wins
on 5/5 seeds; recall is 0.4741 ± 0.1393 versus 0.9970 ± 0.0066, outside the
5-point tolerance. On `stable_attacks`, drift-adaptive FAR is 0.0004 ± 0.0009
versus 0.0086 ± 0.0051, but recall is 0.2430 ± 0.0460 versus 0.8252 ± 0.0763.
Thus the two recall conditions fail; the JSON report records both conditions
and the NO-GO verdict without altering the thresholds or seeds.

Phase 6 was run on the same held-out seeds after pinning XGBoost 3.2.0 and
SHAP 0.51.0; the NumPy pin was adjusted to 2.3.5 because the previous 2.5.4
pin had no Python 3.11 wheel. The complete requirements set resolves to binary
wheels for Python 3.11 and 3.14. The development-seed model comparison was
rerun with the final XGBoost/SHAP pins (seeds 1–3); both models tied at mean
macro F1 1.0000, and Random Forest was selected by the recorded
interpretability tie-break. Phase 6 focused tests passed (8 tests). Its
engineering result remains separate from the Layer 1 research failure. All
classifier scores are synthetic and not evidence of real-world performance;
CIC TrafficLabelling remains SKIPPED.
