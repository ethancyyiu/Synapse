# Prospective KC information study (locked before new validation)

**. Reward-free, separate from the failed practical-transfer redesign.** This study asks whether the recorded KC population supports linear reconstruction of observable grasp offset, slip, and resistance better than the already tested 42 individual MBON counts. It does not test reward learning or teacher–student transfer. The previous information gate and its failed result remain unchanged.

## Fixed observations and neural procedure

Reuse the existing 96 calibration probes from neural seed **19101**, case seed **20261109115**, in `data/practical_transfer_redesign_initial_sensory.json`. They were collected before this study and include real KC counts. Collect exactly **96 new probes per validation stream**, using independent neural seeds **19202** and **19203** and case seeds **20261109202** and **20261109203**. Each stream has 12 independently sampled continuous triples in each of eight binary half-range octants, with octants shuffled by seeded Fisher–Yates using the repository RNG. In the shuffled order, draw three independent within-octant uniforms per case in `(offset,slip,resistance)` order; draw no potential outcomes. No action, reward, or hidden cause is an input. The two new streams are the primary independent evaluation; the archived 19102 stream is a historical comparator only and will not decide gates.

For each probe, create the original real-wiring network for that stream, call `resetState()`, set the keyed RNG to `neuralSeed × 10^7 + 10^6 + caseIndex × 1000 + 1`, and stimulate `laneVpn[j].slice(2)` simultaneously at `40 + 110 × telemetry[j]` Hz for `t=0..239` ms. Call `resetCounts()` at `t=50` ms, clear drives at 240 ms, and read the 50–240 ms counts of all VPN lanes, all **2,045 KCs** in `net.idx.kc` order, and all **42 MBONs** in `net.idx.mbon` order. Thus all three representations come from the same observation and decision window. Freeze all synaptic multipliers at one. Require distinct effective 32-bit RNG keys across new probes and disjoint case and neural seeds. No simulator or sensory-encoder change is allowed.

The previous artifact stores active KC `[index,count]` pairs; absent KC indices mean zero count. Save new records in the same sparse form, plus actual drives, VPN counts, 42 MBON counts, window, reset metadata, split, seeds, and telemetry. Save no synthesized spike data.

## Fixed reconstruction

Use **one ordinary ridge linear reconstruction per observable dimension** for every representation: minimize squared calibration error plus `4 × ||slope||²`, with unpenalized intercept. Solve in the calibration sample/dual space where needed. Fit only on the 96 archived calibration rows, then freeze coefficients and scaling before opening either new validation outcome. Predict each telemetry dimension independently, without using the other dimensions' labels as inputs.

- **VPN upstream reference:** three measured lane spike counts, transformed by `log1p`, then calibration mean subtraction and sample-SD scaling (SD floor 0.1), divided by `sqrt(3)`.
- **KC primary:** the full, predeclared 2,045-dimensional `log1p` spike-count vector in simulator KC index order, centered by its calibration mean and divided by one global calibration RMS Euclidean radius (floor 0.1). No KC is chosen by a label or validation result. This population normalization prevents a larger feature count alone from trivially changing the ridge scale.
- **MBON downstream comparator:** the original 42-dimensional `log1p` individual-count vector with the archived calibration means and sample SDs (floor 0.1), divided by `sqrt(42)`, exactly as in the failed sensory experiment. Refit only the same ridge-4 calibration model and verify its archived 19102 predictions reproduce the saved reconstruction before evaluating new streams.
- **Dimension-matched KC secondary control:** preassign the 2,045 KC indices to 42 buckets by position modulo 42, sum actual counts within each bucket, apply `log1p`, then calibration mean/sample-SD scaling (floor 0.1) and divide by `sqrt(42)`. Its assignments are fixed without outcomes or labels. This lossy control tests whether any full-KC advantage persists with 42 features; a failure cannot prove the difference is solely dimensionality.

No reconstruction output enters a fly. There is no feature selection, nonlinear decoder, validation refit, or parameter search.

## Metrics and gates

For **each of the two new streams and each telemetry dimension**, report `R² = 1 − SSE/SST`, mean absolute error normalized by the full `[0,1]` sensor range (numerically the MAE), and signed prediction bias `mean(predicted − actual)`. Report between-stream variation, mean/median/nonzero KC recruitment, VPN count-to-requested-drive Pearson correlations, and descriptive KC/MBON response relationships. Keep raw predictions and trial rows.

**Recording validity gate:** 96 rows and 12 per octant per stream; matching input/window across layers; all counts nonnegative integers; 2,045 KC and 42 MBON identities in fixed order; all neural multipliers unchanged; exact keyed repeat of the first probe in each stream; no key collision. If it fails, stop before decoding.

**Upstream reference gate:** own-lane VPN count-versus-drive Pearson correlation at least **0.80** in every dimension and stream, and the fixed VPN ridge reconstruction `R² ≥ 0.70` per dimension and stream. If it fails, report the failure and do not attribute a downstream difference to KC or MBON.

**KC absolute information gate:** full-KC reconstruction must reach `R² ≥ 0.40` and normalized MAE `≤ 0.20` on **offset, slip, and resistance separately in both streams**, matching the old MBON information floors. An improvement on only one dimension is not a pass.

**KC-over-MBON gate:** additionally require full KC `R²` to exceed same-row 42-MBON `R²` by at least **0.10** on **both previously failed dimensions, slip and resistance, in both streams**, and be no worse than **−0.05** on offset in either stream. This establishes a consistent method-specific improvement, not an irrecoverable biological information loss. Report the 42-bucket KC control without using it to rescue or overturn the primary gates. Both KC gates are needed to justify a later, separately locked KC-input learning experiment.

Stop after this one reward-free validation, whether it passes or fails. Do not train Fly A or Fly B, alter the earlier gate, change decoding after results, run more seeds, or modify RPS or the demo. Write one new exclusive-creation JSON artifact containing this protocol's SHA-256, full rows and results; document all gates and failures.
