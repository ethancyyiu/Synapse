# KC population information in synthetic recovery telemetry

**, reward-free study.** The [separate prospective protocol](practical-transfer-kc-information-protocol.md) was locked before the two new neural streams were run. It reuses 96 archived calibration recordings and compares actual measured VPN, KC, and 42 individual-MBON spikes on **96 new observations each** for independent neural seeds **19202 and 19203**. The previous [practical-transfer redesign sensory gate](practical-transfer-redesign-report.md) remains a valid failure; this study neither alters it nor trains a fly.

## Measurement and reconstruction

The synthetic observable telemetry is `(grasp offset, slip/rotation, obstruction resistance)`, each in `[0,1]`. The unchanged encoder drives the three existing VPN lanes at `40 + 110 × telemetry` Hz from 0–240 ms. Each probe fully resets its neural state, resets ordinary counts at 50 ms, then measures **the same 50–240 ms window** for all VPN lanes, 2,045 KCs, and 42 MBONs. The artifact stores the actual counts, KC identities, requested drives, telemetry, split/seed/key, window, and reset rule. There were no rewards, actions, controller updates, or synaptic changes.

All four analysis-only representations use a calibration-only, intercept-unpenalized ridge-4 linear reconstruction of the three *observable* telemetry dimensions: three measured VPN lane counts; the full KC population; the earlier 42 individual MBON counts; and a secondary 42-bucket, label-free KC summary. Feature scaling and the bucket assignment were fixed before validation. The MBON procedure reproduced every archived 19102 prediction to maximum absolute error **5.6×10⁻¹⁶**. The copied historical calibration rows also contain unused synthetic outcome fields from their original artifact; neither fitting nor validation reads those fields. No decoder was installed in Fly A or B.

## Independent validation

Entries below are `R² / normalized MAE`; MAE is normalized by the full sensor range of one. All rows use the same 96 observations within a seed. Values were fitted on calibration **19101 only**.

| Neural stream | Representation | Offset | Slip | Resistance |
| --- | --- | ---: | ---: | ---: |
| 19202 | VPN spikes | .926 / .066 | .935 / .058 | .927 / .062 |
| 19202 | **Full KC** | **.710 / .133** | **.709 / .127** | **.595 / .158** |
| 19202 | 42 MBONs | .308 / .210 | .223 / .212 | .373 / .200 |
| 19202 | 42 fixed KC buckets | .626 / .152 | .570 / .155 | .516 / .174 |
| 19203 | VPN spikes | .923 / .063 | .929 / .061 | .932 / .066 |
| 19203 | **Full KC** | **.717 / .128** | **.608 / .150** | **.632 / .164** |
| 19203 | 42 MBONs | .368 / .193 | .202 / .220 | .387 / .203 |
| 19203 | 42 fixed KC buckets | .661 / .142 | .497 / .171 | .564 / .179 |

Signed `predicted − actual` biases for offset/slip/resistance were **−.018/+.026/+.022** for KC 19202 and **−.020/+.021/+.026** for KC 19203. The MBON biases were **−.023/+.038/+.014** and **−.009/+.020/+.034**, respectively. VPN biases stayed between −.018 and +.008. The full trial predictions in the artifact retain unrounded values.

The intended VPN count-to-drive correlations were **.988/.985/.982** for 19202 and **.986/.983/.987** for 19203. Full KC recruitment averaged **31.8** and **32.4** active neurons per observation (medians **29** and **28.5**), with nonzero KC activity on **96/96** observations in each stream. Total KC recruitment and total MBON spike count correlated **.925** and **.941** across observations. These correlations describe co-variation; they do not prove which synapses preserve or discard information.

## Locked gates and interpretation

| Predeclared gate | Result |
| --- | --- |
| Valid counts, 12 observations per octant, matched window, unchanged weights, keyed repeat | **Pass** on both streams. |
| VPN count-to-drive correlation ≥.80 and VPN reconstruction R² ≥.70, every dimension/stream | **Pass**. |
| Full KC R² ≥.40 and normalized MAE ≤.20, every dimension/stream | **Pass**. The lowest KC R² was .595; the largest KC MAE was .164. |
| KC–MBON R² gain ≥.10 on slip and resistance, offset no worse than −.05, both streams | **Pass**. Slip gains were **+.485/+ .406**; resistance **+.222/+ .245**; offset **+.402/+ .350**. |

Thus, **under this fixed linear reconstruction**, KC population activity retained useful information about all three continuous diagnostics that the previously tested 42-MBON procedure did not recover reliably. The result was consistent across both independent streams. The 42-bucket KC control also outperformed the 42-MBON comparator on all three dimensions in both streams, so a higher *feature count alone* does not account for the descriptive difference. The bucket transform and anatomical layer differ as well, so this is not a controlled causal proof of why MBON reconstruction was worse.

The upstream VPN reference was strongest. Full KC did not perfectly reconstruct telemetry, and a poor linear fit cannot prove information is absent from MBON spikes or identify a particular loss-making connection. The validation has only two new neural streams, 96 cases each, and an analysis-only decoder. It does **not** establish that a reward-trained KC controller will learn recovery actions, that teacher advice will help a student, or that the synthetic assumptions describe real warehouse robots.

**Decision:** The locked KC information gates passed, so one *separately locked* KC-input independent Fly A learning feasibility experiment is scientifically justified. Its controller, budget, and gates must be fixed before any reward training. No such training or teacher–student experiment was run here.

## Artifact and reproduction

- [Write-once trial result](../data/practical_transfer_kc_information.json), with protocol SHA-256 `70c6ecf2b8bc7d5958a21e8d68412cafdb9cb3d7a690b6eb3295369d91523503`, 192 new trial rows, actual counts, predictions, and all gates.
- [Earlier failed MBON sensory result](../data/practical_transfer_redesign_initial_sensory.json) remains unchanged.

From the repository root: `node --check tools/check_practical_transfer_kc_information.mjs`; `node tools/check_practical_transfer_kc_information.mjs preflight`. To reproduce the neural simulation, use a **copy without the new write-once artifact** and run `node tools/check_practical_transfer_kc_information.mjs`; exclusive creation prevents overwriting this result. The saved artifact's protocol hash, row counts, octant balance, window, finite predictions, and gate values were checked read-only after creation. No Git operation, RPS change, demo change, or model update occurred.

The first execution completed both locked streams but the renamed repository's stale workspace write boundary rejected artifact creation (`EPERM`); it produced no result file. The same code, protocol, seeds, and streams were rerun once with approved filesystem access to preserve the write-once result. No setting or gate changed between executions.
