# Practical-transfer redesign feasibility: stopped at sensory gate

**.** The [prospectively locked protocol](practical-transfer-redesign-protocol.md) was executed once for initial identity **19102**, in its required order. The raw-observable learner passed. The reward-free 42-individual-MBON sensory representation **failed** its independent information gate. The runner stopped before neural Fly A reward training, reserved identity 19103, DN receiving, or any student study. This does not revise the earlier failed recovery studies or the audited Strategy A/B RPS results.

## Implementation and preflight

The isolated [controller](../tools/practical_transfer_redesign_controller.mjs) implements exactly the locked per-action ridge posterior approximation: initial inverse `(4I)⁻¹`, mean `xᵀA⁻¹b`, uncertainty `0.7 sqrt(xᵀA⁻¹x)`, normal-sampled action during free training, selected-arm Sherman–Morrison and reward update, and mean-only frozen action. The [runner](../tools/check_practical_transfer_redesign.mjs) implements the continuous synthetic telemetry and Bernoulli outcomes, 288 forced-coverage cases plus 192 free cases, and independent 120-case frozen evaluation. The controller receives only the four raw features or, if reached, the 43-dimensional vector of measured MBON counts; the probability formula, potential outcomes, and analytical best action are held in the case/evaluation layer and never passed to its decision or update. Only the chosen outcome supplies reward. Neural simulation uses the existing real wiring and unchanged dynamics; no RPS or demo file was modified.

`node --check` passed for the controller, runner, and focused test. `node tools/test_practical_transfer_redesign_controller.mjs` passed posterior, uncertainty, selected-arm update, forced-choice RNG isolation, deterministic replay, and frozen-choice checks. A reward-free `preflight` verified **480** training cases, **288** forced choices, **12 selected cases per octant/action**, **120** held-out cases, both 96-probe sensory tapes, and the original analytical grid result **28/53/44** optimal-action counts with oracle–constant gap **0.1348**. The archived protocol SHA-256 embedded in both new artifacts is `8ddb20f95c8b2cd934c4e6d65260252584ff81b749bd27dfbaa2a81330181493`.

## Stage 1: raw-observable learning passed

The same Bayesian ridge equation used the legitimate observable `(offset, slip, obstruction)` input. The predetermined case/outcome tape and controller seed were used once. Frozen evaluations used independent controller copies, disabled posterior sampling and updates, and did not change parameters. Values below are from the saved trial rows; expected net reward uses the known synthetic outcome probabilities **for scoring only**.

| Training attempts | Frozen expected net reward | Realized recoveries / 120 | Frozen A0/A1/A2 choices | Frozen delay seconds / escalations |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.3912 | 71/120 | 120/0/0 | 240 / 0 |
| 288 | 0.6366 | 107/120 | 17/48/55 | 611 / 55 |
| 480 | **0.6412** | **108/120** | **31/38/51** | 571 / 51 |

On this fixed held-out tape, the best constant action has mean expected reward **0.5044**, and the observation-aware oracle **0.6477**. The raw learner gained **+0.2500** over its own checkpoint 0 and **+0.1368** over the best constant, closing **95.5%** of the oracle–constant gap. Each frozen action was selected at least 12 times. It passed every predeclared raw gate: ≥0.05 gain, ≥0.05 over constant, ≥50% gap closure, and ≥12/120 per action. This establishes independent learning from raw observable telemetry on the stipulated simulator for **one seed**, not neural learning or real robotic utility.

### Cost of the fixed coverage phase

| Training phase | Chosen A0/A1/A2 | Successes / attempts | Failed attempts | Delay seconds | Escalations | Mean realized net reward |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Forced coverage, attempts 1–288 | 96/96/96 | 205/288 | 83 | 1,248 | 96 | 0.4332 |
| Reward-guided free phase, attempts 289–480 | 41/77/74 | 171/192 | 21 | 908 | 74 | 0.6665 |

Coverage guaranteed **12 own selected outcomes for each action in each of eight observable regions**, including 96 costly assisted clears. The 83 failures, 1,248 seconds of modeled delay, and 96 escalations are the *observed phase totals*. They are not a causal incremental cost of exploration: the controller changed over time, and no matched non-forced training arm was in this protocol. The synthetic delay is a cost assumption, not measured wall-clock or robot time.

## Stage 2: neural sensory gate failed

The fixed reward-free calibration identity **19101** supplied 96 neural probes to normalize `log1p` counts of all 42 individual action MBONs and fit one **analysis-only** ridge reconstruction of the three *observable* telemetry values. Independent identity **19102** supplied 96 new training-style probes. Each reset the network, drove the three existing VPN lanes simultaneously at the locked `40+110×telemetry` rates over 0–240 ms, reset counts at 50 ms, and recorded actual VPN counts, active KC identities, individual MBON counts, and three pooled rates. A repeated probe with the same key reproduced its complete saved record. KC→MBON multipliers stayed one. The reconstruction was never installed as a controller or trained on validation data.

| Observable dimension | Held-out R² (required ≥0.40) | Mean absolute error (required ≤0.20) | VPN count versus requested-drive correlation (required ≥0.80) |
| --- | ---: | ---: | ---: |
| Grasp offset | **0.432** | **0.194** | **0.988** |
| Slip / rotation | **0.221, fail** | **0.218, fail** | **0.988** |
| Obstruction resistance | **0.356, fail** | **0.195** | **0.984** |

All requested VPN dimensions produced strongly graded actual spike counts, so gross drive-to-VPN failure is not supported. The locked 42-MBON/linear-reconstruction information gate failed on slip and obstruction, including slip's error floor. This result does **not** locate a unique loss at KC, individual MBONs, or the analytical ridge boundary; it also does not prove that a reward learner could not use these features. Under the predeclared stop rule, such learning was **not tested**. No pooled-decoder substitution, feature change, refit, or threshold adjustment was made.

## Gate ledger and decision

| Gate / stage | Outcome |
| --- | --- |
| Task validity and fixed case/action coverage | **Pass**: 28/53/44 optimal grid points; +0.1348 grid gap; 12 forced outcomes per octant/action. |
| Raw-observable independent learning | **Pass**: final 0.6412 expected reward, 108/120 realized recoveries, 31/38/51 actions, +0.2500 versus initial, 95.5% oracle-gap closure. |
| Neural VPN drive and deterministic replay | **Pass**: correlations 0.988/0.988/0.984; exact keyed repeat; unchanged neural multipliers. |
| Neural sensory information, all three dimensions | **Fail**: slip R² 0.221 and MAE 0.218; obstruction R² 0.356. |
| Neural Fly A own-reward learning and independent 19103 replication | **Not run**, by the locked stop rule. |
| DN messages, Fly B training, five-arm teaching comparison | **Not run**; no inference about recovery teaching. |

The **single next step under this protocol is stop** and preserve this sensory failure. A new neural encoding or decoder would require a separately justified, prospectively locked study; this run gives no permission to tune one. A controlled teacher–student recovery experiment is **not justified** because the neural sensory prerequisite failed. The successful RPS teaching results remain distinct and unchanged.

## Artifacts and reproduction

- [Raw write-once result](../data/practical_transfer_redesign_initial_raw.json): complete 480 training and 360 frozen rows, matched case/outcome tapes, controller snapshots, costs, and raw gates.
- [Sensory write-once result](../data/practical_transfer_redesign_initial_sensory.json): 96 calibration and 96 independent validation neural records with VPN, KC, individual and pooled MBON measurements, fixed scaling/reconstruction, and sensory gates.
- There is **no** neural-teacher or confirmation artifact because those stages were not authorized to run after the failed gate.

From the repository root, reproduce only in a **copy without these write-once files**: `node tools/test_practical_transfer_redesign_controller.mjs`, `node tools/check_practical_transfer_redesign.mjs preflight`, then `node tools/check_practical_transfer_redesign.mjs initial`. The runner refuses overwriting either artifact. No Git operation, historical-artifact edit, RPS-system change, or demo change occurred.
