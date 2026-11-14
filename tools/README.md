# Tools

Most files here are one-off runners for experiments that have finished. They are kept, unchanged, so each lab-notebook entry can still be reproduced with the command it gives. You only need the short list below.

## Run and check

| Command | What it does | Time |
|---|---|---|
| `python tools/serve.py` | Serves the demo at <http://localhost:8766> with caching off | - |
| `npm test` | `verify_claims.mjs` then `check_links.mjs` (below) | under 2 s |
| `node tools/verify_claims.mjs` | Recomputes the locked tests of the cohorts from the per-seed files, and checks the summary files, README, and page all quote the same numbers | under 1 s |
| `node tools/test_engine.mjs` | Checks `js/engine.js` against closed-form answers; add `--compare` to print its activity beside `js/sim.js` on the real connectome | under 1 s (about 1 min with `--compare`) |
| `node tools/check_links.mjs` | Checks every local link, script, and data file referenced by the pages and main documents | under 1 s |
| `npm ci` | Once, before the two demo checks: Node loads three.js from `node_modules` (the browser uses `vendor/three/`) | - |
| `node tools/check_rps_challenge_demo.mjs` | Main demo: rules, recorded run, live students, frozen tests | about 1 min |
| `node tools/check_rps_demo.mjs` | Legacy page | about 1 min |
| `npm run check:all` | Everything above plus the Python tests (`flyhero.test`, `flyhero.test_rps`; needs NumPy) | a few min |

## The current studies

Each study has a protocol in `docs/` that was committed before its seeds were run, a runner that writes one write-once file per seed, and an analysis script that writes the summary.

| Study | Protocol | Runner | Analysis | Results |
|---|---|---|---|---|
| Rule switch, 15 seeds | `docs/rule-switch-cohort-protocol.md` | `check_rule_switch_cohort.mjs` | `analyze_rule_switch_cohort.mjs` | `data/rule_switch_cohort/` |
| Out-of-date teacher, 3 + 15 seeds | `docs/stale-teacher-control-protocol.md`, `docs/stale-teacher-cohort-protocol.md` | `check_stale_teacher_control.mjs` | `analyze_stale_teacher_cohort.mjs` | `data/stale_teacher_control/`, `data/stale_teacher_cohort/` |
| Trust gate, 3 + 15 seeds | `docs/trust-gate-pilot-protocol.md`, `docs/trust-gate-cohort-protocol.md` | `check_trust_gate_pilot.mjs` | `analyze_trust_gate_cohort.mjs` | `data/trust_gate_pilot/`, `data/trust_gate_cohort/` |
| Second task (fault response), 3 + 3 + 15 seeds | `docs/fault-response-pilot-protocol.md`, `docs/fault-response-full-change-protocol.md`, `docs/fault-response-cohort-protocol.md` | `check_fault_response_pilot.mjs` | `analyze_fault_response_cohort.mjs` | `data/fault_response_pilot/`, `data/fault_response_full_change/`, `data/fault_response_cohort/` |
| Own engine (`js/engine.js`), 3 + 15 seeds | `docs/engine-replication-pilot-protocol.md`, `docs/engine-replication-cohort-protocol.md` | `check_engine_replication.mjs` | `analyze_engine_replication_cohort.mjs` | `data/engine_replication_pilot/`, `data/engine_replication_cohort/` |
| Own engine, teacher scaling, 15 seeds | `docs/engine-teacher-scaling-protocol.md` | `check_engine_teacher_scaling.mjs` | `analyze_engine_teacher_scaling.mjs` | `data/engine_teacher_scaling/` |
| Two teachers, 3 + 15 seeds | `docs/two-teacher-pilot-protocol.md`, `docs/two-teacher-cohort-protocol.md` | `check_two_teacher_pilot.mjs` | `analyze_two_teacher_cohort.mjs` | `data/two_teacher_pilot/`, `data/two_teacher_cohort/` |

Runners refuse to overwrite an existing result file, so reproduce a seed in a copy of the repository without that file. One seed takes about 5 to 8 minutes.

## Everything else

- `check_rps_*`, `check_two_cue_*`, `check_package_recovery*`, `check_practical_transfer*`, `check_sensor_classification.mjs`, `check_warehouse_navigation.mjs`, and the `analyze_*`, `diagnose_*`, `probe_*`, and `plot_*` files belong to earlier experiments, most of which failed their gates. `docs/lab-notebook.md` gives the command for each.
- `record_demo_replay.mjs` and `record_rule_switch_replay.mjs` made the recorded runs the page plays back.
- `extract_subgraph.py` and `extract_brain_coordinates_3d.py` built `data/subgraph.json` and the brain positions from the connectome release.
- `research-loop/` holds the scripts that drove the unattended research sessions, and their logs in `research-loop/runs/`. The early free-form experiment log is `docs/notes/experiments.log`.
