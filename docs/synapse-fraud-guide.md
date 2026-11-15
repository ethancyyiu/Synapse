# Synapse Fraud: run and demonstrate

Synapse Fraud is a human-reviewed investigation queue for **mock** banking purchases. It retrieves Nessie sandbox records, computes account-history features, ranks purchases with a local NumPy student, shows observable indicators, and lets a reviewer mark a transaction for review or dismissal. Review decisions remain in browser session storage; they do not change banking records. The original [Science demo](../index.html) is separate and unchanged.

## Start

From the repository root, install `requirements.txt` if needed, put credentials in the root `.env`, run `python -m fraud.server`, then open <http://localhost:8767/fraud.html>. Restart the server after editing `.env` or training artifacts. The `.env` is ignored by Git and blocked by the local web servers. Existing process environment variables take precedence.

```dotenv
NESSIE_API_KEY=your_nessie_sandbox_key
GEMINI_API_KEY=your_gemini_key
TEACHER_PROVIDER=gemini
GEMINI_MODEL=gemini-3.5-flash-lite
```

An `nsk_` key selects Nessie v2 at `https://v2-prod-api.nessieisreal.com` using bearer authentication. The dashboard's read adapter makes only GET requests. The approved fictional demo records were created in the mock sandbox: one customer, one checking account, Corner Market and Nova Electronics, and six purchases. On 14 November 2026, the adapter read back all six purchases with both merchant names and categories. The page displays **LIVE NESSIE** only after purchase records are retrieved. If Nessie is unavailable, it offers clearly labeled offline fixtures. Nessie records have no verified fraud labels.

The separate `python -m fraud.seed_nessie_demo` script was run only after approval of those exact records. It stores returned resource IDs in `data/fraud/nessie-demo-records.json` so a stopped run can resume. Do not run it against another key without separate authorization.

## Models and teacher

The original saved students use six history features and seven logistic parameters including bias. The separate seven-feature study adds the count of purchases at the same merchant more than 24 hours earlier; its control and Gemini-guided local students are available in the model selector. The seven features are derived only from purchase amount, time, category, merchant, and earlier account purchases. Scores are uncalibrated review rankings, never verified fraud predictions. The dashboard shows the earlier deterministic rule fixture separately from actual Gemini messages.

With `TEACHER_PROVIDER=gemini`, the optional on-demand teacher uses structured JSON and caches successful responses in `fraud/.teacher-cache/`, labeled **live Gemini** or **cached Gemini**. A teacher failure is reported, not replaced with fixture advice. Local student inference makes no Gemini request. OpenAI remains available with `TEACHER_PROVIDER=openai` and `OPENAI_API_KEY`. Teacher requests send only derived numeric features; no customer or merchant IDs.

## Saved research, including failures

- The original [rule-fixture study](synapse-fraud-protocol.md) caught 28/36 injected suspicious cases versus 25/36 for the limited-label baseline, but made five false alerts versus zero. F1 fell from 0.820 to 0.812; its [saved gate failed](../data/fraud/experiment.json).
- The first [12-message real-Gemini study](synapse-fraud-gemini-protocol.md) matched the baseline on its 120 synthetic test cases: TP 25, FN 11, FP 0, TN 84, F1 0.820. Its [gate failed](../data/fraud/gemini-experiment.json).
- The [seven-feature merchant-tenure study](synapse-fraud-tenure-protocol.md) gave both matched arms TP 36, FN 0, FP 0, TN 84 on a new account-disjoint but template-repeated synthetic set. Gemini did not improve the control; its [gate failed](../data/fraud/tenure-heldout.json). Template repetition limits what perfect scores mean.
- The [new-scenario three-arm study](synapse-fraud-new-scenarios-protocol.md) used 36 base labels and 12 additional cases for either independent labels or structured Gemini messages. On 48 held-out cases from different accounts and scenario families, all three arms had TP 24, FN 0, FP 24, TN 0, recall 1.000, precision 0.500, F1 0.667, and false-positive rate 1.000. The [teacher-advantage gate failed](../data/fraud/new-scenarios-heldout.json). The complete per-case predictions and [development result](../data/fraud/new-scenarios-development.json) are saved. A preflight generator check counted final examples before training, but final feature values and predictions were not used for model selection.

The [new study execution record](synapse-fraud-new-scenarios-execution.md) documents one discarded malformed Gemini response and a transient 503 before the 12 valid messages were collected.

## Separate neural teaching demonstration

The **Neural Teaching** section is a new, separate synthetic response task. It reuses the existing 6,900-neuron connectome, Fly A and Fly B controllers, and the two DNge175 spike channel from the fault-response experiment. It does not alter the original RPS, fault-response, Gemini, or NumPy classifier results. The [fixed neural protocol](synapse-fraud-neural-protocol.md) declares policy mappings, controls, checkpoints, thresholds, and final seeds before comparison.

The backend maps each loaded mock purchase to one of three observable cues: other activity; at least two earlier purchases in 24 hours; or an amount at least three times the account median combined with merchant novelty or a recent cluster. This is the only information sent from a selected Nessie or fixture purchase into Fly B. The incident response rule and correct actions are synthetic, never confirmed fraud labels. The frozen Fly B browser probe uses the real connectome and trained local readout with Fly A disconnected; it does not call Gemini or change the NumPy risk score. The teaching trace and multi-seed comparison replay saved, measured simulations. A “block” action here is an experimental response symbol only and never modifies Nessie.

To reproduce the bounded study from scratch in a clean checkout, run these commands from the repository root. Result files are write-once, so do not rerun seeds already present.

```powershell
node tools/run_fraud_neural.mjs probe
node tools/run_fraud_neural.mjs seed 401
420..424 | ForEach-Object { node tools/run_fraud_neural.mjs seed $_ }
node tools/summarize_fraud_neural.mjs
python -m fraud.server
```

Open <http://localhost:8767/fraud.html>, select a mock account and purchase, then open **Neural Teaching**. The saved Fly A episode scores and measured DN message trace appear first. Click **Next neural message**, then **Disconnect Fly A & test Fly B** to run the selected purchase cue in a browser worker. The table shows the five frozen comparisons and the predeclared gate. The study tests adaptation in a small synthetic policy task, not real-world fraud detection or biological fly-to-fly teaching.

None of these synthetic scores measures real banking fraud accuracy. The new family shift showed that even a classifier with high recall can be unusable when false alerts are high. No parameters or thresholds were adjusted after the final test.

## 60-second judge path

1. Select **Nessie live mock API** and **Synapse Demo**. Point to **LIVE NESSIE** after the six purchases load, then show both merchant names and the ranked review queue.
2. Open a purchase. Explain its amount, history median, recent activity, merchant tenure, and observed indicators. Mark it for review; the action stays in this browser session.
3. Switch between the seven-feature local control and Gemini-guided student. Both run with Gemini disconnected. The optional teacher button is a separate live or cached comparison, not the source of the current score.
4. Show the new three-arm result, including the 24 false alerts and failed gate. Explain that the practical workflow is working, while the synthetic research does not yet validate an accurate fraud detector. Link the separate Science demo as the origin of the teacher–student research question.
5. Open **Neural Teaching**. Show the selected purchase cue, Fly A training scores, and a measured DN message. Disconnect Fly A and probe frozen Fly B, then compare all five saved seeds against alone, random, and shuffled advice.
