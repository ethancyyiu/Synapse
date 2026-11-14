# Synapse Fraud: two-minute demo

## Start

From the repository root, run `python -m fraud.server` and open <http://localhost:8767/fraud.html>. Python 3 and NumPy are required. No key is needed for the clearly labeled generated-fixture mode. The original Synapse page remains at <http://localhost:8767/index.html>.

To use live Nessie **mock** data, set `NESSIE_API_KEY` in the server process environment and restart. The read-only adapter then retrieves accessible accounts, merchants and purchases. It will show an error if the Nessie API fails; it does not silently pretend a fixture was a live response. `NESSIE_BASE_URL` optionally overrides the HTTPS API host. Never put the key in the page, URL you share, or repository.

For an optional on-demand frontier comparison, set both `OPENAI_API_KEY` and `SYNAPSE_FRAUD_TEACHER_MODEL` before server startup. The dashboard then exposes a button for a selected transaction. Only six derived numerical features are sent, with no account ID, merchant ID, name or transaction amount. The response must pass the declared LOW/REVIEW/HIGH schema. This call is **not** part of the saved controlled experiment. No frontier call was made for the recorded results.

## Judge path, about two minutes

1. **0:00–0:20 — Problem.** “When a transaction looks strange, a reviewer needs context, not a black-box accusation. We built a local risk queue for mock banking transactions. The decision stays with a person.” Point to the fixture/Nessie mode badge.
2. **0:20–1:00 — Live workflow.** Select a mock account, press **Retrieve & analyze**, and open the highest-ranked transaction. Point to the account median, recent activity, new-merchant evidence, and the local model score. Mark it for review. Explain that this changes only browser session state; no transaction is blocked.
3. **1:00–1:30 — Transfer question.** Switch between **Independent** and **Teacher-guided** local students. “We trained the same seven-parameter model with and without 408 compact LOW/REVIEW/HIGH messages from a deterministic rule teacher. In both cases the teacher is disconnected when the student scores these transactions.” The teacher fixture is not a frontier LLM.
4. **1:30–1:55 — Controlled result.** Scroll to the recorded experiment. “Guidance caught 28 of 36 injected suspicious cases versus 25 independently, but created five false alerts versus zero. F1 was 0.812 versus 0.820, so our predefined success test failed. Shuffled messages performed much worse.” Show the raw-result and protocol links.
5. **1:55–2:00 — Connection and boundary.** “Synapse's original fly-brain work separately showed compact three-symbol messages can travel as measured simulated spikes. We have not connected that neural channel to this fraud student. This is a synthetic research prototype.”

## If an API or browser step fails

The fixture mode is the reliable presentation path. Refresh the page and choose mock account 060. The saved result table is served from a local file and needs no external network. Keep the original RPS demo available in a separate tab for Science + Discovery. Do not call this a deployed fraud detector or claim Nessie purchases have verified fraud labels.
