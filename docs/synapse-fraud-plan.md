# Synapse Fraud: implementation plan

## Architecture

An isolated `fraud.html` page calls a local Python server in `fraud/server.py`. The server serves the original Synapse files unchanged and adds read-only `/api/fraud/*` endpoints. `fraud/nessie.py` retrieves Nessie mock accounts, purchases and merchants with the key held only in the server process. With no key, the API says `fixture` and uses generated mock accounts. `fraud/core.py` normalizes purchases, builds history-derived features, trains a small NumPy logistic student and generates reproducible synthetic fraud cases. The student runs locally and is frozen at evaluation and demo inference.

## What can be reused

The existing page and Python static-server pattern can be reused. The RPS teacher and student code is task-specific and should remain untouched. `js/rps_challenge_students.js` defines a three-symbol DNge175 two-neuron code and measures actual simulated spike counts before decoding. That can be tested separately as a LOW/REVIEW/HIGH transport channel; it is not equivalent to distilling an LLM. The successful RPS result is prior research, not evidence that neural messages improve fraud detection.

Python has NumPy and requests, but no scikit-learn, Torch, Transformers or local LLM stack. No Nessie or frontier API key was present in the inspected process environment. Do not log, copy to the browser, or save credentials. A frontier API is optional; its deterministic fixture must be labeled as such.

## Stages

1. Read-only Nessie adapter and fixture fallback; normalize purchase and merchant fields and compute account-history features.
2. Fixed-seed account-disjoint train, development and test synthetic cases, including high-value benign and burst-like benign cases. Nessie purchases have no fraud ground truth.
3. Train independent, compact-teacher and shuffled-teacher local students with actual parameter updates. Keep feature set, architecture and training budget matched. Lock the protocol before the held-out run.
4. Save raw held-out predictions and metrics, including false-positive rate and latency. Report a failed teacher benefit if observed.
5. Build a review dashboard: select account, retrieve, rank, inspect evidence, review or dismiss in browser session, and show a frozen research comparison. No financial actions.
6. If time and validated evidence permit, separately test neural delivery versus digital delivery and a configured frontier teacher. Do not let optional work destabilize the application.

## Demo path

Open `fraud.html`, select a clearly labeled mock account, load its purchases, inspect the top-ranked transaction and history evidence, make a simulated review decision, then show the frozen teacher-disconnected experiment. Explain that the local model is a trainable classifier, the teacher fixture is deterministic, and the fraud labels are synthetic. Link to the original RPS demo for the separate neural-channel result.

## Limits

This is a research prototype for simulated fraud scenarios, not a deployed fraud detector. Risk scores are not calibrated probabilities. Unusual purchases alone are not labeled fraud. Nessie is a banking simulation, and its data is not inherently labeled fraudulent. The [Nessie documentation](https://prod.nessieisreal.com/docs) describes mock banking resources.
