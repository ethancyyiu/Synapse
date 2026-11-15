# Synapse Fraud: run and demo

Synapse Fraud is an experimental, human-reviewed queue for **mock** banking purchases. Select an account, analyze transactions with a local logistic classifier, inspect history-based indicators, and mark an item for review or dismissal. Those actions live only in the browser session; no financial transaction is changed. The original [Synapse Science demo](../index.html) stays separate.

## Start and data sources

From the repository root, install `requirements.txt` if needed, put keys in the root `.env` file, run `python -m fraud.server`, and open <http://localhost:8767/fraud.html>. Restart after editing `.env` or training a student. The `.env` file is ignored by Git and blocked from both local web servers; existing process environment variables take precedence. Use:

```dotenv
NESSIE_API_KEY=your_nessie_key
GEMINI_API_KEY=your_gemini_key
TEACHER_PROVIDER=gemini
GEMINI_MODEL=gemini-3.5-flash-lite
```

Choose **Offline generated fixtures** or **Nessie live mock API** in the dashboard. An `nsk_` key uses Nessie v2 bearer authentication; older keys use the v1 adapter. Both adapters only read mock banking data. `NESSIE_BASE_URL` can override the detected HTTPS host. The configured v2 key authenticated successfully on 14 November 2026, but its sandbox returned **zero accounts**. Transactions and merchant lookup therefore could not be verified live. The page displays labeled offline fixtures and never marks them LIVE NESSIE. Live Nessie purchases have no verified fraud labels.

## Models and teacher

The saved baseline, rule-guided, and shuffled-control students are seven-parameter **NumPy logistic classifiers**, not LLMs. Scoring reads purchase amount and merchant/category patterns against account history. Their displayed scores rank human review work and are not calibrated probabilities. The saved study used a deterministic LOW/REVIEW/HIGH **rule fixture**, never OpenAI.

With `TEACHER_PROVIDER=gemini`, the server uses `GEMINI_API_KEY` and the [Gemini 3.5 Flash-Lite](https://aistudio.google.com/docs/models/gemini-3.5-flash-lite) model. One real structured response succeeded with mock features; its risk, explanation, and indicators were checked against the supplied values. The dashboard's optional teacher button reports **live Gemini** or **cached Gemini** explicitly. Successful responses are cached in `fraud/.teacher-cache/`. No teacher API is called while local students score transactions. To use OpenAI instead, set `TEACHER_PROVIDER=openai`, `OPENAI_API_KEY`, and optionally `SYNAPSE_FRAUD_TEACHER_MODEL`. Both providers receive only six derived numerical features, without account or merchant identifiers. An API or validation failure is shown as an error; it never becomes fixture advice.

## Experiment and limits

The [locked protocol](synapse-fraud-protocol.md) uses account-disjoint synthetic training, development, and held-out sets with independently assigned suspicious and challenging benign scenarios. All three saved models saw the same 72 true training labels; the guided and shuffled arms received 408 extra compact fixture messages. At the fixed 0.5 evaluation threshold, guided caught 28/36 suspicious cases versus 25/36 for the independent model, but added five false alerts versus zero. F1 fell from 0.820 to 0.812, so the predeclared success gate **failed**. See the [research report](synapse-fraud-research.md) and [raw predictions](../data/fraud/experiment.json). The dashboard's priority bands use separate 0.35 and 0.7 cutoffs. No real-world fraud accuracy, neural transport for fraud, or frontier-teacher benefit is established.

The separate [prospective Gemini protocol](synapse-fraud-gemini-protocol.md) used 12 valid training-only Gemini messages and the same local learner. On the untouched 120-case synthetic test, the Gemini-guided student **matched** the baseline: TP 25, FN 11, FP 0, TN 84; fraud recall 25/36, precision 1.000, F1 0.820. The strict improvement gate **failed**. See the [separate result](../data/fraud/gemini-experiment.json). The trained local model appears as a separate dashboard option, with no teacher needed for inference. Neither synthetic result establishes live fraud accuracy.

## 60-second judge path

1. **0–15 s:** Point to the source badge. Explain that a reviewer needs a ranked queue with context and that this run uses offline mock purchases.
2. **15–35 s:** Select account `mock-060`, press **Analyze Transactions**, open the highest-ranked purchase, and show amount/history indicators. Mark it for review; this records only a local session decision.
3. **35–50 s:** Switch baseline and guided local students. Explain that the teacher was disconnected before these scores were produced.
4. **50–60 s:** Show the recorded comparison: three more suspicious cases caught, five extra false alerts, and a failed overall gate. Link the distinct Science demo as the origin of the teacher–student question.
