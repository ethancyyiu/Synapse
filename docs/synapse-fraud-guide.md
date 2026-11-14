# Synapse Fraud: run and demo

Synapse Fraud is an experimental, human-reviewed queue for **mock** banking purchases. Select an account, analyze transactions with a local logistic classifier, inspect history-based indicators, and mark an item for review or dismissal. Those actions live only in the browser session; no financial transaction is changed. The original [Synapse Science demo](../index.html) stays separate.

## Start and data sources

From the repository root, install `requirements.txt` if needed, put your keys in the root `.env` file, run `python -m fraud.server`, and open <http://localhost:8767/fraud.html>. The `.env` file is ignored by Git and blocked from both local web servers. Existing process environment variables take precedence. Restart the server after editing `.env`. With no `NESSIE_API_KEY`, the page clearly says **Offline fixture mode** and uses deterministic generated mock purchases. With `NESSIE_API_KEY`, it retrieves read-only Nessie mock accounts, purchases, and merchants. `NESSIE_BASE_URL` can override the default HTTPS API host. Nessie mock purchases are *not* verified fraud labels; the adapter has not yet been exercised against a live credential in this project.

## Models and teacher

The saved baseline, rule-guided, and shuffled-control students are seven-parameter **NumPy logistic classifiers**, not LLMs. Scoring reads purchase amount and merchant/category patterns against account history. Their displayed scores rank human review work and are not calibrated probabilities. The saved study used a deterministic LOW/REVIEW/HIGH **rule fixture**, never OpenAI.

With `OPENAI_API_KEY` in `.env`, the server offers an on-demand structured teacher comparison for a selected purchase. It sends only six derived numerical features, without customer or account identifiers. The configured cost-oriented model is [`gpt-5.6-luna`](https://developers.openai.com/api/docs/models/gpt-5.6-luna); change `SYNAPSE_FRAUD_TEACHER_MODEL` in `.env` if that model is unavailable to your account. To train a separate local student from **20 training-only teacher responses** by default, run `python -m fraud.train_frontier --teacher-examples 20`, then restart the server. Successful responses are cached in `fraud/.teacher-cache/`; repeating the same run reuses them. `--teacher-examples` may be raised to 408 if that API spend is intended. The model is saved at `data/fraud/frontier-student.json` and appears as a separate local option. Its inference needs no API key or teacher call. This optional path has **not** been run with a real key or given a held-out scientific result. scikit-learn is not installed; the NumPy implementation keeps the working demo dependency-light.

## Experiment and limits

The [locked protocol](synapse-fraud-protocol.md) uses account-disjoint synthetic training, development, and held-out sets with independently assigned suspicious and challenging benign scenarios. All three saved models saw the same 72 true training labels; the guided and shuffled arms received 408 extra compact fixture messages. At the fixed 0.5 evaluation threshold, guided caught 28/36 suspicious cases versus 25/36 for the independent model, but added five false alerts versus zero. F1 fell from 0.820 to 0.812, so the predeclared success gate **failed**. See the [research report](synapse-fraud-research.md) and [raw predictions](../data/fraud/experiment.json). The dashboard's priority bands use separate 0.35 and 0.7 cutoffs. No real-world fraud accuracy, neural transport for fraud, or frontier-teacher benefit is established.

## 60-second judge path

1. **0–15 s:** Point to the source badge. Explain that a reviewer needs a ranked queue with context and that this run uses offline mock purchases.
2. **15–35 s:** Select account `mock-060`, press **Analyze Transactions**, open the highest-ranked purchase, and show amount/history indicators. Mark it for review; this records only a local session decision.
3. **35–50 s:** Switch baseline and guided local students. Explain that the teacher was disconnected before these scores were produced.
4. **50–60 s:** Show the recorded comparison: three more suspicious cases caught, five extra false alerts, and a failed overall gate. Link the distinct Science demo as the origin of the teacher–student question.
