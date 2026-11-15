# Synapse Fraud prospective evaluation protocol

Locked before the first held-out evaluation. No threshold, seed, feature, architecture or training budget is changed in response to test results. Results will be appended to `docs/synapse-fraud-research.md` and raw predictions saved under `data/fraud/`.

## Dataset and labels

Generator version `fraud/core.py`, seed 720260. Accounts 0–47 are training, 48–59 development, and 60–71 held-out test. Account profiles and transaction values are generated with separate seeded streams per account; no account or exact transaction is shared across splits. Each account has a private normal-spending profile, thirty baseline purchases, and ten independent candidate scenarios. Ground truth comes from scenario assignment, not an inferred property of Nessie purchases. Cases: four ordinary benign, one large purchase from a familiar merchant, one small purchase from a new merchant, one burst from a familiar merchant, and three injected suspicious patterns (large purchase at a new unfamiliar merchant; repeated purchases at a new merchant; and a fast burst in an unfamiliar category). Benign unusual cases are explicit false-positive challenges. Names, amounts, dates and merchants vary by account. This is a synthetic benchmark, not verified real fraud.

For each candidate, history is the account's baseline plus any scenario-specific prior purchases before the target. Features available to every arm: log amount/median history amount, new merchant, unfamiliar category, count of purchases in the last 24 hours, same-merchant count in the last 24 hours, and recent spend/median history amount. No scenario name or label enters any feature. Only training accounts are used for feature scaling.

## Models and supervision

All students are the same NumPy logistic classifier with bias, six numerical features, L2 penalty 0.01 and zero initial parameters. Training uses 400 full-batch gradient steps at learning rate 0.12. A fixed 15% of training cases receives its true scenario label; the other 85% has no label available to the independent arm. The trained-teacher arm additionally receives one compact LOW/REVIEW/HIGH message from a deterministic rule-based teacher fixture for each otherwise unlabeled training case. LOW and HIGH provide soft targets 0.05 and 0.95; REVIEW provides 0.5. True-label loss has weight 1 and teacher-supervision loss has weight 0.4. The shuffled control gets the same message histogram randomly permuted across otherwise unlabeled cases. All arms use the same features, initial parameters and training code; only supervision differs. The fixture teacher reads features, not scenario labels. It is not a frontier LLM.

## Evaluation and gate

Use a fixed 0.5 score threshold for binary suspicious/benign evaluation. Report precision, recall, F1, false-positive rate, fraud-class recall, confusion matrices, average inference time, message count (one three-class message per otherwise unlabeled training case), and raw predictions on all held-out cases. Development cases may be inspected only to debug implementation, not to change this protocol. No teacher call or message occurs during test scoring. Success requires guided F1 to exceed both independent and shuffled F1 and guided false-positive rate to exceed neither control by more than 0.02. Otherwise report failure. No rerun with changed seeds or hyperparameters to seek a pass.

## Stop and interpretation

Stop if splits overlap, teacher reads labels, model parameters do not change, or held-out scoring requires teacher access. A pass supports only this synthetic generator and fixture teacher. A failure is recorded without changing the gate. No neural-spike or frontier-model outcome is inferred from this comparison.
