# Synapse Fraud: first locked synthetic study

## Question and status

Can compact risk categories from a teacher improve a small student's fraud detection after the teacher disconnects? In this first fixed-seed **synthetic** study, the predetermined overall success rule **failed**. The guided student caught three more injected suspicious cases, but produced five more false alerts and had slightly lower F1 than independent learning.

The teacher was a deterministic rule fixture, **not** a frontier LLM. No Nessie or OpenAI API call was made for this result. Nessie purchases have no fraud ground truth and were not labeled as fraud. The financial prototype and the original fly-brain experiment are distinct systems.

## Method

The [prospective protocol](synapse-fraud-protocol.md) fixed one generator seed (720260), account-disjoint splits, six history features, a logistic student with seven trainable parameters, 400 gradient steps, a 0.5 binary decision threshold, and the comparison arms before held-out evaluation. Training had 48 mock accounts and 480 synthetic cases; development had 12 accounts and 120 cases; test had 12 new accounts and 120 cases. There were 36 injected suspicious cases in the held-out set and 84 benign cases, including large familiar purchases and benign bursts.

All arms received the same 72 labeled training cases. The guided arm also received 408 LOW/REVIEW/HIGH fixture-teacher messages for the other training cases; the shuffled arm received the same messages in random order. The independent arm had no teacher messages. The fixture teacher saw history features only, never the scenario label. The three students used the same architecture, feature scaling, initialization, and optimization. All student weight vectors changed from zero. No teacher was used to score held-out cases.

## Held-out result

| Student | TP | FN | FP | TN | Precision | Fraud recall | F1 | False-positive rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Independent | 25 | 11 | 0 | 84 | 1.000 | 0.694 | **0.820** | 0.000 |
| Teacher-guided | 28 | 8 | 5 | 79 | 0.848 | **0.778** | 0.812 | 0.060 |
| Shuffled advice | 12 | 24 | 0 | 84 | 1.000 | 0.333 | 0.500 | 0.000 |

The preregistered gate required guided F1 above both controls and false-positive rate no more than 0.02 above either. It failed both comparisons with the independent arm: 0.812 versus 0.820 F1 and 0.060 versus 0.000 false-positive rate. The gain in recall is descriptive and comes with extra false alerts. The shuffled arm was much worse, but this alone does not establish a useful teacher benefit.

The dashboard's HIGH/REVIEW/LOW queue bands use score cutoffs 0.7 and 0.35 to prioritize human review. The controlled binary metrics use the locked 0.5 threshold. These serve different display and evaluation purposes.

The raw artifact at [`data/fraud/experiment.json`](../data/fraud/experiment.json) includes every held-out prediction, trained coefficients, feature scaler, confusion matrices, and per-case inference timings. Timings were a few microseconds per transaction on this machine and are not a measured end-to-end API latency or deployment benchmark.

## Interpretation and limits

The fixture's three-class message provides information but is an imperfect supervision source. The independent model learned well from 72 labels on this simple generator. The guided model raised more alerts, including five benign cases. We will not tune this locked run after seeing the held-out results or claim that guidance improved overall fraud detection.

The cases are generated from fixed scenario families. Account identities, baselines, amounts, dates, and merchant identifiers are disjoint across splits, but this does not simulate the full diversity, adversarial adaptation, or class imbalance of real transactions. The scores are not calibrated probabilities. False positives matter because they cost review time and can harm customers if acted on automatically; this prototype takes no automatic financial action.

The existing RPS research validates measured three-symbol DN spike transport on its own task. Synapse Fraud did **not** send its risk messages through the neural channel, did not compare neural versus digital transport for fraud, and did not fine-tune a local LLM. The local student is a NumPy classifier. A configured frontier model can provide a separate, on-demand schema-validated comparison in the dashboard; it was unavailable and uncalled here.

## Reproduction

Run `python -m unittest fraud.test_fraud -v`. The experiment runner is write-once and refuses to overwrite the saved artifact. To reproduce from scratch, copy the repository without `data/fraud/experiment.json`, then run `python -m fraud.run_experiment` in that copy. The interactive dashboard runs with `python -m fraud.server` and <http://localhost:8767/fraud.html>.

## Short research log

- **Hypothesis:** Compact fixture-teacher supervision improves held-out F1 without a material false-positive-rate increase.
- **Experiment:** One fixed train/development/test synthetic cohort; independent, guided and shuffled arms.
- **Result:** Guided recall 0.778 versus 0.694 independent; F1 0.812 versus 0.820; false-positive rate 0.060 versus 0.000. Gate failed.
- **Conclusion:** No supported overall benefit on this benchmark. The model still supports a local, human-reviewed prototype.
- **Next experiment:** Separate, prospectively defined study with a validated stronger teacher and realistic data, if resources permit. The present artifact stays unchanged.
