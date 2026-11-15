# Prospective Gemini supervision check — locked before API calls

This is separate from the saved deterministic-fixture experiment in `data/fraud/experiment.json`. Its result must never replace that artifact.

- Dataset: `fraud.core.make_dataset()` with seed 720260. Training accounts `mock-000` through `mock-047`; untouched held-out accounts `mock-060` through `mock-071`. Synthetic scenario labels are generated independently of Gemini. No Nessie records enter this evaluation.
- Teacher: `gemini-3.5-flash-lite` via Gemini `generateContent`, JSON schema `fraud.teacher.SCHEMA`, and the exact `GEMINI_PROMPT` constant in `fraud/teacher.py`. Input is the six `FEATURE_NAMES` values rounded to three decimals. No account or merchant identifiers. No search or outside data.
- Supervision budget: exactly 12 distinct training cases, selected by `np.random.default_rng(SEED + 99).permutation(480)` immediately after the first 72 labeled indices. The first selected case doubles as the provider connectivity check and is cached. Stop on an API or schema failure; never substitute a fixture response.
- Student: existing `fit_student` NumPy logistic learner, 400 fixed updates, step size .12, regularization .01, teacher weight .4, teacher targets LOW=.05, REVIEW=.5, HIGH=.95. Both arms use the same 72 true training labels, train-only scaler, and fixed 0.5 decision threshold. Baseline uses no teacher messages.
- Metrics on the same 120 held-out cases: fraud recall, precision, F1, false alerts, and TP/FP/FN/TN. Success requires Gemini-guided F1 strictly above baseline, fraud recall at least baseline, and false-positive rate no more than baseline plus 0.02. All conditions must hold. No threshold or hyperparameter changes after observing held-out scores.
- Output: one write-once `data/fraud/gemini-experiment.json` plus a separate local student artifact. Report a failed gate as a negative result, without changing the fixture result.

Implementation note before held-out evaluation: the first teacher call validated, but the next training case repeatedly returned an `uncertainty` string beyond the existing 140-character application limit. The schema descriptions were clarified to state the already locked length limits; the prompt, selected cases, learner, metrics, and gate were unchanged. Failed responses are discarded and never converted to fixture advice.
