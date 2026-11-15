# New-scenario Gemini study — locked protocol

This is a separate synthetic study. The original rule-teacher result, the first real-Gemini result, and the merchant-tenure result remain unchanged. The old held-out templates influenced the decision to add merchant tenure, so none is reused as new held-out evidence.

## Question and data

Can 12 compact Gemini risk messages improve a seven-feature local classifier on new, independently labeled scenarios? Generate deterministic synthetic accounts using seed `20261114`. Accounts 1000–1035 are training only; 2000–2011 are development only; 3000–3023 are final evaluation only. Training families are `marketplace_probe`, `service_charge`, and `routine_disruption`; development families are `vendor_cluster` and `renewal_jump`; final families are `ticket_exchange` and `pharmacy_errand`. Family names and account IDs never enter the seven features or teacher prompts. Each family has both injected unauthorized and authorized variants, determined by the generator's independent intent bit. The bit, not Gemini, defines the synthetic label. The final family definitions are distinct from the old fixed scenario templates, and final accounts and families are excluded from training and development.

The student uses exactly `log_amount_ratio`, `new_merchant`, `new_category`, `recent_count`, `same_merchant_count`, `recent_spend_ratio`, and `merchant_older_count`. All are computed from prior mock transactions. Only training data set the scaler. A zero-initialized logistic model is trained for 400 full-batch steps at learning rate 0.12 and L2 0.01; threshold is 0.5. No threshold or architecture search.

## Three arms and teacher budget

Select 36 training cases for independent labels and 12 different training cases for additional supervision using deterministic label-blind permutations. All three arms have the same seven-feature representation, scaler, 400-step optimizer, and 36 base labels.

1. **Limited baseline:** 36 independent labels only.
2. **Independent-label control:** the same 36 plus the same 12 extra cases with their generator-supplied labels, extra-case weight 0.4.
3. **Gemini-guided:** the same 36 plus 12 Gemini messages on those exact extra cases. Use `gemini-3.5-flash-lite`, the existing validated JSON schema, and the exact `NEW_SCENARIO_PROMPT` in the implementation. Send only seven rounded feature values. Map LOW/REVIEW/HIGH to soft targets 0.05/0.5/0.95. Parse only indicators that begin with an exact feature name and colon; use the number of distinct cited features to set each extra-case weight to `0.2 + 0.1 × min(count, 4)`. The teacher gives no account identity, scenario, intent, or ground truth. Cache valid responses keyed by model, prompt, and features. At most 12 new teacher messages; no teacher call at inference.

## Evaluation and decision

Run one development evaluation for all arms. Freeze the candidate, then run one final evaluation on the untouched final accounts and families, regardless of development outcome. Record every per-case prediction; TP, FP, FN, TN; recall, precision, F1, false-positive rate; and teacher message count. The primary gate passes only if Gemini F1 is strictly above **both** controls, recall is at least both, and false-positive rate is no more than 0.05 above either control. An independently labeled control may outperform Gemini; preserve that result. No seed search or test-driven changes. These synthetic labels do not measure real-world fraud accuracy.
