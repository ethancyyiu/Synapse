# Bank view: a credit-union agent around the real-data fraud study

**An engineering demo, not a locked study.** It adds no tested claim. Its evidence is the locked study it wraps: [`docs/real-data-advice-v2-protocol.md`](real-data-advice-v2-protocol.md), result in [`data/real_data_advice/fraud_v2_summary.json`](../data/real_data_advice/fraud_v2_summary.json), which **failed its pass rule narrowly** and is reported that way on the page.

It uses real card-fraud data and ordinary machine-learning models. **It does not use the fly connectome, spikes, or either fly.**

## Run it

```bash
pip install scikit-learn pandas numpy
python tools/fetch_real_data.py                 # once; you only need the fraud file
python tools/bank_demo.py sample                # choose the ledger purchases
NESSIE_KEY=your_key python tools/bank_demo.py sync    # post them to Nessie, read them back
python tools/bank_demo.py run                   # the agent
python tools/serve.py                           # then open http://localhost:8766/#bank
```

`python tools/bank_demo.py sync --offline` skips Nessie and writes the same file marked `offline`; the page then says Nessie was not used. Commit `data/bank_demo/` after a Nessie run so the page works without a network.

## What it does

1. **Sample.** 20 fraud and 180 legitimate purchases are drawn (rng `7101`) from the held-out test cases of study seed 101. Frauds are over-sampled for display; the real rate is about 1 in 600.
2. **Ledger.** Each is posted to the Nessie API as a purchase on one demo account, with the real amount and a description that names its row in the data. The purchases are then read back from Nessie. Nessie holds only the amount and that row number; the 28 anonymised features stay in the local data file.
3. **Agent**, as the credit union:
   - reads the ledger;
   - counts its labelled frauds since the change and asks the bank for advice if there are fewer than 30;
   - receives one bit per purchase from the bank's model (the study's 24,500 advice bits, plus 500 bits on its labelled cases, used only for checking);
   - accepts the advice only if (a) the bank's model beat the credit union's old model by 0.02 average precision on the bank's own held-out cases, the study's teacher check, (b) the bank flags at most 10 times the credit union's old fraud rate, and (c) the bits agree with at least 99% of the credit union's labelled cases;
   - retrains with or without the advice and scores the ledger purchases;
   - reports average precision on all held-out purchases and checks that it equals the study's saved result for seed 101. If it does not, the page shows a warning.

## Why seed 101

Fixed by rule before the demo was run on the real data: among the 12 seeds that completed the locked study, the median `taught - best control` is +0.023, shared by seeds 101, 111 and 114; 101 is the lowest. It is one run. The 15-seed result is the evidence.

## Limits

- **The agent's rules are untested.** The thresholds (30 labelled frauds, 10 times the fraud rate, 99% agreement) are our choices. No control arm measures whether they help.
- **The credit union cannot verify the advice itself.** It has about one labelled fraud, so check (c) mostly measures agreement on legitimate purchases. Check (a) relies on the bank's own report.
- **The checks are also run on an out-of-date bank model** and the page reports whether they would have rejected it. In the fly studies, learning how much to trust a teacher removed most but not all of the harm from an out-of-date one, and that study failed its pass rule (`docs/trust-gate-cohort-protocol.md`).
- **Asking only when unsure was tested in the fly and failed** its teacher-specific test (`docs/adaptive-sparse-teaching-report.md`). The agent's ask rule here is a different, simpler rule and makes no claim about saving messages.
- **Nessie is a sandbox ledger.** Customers, accounts and merchants there are mock. The amounts and fraud labels come from the Worldline/ULB data. No bank or credit union was involved.
- **The PSI figure is reported, not used.** The agent does not decide from it.
