"""Bank view: a credit-union agent on the real card-fraud data, with Capital One's Nessie API as the ledger.

This is an ENGINEERING DEMO, not a locked study. It re-runs one seed of the locked real-data fraud study
(docs/real-data-advice-v2-protocol.md) with the same splits, models and advice, and wraps an agent around it
that decides whether to ask for advice and whether to trust it. The agent's rules were not tested against
controls. Notes and limits: docs/bank-demo.md.

  python tools/fetch_real_data.py          # once: downloads the public fraud data into data/raw/
  python tools/bank_demo.py sample         # choose the purchases to put in the ledger
  NESSIE_KEY=... python tools/bank_demo.py sync      # post them to Nessie and read them back
  python tools/bank_demo.py sync --offline           # or: no Nessie, same file shape, marked "offline"
  python tools/bank_demo.py run            # the agent; writes data/bank_demo/agent_run.json for the page

`python tools/bank_demo.py all` does the three steps (Nessie if NESSIE_KEY is set, else offline).
"""
import argparse
import datetime
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

import numpy as np
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from run_real_data_advice import Student, evaluate, load_fraud  # noqa: E402
from run_real_data_advice_v2 import GATE_MARGIN, N_LABELLED, N_STREAM, pick_and_fit_teacher  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "bank_demo"
STUDY = ROOT / "data" / "real_data_advice"
NESSIE = os.environ.get("NESSIE_BASE", "https://v2-prod-api.nessieisreal.com").rstrip("/")

# Seed 101 was fixed by a rule before this demo was run: among the 12 seeds that completed the locked
# study, the median `taught - best control` is +0.023, shared by seeds 101, 111 and 114; 101 is the lowest.
SEED = 101
SAMPLE_FRAUD, SAMPLE_LEGIT = 20, 180     # ledger sample, drawn from the held-out test cases
SAMPLE_RNG = 7000 + SEED
MIN_LABELLED_FRAUDS = 30                 # below this the agent asks for advice
MAX_FLAG_RATE_FACTOR = 10                # advice may flag at most this many times the agent's old fraud rate
MIN_LABEL_AGREEMENT = 0.99               # advice must agree with the agent's own labelled cases this often
FLAG_AT = 0.5                            # a purchase is flagged when the model's fraud probability reaches this
TAG = "synapse row "


# ---------- splits: the same formulas as run_real_data_advice_v2.run_seed ----------

def split(d, seed):
    unit = d["new_unit"]
    units = np.unique(unit)
    order = np.random.default_rng(1000 + seed).permutation(units)
    n = len(order)
    t_units, s_units, e_units = order[: int(0.4 * n)], order[int(0.4 * n): int(0.6 * n)], order[int(0.6 * n):]
    s_rows = np.flatnonzero(np.isin(unit, s_units))
    e_rows = np.flatnonzero(np.isin(unit, e_units))
    stream = np.random.default_rng(1000 + seed).choice(s_rows, size=min(N_STREAM, len(s_rows)), replace=False)
    t_perm = np.random.default_rng(5000 + seed).permutation(t_units)
    fit_units, gate_units = t_perm[: int(0.75 * len(t_perm))], t_perm[int(0.75 * len(t_perm)):]
    return dict(
        fit_rows=np.flatnonzero(np.isin(unit, fit_units)), gate_rows=np.flatnonzero(np.isin(unit, gate_units)),
        n_fit_units=len(fit_units), e_rows=e_rows, labelled=stream[:N_LABELLED], advised=stream[N_LABELLED:],
    )


# ---------- step 1: choose the ledger sample ----------

def cmd_sample(_args):
    d = load_fraud()
    sp = split(d, SEED)
    amount = d["new_X"][:, d["features"].index("Amount")]
    rows = sp["e_rows"][amount[sp["e_rows"]] >= 0.01]      # Nessie needs at least one cent
    y = d["new_y"][rows]
    rng = np.random.default_rng(SAMPLE_RNG)
    fraud = rng.choice(rows[y == 1], size=min(SAMPLE_FRAUD, int((y == 1).sum())), replace=False)
    legit = rng.choice(rows[y == 0], size=SAMPLE_LEGIT, replace=False)
    picked = rng.permutation(np.concatenate([fraud, legit]))
    OUT.mkdir(parents=True, exist_ok=True)
    sample = dict(seed=SEED, rng=SAMPLE_RNG, note="held-out test cases of locked-study seed %d; frauds over-sampled "
                  "for display (real rate is about 1 in 600); no labels in this file" % SEED,
                  purchases=[dict(row=int(r), amount=round(float(amount[r]), 2)) for r in picked])
    (OUT / "sample.json").write_text(json.dumps(sample, indent=1))
    print(f"sample: {len(picked)} purchases ({len(fraud)} fraud, {len(legit)} legitimate) -> data/bank_demo/sample.json")


# ---------- step 2: Nessie (v2 API: https://v2.nessieisreal.com/docs) ----------

def nessie(method, path, key, body=None, tries=6):
    """One call to the Nessie v2 API. The key goes in the Authorization header, never in the URL."""
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "application/json",
               "User-Agent": "synapse-bank-demo"}
    for attempt in range(tries):
        req = urllib.request.Request(f"{NESSIE}{path}", data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read() or b"null")
        except urllib.error.HTTPError as e:
            text = e.read().decode(errors="replace")[:500]
            if e.code in (429, 500, 502, 503) and attempt < tries - 1:
                wait = e.headers.get("Retry-After", "")
                time.sleep(float(wait) if wait.replace(".", "", 1).isdigit() else 2 * (attempt + 1))
                continue
            raise SystemExit(f"Nessie {method} {path} -> HTTP {e.code}: {text}") from None
        except urllib.error.URLError as e:
            raise SystemExit(f"Could not reach Nessie at {NESSIE} ({e.reason}). Check NESSIE_BASE, your network, "
                             "and any VPN or proxy.") from None
        except json.JSONDecodeError:
            raise SystemExit(f"{NESSIE}{path} did not answer with JSON, so NESSIE_BASE is probably a web page, "
                             "not the API. Unset NESSIE_BASE to use the default.") from None


def cmd_sync(args):
    sample = json.loads((OUT / "sample.json").read_text())["purchases"]
    today = datetime.date.today().isoformat()
    key = os.environ.get("NESSIE_KEY")
    if args.offline or not key:
        if not args.offline:
            print("NESSIE_KEY is not set: writing the offline ledger instead.")
        purchases = [dict(_id=f"offline-{i}", amount=p["amount"], description=f"{TAG}{p['row']}", purchase_date=today,
                          status="pending") for i, p in enumerate(sample)]
        ledger = dict(source="offline", fetchedAt=today, purchases=purchases)
    else:
        cents = [int(round(p["amount"] * 100)) for p in sample]
        address = dict(street_number="1", street_name="Nassau St", city="Princeton", state="NJ", zip="08540")
        customer = nessie("POST", "/customers", key, dict(first_name="Synapse", last_name="Credit Union", address=address))["id"]
        account = nessie("POST", f"/customers/{customer}/accounts", key,
                         dict(type="Checking", nickname="Synapse demo ledger", balance_cents=sum(cents) + 1_000_000))["id"]
        merchant = nessie("POST", "/merchants", key, dict(name="Synapse demo merchant", category="retail"))["id"]
        for i, (p, c) in enumerate(zip(sample, cents), 1):
            nessie("POST", f"/accounts/{account}/purchases", key,
                   dict(merchant_id=merchant, amount_cents=c, description=f"{TAG}{p['row']}"))
            time.sleep(0.55)                              # the key allows 120 requests a minute
            if i % 25 == 0:
                print(f"posted {i}/{len(sample)}", flush=True)
        items = nessie("GET", f"/accounts/{account}/transactions?limit=5000", key).get("items") or []
        back = [dict(_id=t["id"], amount=t["amount_cents"] / 100, description=t["description"],
                     purchase_date=str(t.get("created_at", today))[:10], status=t.get("status"))
                for t in items if t.get("kind") == "purchase" and str(t.get("description", "")).startswith(TAG)]
        if len(back) != len(sample):
            raise SystemExit(f"posted {len(sample)} purchases but read back {len(back)}; not writing the ledger")
        ledger = dict(source="nessie", base=NESSIE, customerId=customer, accountId=account, merchantId=merchant,
                      fetchedAt=today, purchases=back)
    (OUT / "nessie_purchases.json").write_text(json.dumps(ledger, indent=1))
    print(f"ledger: {len(ledger['purchases'])} purchases from {ledger['source']} -> data/bank_demo/nessie_purchases.json")


# ---------- step 3: the agent ----------

def psi(old_scores, new_scores, bins=10):
    """Population stability index of the agent's fraud scores, old period against new stream."""
    edges = np.unique(np.quantile(old_scores, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(old_scores, edges)[0] / len(old_scores) + 1e-6
    q = np.histogram(new_scores, edges)[0] / len(new_scores) + 1e-6
    return float(np.sum((q - p) * np.log(q / p)))


def advice_checks(gate_margin, flag_rate, old_rate, agreement):
    return [
        dict(name="The bank's model beat the credit union's old model on the bank's own held-out cases",
             value=f"{gate_margin:+.3f} average precision (needs {GATE_MARGIN:+.2f})", passed=bool(gate_margin >= GATE_MARGIN)),
        dict(name="The share of purchases it flags is plausible",
             value=f"{flag_rate:.2%} flagged; the credit union's old fraud rate was {old_rate:.2%} (limit {MAX_FLAG_RATE_FACTOR}x)",
             passed=bool(0 < flag_rate <= MAX_FLAG_RATE_FACTOR * old_rate)),
        dict(name="It agrees with the credit union's labelled cases",
             value=f"{agreement:.1%} agreement (needs {MIN_LABEL_AGREEMENT:.0%})", passed=bool(agreement >= MIN_LABEL_AGREEMENT)),
    ]


def cmd_run(_args):
    ledger = json.loads((OUT / "nessie_purchases.json").read_text())
    d = load_fraud()
    sp = split(d, SEED)
    X, y, oX, oy = d["new_X"], d["new_y"], d["old_X"], d["old_y"]
    e_rows, labelled, advised = sp["e_rows"], sp["labelled"], sp["advised"]
    in_test = set(e_rows.tolist())
    rows = []
    for p in ledger["purchases"]:
        row = int(str(p["description"])[len(TAG):])
        if row not in in_test:
            raise SystemExit(f"purchase {p.get('_id')} points at row {row}, which is not a held-out test case")
        rows.append(row)
    rows = np.array(rows)
    steps = []

    # 1. observe
    total = sum(float(p["amount"]) for p in ledger["purchases"])
    steps.append(dict(id="observe", title="Read the ledger",
                      text=f"Pulled {len(rows)} purchases ({total:,.2f} in total) from "
                           f"{'the Nessie API' if ledger['source'] == 'nessie' else 'the offline ledger (Nessie was not used)'}."))

    # 2. where does the credit union stand? (everything from the old-only model is computed before any refit)
    base = Student(oX)
    old_only = base.fit(oX, oy)
    old_scores, stream_scores = old_only.score(oX), old_only.score(X[np.concatenate([labelled, advised])])
    gate_student = float(average_precision_score(y[sp["gate_rows"]], old_only.score(X[sp["gate_rows"]])))
    shift = psi(old_scores, stream_scores)
    lX, ly, aX = X[labelled], y[labelled], X[advised]
    known_frauds, old_rate = int(ly.sum()), float(oy.mean())
    ask = known_frauds < MIN_LABELLED_FRAUDS
    steps.append(dict(
        id="assess", title="Check what changed",
        text=f"The credit union's model was trained on day 1. Since then it has seen {len(labelled) + len(advised):,} new purchases, but true labels "
             f"for only {len(labelled)} of them, and {known_frauds} of those {'is' if known_frauds == 1 else 'are'} fraud. "
             f"Its fraud scores on the new purchases have shifted by PSI {shift:.3f} "
             f"({'small' if shift < 0.1 else 'moderate' if shift < 0.25 else 'large'}; reported, not used to decide).",
        decision=(f"Fewer than {MIN_LABELLED_FRAUDS} labelled frauds: too few to relearn from its own labels, so the credit union asks the bank for advice."
                  if ask else f"At least {MIN_LABELLED_FRAUDS} labelled frauds: the credit union relearns from its own labels and does not ask.")))

    # 3. the bank answers (its model and data never leave it)
    current, setting, _ = pick_and_fit_teacher(X[sp["fit_rows"]], y[sp["fit_rows"]], d["new_unit"][sp["fit_rows"]], SEED)
    gate_teacher = float(average_precision_score(y[sp["gate_rows"]], current.predict_proba(X[sp["gate_rows"]])[:, 1]))
    bits = (current.predict_proba(aX)[:, 1] >= 0.5).astype(int)
    check_bits = (current.predict_proba(lX)[:, 1] >= 0.5).astype(int)
    n_bits = len(bits) + len(check_bits)
    steps.append(dict(
        id="ask", title="Ask the bank",
        text=f"The bank has a differently built model (boosted trees; the credit union's is a logistic regression) trained on its own "
             f"new-period customers. It sent one bit per purchase, \"flag\" or \"allow\": {len(bits):,} advice bits "
             f"(as in the locked study) plus {len(check_bits)} on the credit union's labelled cases for checking, {-(-n_bits // 8):,} bytes in all. "
             f"It flagged {int(bits.sum()) + int(check_bits.sum())}. No model, no customer data."))

    # 4. check before trusting
    agreement = float((check_bits == ly).mean())
    checks = advice_checks(gate_teacher - gate_student, float(bits.mean()), old_rate, agreement)
    accept = ask and all(c["passed"] for c in checks)
    # the same checks on an out-of-date bank model, to show what they can and cannot catch
    old_units = np.unique(d["old_unit"])
    pick = np.random.default_rng(4000 + SEED).choice(old_units, size=min(sp["n_fit_units"], len(old_units)), replace=False)
    o_rows = np.flatnonzero(np.isin(d["old_unit"], pick))
    stale, _, _ = pick_and_fit_teacher(oX[o_rows], oy[o_rows], d["old_unit"][o_rows], SEED)
    stale_gate = float(average_precision_score(y[sp["gate_rows"]], stale.predict_proba(X[sp["gate_rows"]])[:, 1]))
    stale_bits = (stale.predict_proba(aX)[:, 1] >= 0.5).astype(int)
    stale_checks = advice_checks(stale_gate - gate_student, float(stale_bits.mean()), old_rate,
                                 float(((stale.predict_proba(lX)[:, 1] >= 0.5).astype(int) == ly).mean()))
    stale_rejected = not all(c["passed"] for c in stale_checks)
    steps.append(dict(
        id="check", title="Check the advice before trusting it", checks=checks,
        text=f"With {known_frauds} labelled fraud{'s' if known_frauds != 1 else ''} the credit union cannot measure the bank's accuracy itself, "
             "so it relies on the bank's own held-out check plus two sanity checks.",
        decision="All checks passed: the credit union accepts the advice." if accept else "A check failed (or no advice was asked for): the credit union keeps learning alone.",
        staleTeacher=dict(rejected=bool(stale_rejected), checks=stale_checks,
                          text="The same checks on an out-of-date bank model (trained on day 1 only): "
                               + ("rejected." if stale_rejected else "NOT rejected. These checks do not catch an out-of-date adviser."))))

    # 5. act: the two arms of the locked study, alone and taught
    eX, ey = X[e_rows], y[e_rows]
    alone_fit = base.fit(oX, oy, lX, ly)
    alone_test, alone_ledger = alone_fit.score(eX), alone_fit.score(X[rows])
    taught_fit = base.fit(oX, oy, np.vstack([lX, aX]), np.concatenate([ly, bits]))
    taught_test, taught_ledger = taught_fit.score(eX), taught_fit.score(X[rows])
    metrics = dict(alone=evaluate(ey, alone_test), taught=evaluate(ey, taught_test))
    final_ledger = taught_ledger if accept else alone_ledger
    actual = y[rows]
    n_fraud = int(actual.sum())

    def tally(scores):
        flagged = scores >= FLAG_AT
        top = np.argsort(-scores, kind="stable")[:n_fraud]
        return dict(flagged=int(flagged.sum()), fraudsFlagged=int((flagged & (actual == 1)).sum()),
                    falseAlarms=int((flagged & (actual == 0)).sum()), fraudsInTopN=int(actual[top].sum()))

    ledger_tally = dict(alone=tally(alone_ledger), taught=tally(taught_ledger), frauds=n_fraud, purchases=len(rows))
    a, t = ledger_tally["alone"], ledger_tally["taught"]
    steps.append(dict(
        id="act", title="Retrain and review the ledger",
        text=f"Retrained {'with the advice' if accept else 'on its own labels only'} and scored the {len(rows)} ledger purchases. "
             f"Of the {n_fraud} frauds, learning alone flags {a['fraudsFlagged']} (with {a['falseAlarms']} false alarms); "
             f"with advice it flags {t['fraudsFlagged']} (with {t['falseAlarms']} false alarms)."))

    # 6. report against the locked study
    saved_path = STUDY / "fraud_v2" / f"seed_{SEED}.json"
    matches = None
    if saved_path.exists():
        saved = json.loads(saved_path.read_text()).get("metrics", {})
        matches = bool(saved and all(abs(saved[k]["ap"] - metrics[k]["ap"]) < 1e-9 for k in ("alone", "taught")))
    summary = json.loads((STUDY / "fraud_v2_summary.json").read_text()) if (STUDY / "fraud_v2_summary.json").exists() else None
    steps.append(dict(
        id="report", title="Report",
        text=f"On all {len(e_rows):,} held-out purchases of this run, average precision is {metrics['alone']['ap']:.3f} alone and "
             f"{metrics['taught']['ap']:.3f} with advice."
             + ("" if matches is None else " These match the locked study's saved result for this seed." if matches
                else " WARNING: these do NOT match the locked study's saved result for this seed.")))

    purchases = [dict(id=p.get("_id"), amount=float(p["amount"]), date=p.get("purchase_date"), row=int(r),
                      alone=round(float(sa), 4), taught=round(float(st), 4), final=round(float(sf), 4),
                      flagged=bool(sf >= FLAG_AT), fraud=bool(f))
                 for p, r, sa, st, sf, f in zip(ledger["purchases"], rows, alone_ledger, taught_ledger, final_ledger, actual)]
    purchases.sort(key=lambda p: -p["final"])
    run = dict(
        kind="engineering demo, not a locked study", seed=SEED,
        seedRule="median `taught - best control` among the 12 completed seeds of the locked study (+0.023; seeds 101, 111, 114); lowest seed number",
        ledgerSource=ledger["source"], nessie={k: ledger.get(k) for k in ("base", "customerId", "accountId") if ledger.get(k)},
        settings=dict(minLabelledFrauds=MIN_LABELLED_FRAUDS, maxFlagRateFactor=MAX_FLAG_RATE_FACTOR,
                      minLabelAgreement=MIN_LABEL_AGREEMENT, flagAt=FLAG_AT),
        accepted=bool(accept), steps=steps, ledger=ledger_tally, testMetrics=metrics, matchesLockedStudy=matches,
        adviceBytes=-(-n_bits // 8), teacherSetting=setting,
        lockedStudy=None if summary is None else dict(
            seedsCompleted=summary["seedsPassingTeacherGate"], seedsRun=summary["seedsRun"],
            meanAP={k: summary["meanAP"][k] for k in ("alone", "taught", "fullLabels", "shuffled", "random", "stale")},
            meanRecallTop1={k: summary["meanRecallTop1"][k] for k in ("alone", "taught")},
            meanAdvantage=summary["mainTest"]["meanAdvantage"], bootstrap95=summary["mainTest"]["bootstrap95"],
            passed=summary["mainTest"]["passed"], adviceBytes=summary["adviceBytes"]),
        purchases=purchases,
    )
    (OUT / "agent_run.json").write_text(json.dumps(run, indent=1))
    for s in steps:
        print(f"\n[{s['title']}] {s['text']}")
        for c in s.get("checks", []):
            print(f"   {'PASS' if c['passed'] else 'FAIL'}  {c['name']}: {c['value']}")
        if "decision" in s:
            print(f"   -> {s['decision']}")
        if "staleTeacher" in s:
            print(f"   {s['staleTeacher']['text']}")
    print("\nwrote data/bank_demo/agent_run.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sample")
    sub.add_parser("sync").add_argument("--offline", action="store_true")
    sub.add_parser("run")
    sub.add_parser("all").add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.cmd in ("sample", "all"):
        cmd_sample(args)
    if args.cmd in ("sync", "all"):
        cmd_sync(args)
    if args.cmd in ("run", "all"):
        cmd_run(args)


if __name__ == "__main__":
    main()
