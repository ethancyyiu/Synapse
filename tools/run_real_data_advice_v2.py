"""Real-data advice after a change, v2 (fraud follow-up, sepsis). Locked protocol: docs/real-data-advice-v2-protocol.md.

Run: python tools/run_real_data_advice_v2.py [fraud|sepsis ...]
Writes data/real_data_advice/<dataset>_v2/seed_<n>.json and data/real_data_advice/<dataset>_v2_summary.json.
Data loading and the student are imported unchanged from the v1 runner.
"""
import json
import pathlib
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from run_real_data_advice import Student, bootstrap, evaluate, load_fraud, load_sepsis, sha  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "real_data_advice"
PROTOCOL = ROOT / "docs" / "real-data-advice-v2-protocol.md"
V1_RUNNER = ROOT / "tools" / "run_real_data_advice.py"
SEEDS = range(100, 115)
N_STREAM, N_LABELLED = 25000, 500
BOOT_RNG = {"fraud": 2026111411, "sepsis": 2026111412}
MIN, GATE_MARGIN, MIN_GATED = 0.02, 0.02, 10
GRID = [dict(learning_rate=lr, max_leaf_nodes=leaves) for lr in (0.05, 0.1) for leaves in (15, 31)]


def make_teacher(setting, seed):
    return HistGradientBoostingClassifier(max_iter=200, l2_regularization=1.0, early_stopping=False,
                                          random_state=seed, **setting)


def pick_and_fit_teacher(X, y, units, seed):
    """Choose a setting by 3-fold group CV on the teacher's own data, then fit it on all of that data."""
    folds = list(GroupKFold(n_splits=3).split(X, y, units))
    best, best_ap = None, -1.0
    for setting in GRID:
        aps = []
        for train, val in folds:
            model = make_teacher(setting, seed).fit(X[train], y[train])
            aps.append(average_precision_score(y[val], model.predict_proba(X[val])[:, 1]))
        if np.mean(aps) > best_ap:
            best, best_ap = setting, float(np.mean(aps))
    return make_teacher(best, seed).fit(X, y), best, best_ap


def run_seed(d, seed):
    unit = d["new_unit"]
    units = np.unique(unit)
    order = np.random.default_rng(1000 + seed).permutation(units)
    n = len(order)
    t_units, s_units, e_units = order[: int(0.4 * n)], order[int(0.4 * n): int(0.6 * n)], order[int(0.6 * n):]
    X, y = d["new_X"], d["new_y"]
    s_rows = np.flatnonzero(np.isin(unit, s_units))
    e_rows = np.flatnonzero(np.isin(unit, e_units))
    stream = np.random.default_rng(1000 + seed).choice(s_rows, size=min(N_STREAM, len(s_rows)), replace=False)
    labelled, advised = stream[:N_LABELLED], stream[N_LABELLED:]
    eX, ey = X[e_rows], y[e_rows]

    # teacher's own slice: 75% to choose settings and fit, 25% for the gate
    t_perm = np.random.default_rng(5000 + seed).permutation(t_units)
    fit_units, gate_units = t_perm[: int(0.75 * len(t_perm))], t_perm[int(0.75 * len(t_perm)):]
    fit_rows = np.flatnonzero(np.isin(unit, fit_units))
    gate_rows = np.flatnonzero(np.isin(unit, gate_units))
    t_rows = np.flatnonzero(np.isin(unit, t_units))
    current, setting, cv_ap = pick_and_fit_teacher(X[fit_rows], y[fit_rows], unit[fit_rows], seed)

    oX, oy = d["old_X"], d["old_y"]
    base = Student(oX)
    old_only = base.fit(oX, oy)
    gate_teacher = float(average_precision_score(y[gate_rows], current.predict_proba(X[gate_rows])[:, 1]))
    gate_student = float(average_precision_score(y[gate_rows], old_only.score(X[gate_rows])))
    gate = dict(teacherAP=gate_teacher, oldOnlyStudentAP=gate_student, margin=gate_teacher - gate_student,
                passed=gate_teacher >= gate_student + GATE_MARGIN, setting=setting, cvAP=cv_ap)
    if not gate["passed"]:
        return dict(status="stopped_at_teacher_gate", seed=seed, teacherGate=gate)
    old_only_scores = old_only.score(eX)

    old_units = np.unique(d["old_unit"])
    pick = np.random.default_rng(4000 + seed).choice(old_units, size=min(len(fit_units), len(old_units)), replace=False)
    o_rows = np.flatnonzero(np.isin(d["old_unit"], pick))
    stale_t, stale_setting, _ = pick_and_fit_teacher(oX[o_rows], oy[o_rows], d["old_unit"][o_rows], seed)

    aX = X[advised]
    bits = (current.predict_proba(aX)[:, 1] >= 0.5).astype(int)
    stale_bits = (stale_t.predict_proba(aX)[:, 1] >= 0.5).astype(int)
    shuffled = np.random.default_rng(2000 + seed).permutation(bits)
    rand_bits = (np.random.default_rng(3000 + seed).random(len(bits)) < bits.mean()).astype(int)
    lX, ly = X[labelled], y[labelled]

    def arm(extra):
        if extra is None:
            return base.fit(oX, oy, lX, ly).score(eX)
        return base.fit(oX, oy, np.vstack([lX, aX]), np.concatenate([ly, extra])).score(eX)

    scores = {
        "alone": arm(None), "taught": arm(bits), "shuffled": arm(shuffled), "random": arm(rand_bits),
        "stale": arm(stale_bits), "fullLabels": arm(y[advised]),
        "oldOnlyStudent": old_only_scores,
        "newSiteStudent": base.fit(oX, oy, X[t_rows], y[t_rows]).score(eX),
        "teacher": current.predict_proba(eX)[:, 1], "staleTeacher": stale_t.predict_proba(eX)[:, 1],
    }
    metrics = {k: evaluate(ey, v) for k, v in scores.items()}
    ap = {k: v["ap"] for k, v in metrics.items()}
    return dict(
        status="completed", seed=seed, teacherGate=gate, staleTeacherSetting=stale_setting, metrics=metrics,
        advantage=ap["taught"] - max(ap["alone"], ap["shuffled"], ap["random"]),
        change=ap["newSiteStudent"] - ap["oldOnlyStudent"], headroom=ap["fullLabels"] - ap["alone"],
        adviceBits=int(len(bits)), adviceBytes=int(len(bits) // 8), teacherFlagRate=float(bits.mean()),
        teacherBitAccuracy=float((bits == y[advised]).mean()),
        teacherBitRecall=float(bits[y[advised] == 1].mean()) if y[advised].sum() else None,
        labelledPositives=int(ly.sum()), advisedTruePositives=int(y[advised].sum()),
        counts=dict(teacherUnits=int(len(t_units)), streamUnits=int(len(s_units)), testUnits=int(len(e_units)),
                    testCases=int(len(e_rows)), testPositives=int(ey.sum()), oldCases=int(len(oy))),
    )


def main(names):
    hashes = dict(protocol=sha(PROTOCOL), runner=sha(__file__), v1Runner=sha(V1_RUNNER))
    for name in names:
        d = load_fraud() if name == "fraud" else load_sepsis()
        folder = OUT / f"{name}_v2"
        folder.mkdir(parents=True, exist_ok=True)
        seeds = []
        for seed in SEEDS:
            path = folder / f"seed_{seed}.json"
            if path.exists():  # write-once
                seeds.append(json.loads(path.read_text()))
                continue
            r = run_seed(d, seed)
            r["hashes"] = hashes
            path.write_text(json.dumps(r, indent=1))
            seeds.append(r)
            g = r["teacherGate"]
            line = f"{name} seed {seed}: gate {'pass' if g['passed'] else 'STOP'} (teacher {g['teacherAP']:.3f} vs student {g['oldOnlyStudentAP']:.3f})"
            if r["status"] == "completed":
                m = r["metrics"]
                line += f"; taught {m['taught']['ap']:.3f} alone {m['alone']['ap']:.3f} adv {r['advantage']:+.3f}"
            print(line, flush=True)
        done = [s for s in seeds if s["status"] == "completed"]
        arms = ["alone", "taught", "shuffled", "random", "stale", "fullLabels", "oldOnlyStudent", "newSiteStudent", "teacher", "staleTeacher"]
        summary = dict(dataset=name, version=2, hashes=hashes, seedsRun=len(seeds), seedsPassingTeacherGate=len(done),
                       gateMargins=[s["teacherGate"]["margin"] for s in seeds])
        if done:
            adv = [s["advantage"] for s in done]
            ci = bootstrap(adv, BOOT_RNG[name])
            change = float(np.mean([s["change"] for s in done]))
            headroom = float(np.mean([s["headroom"] for s in done]))
            summary.update(
                changeTest=dict(meanChange=change, passed=change >= MIN),
                headroomTest=dict(meanHeadroom=headroom, passed=headroom >= MIN),
                mainTest=dict(meanAdvantage=float(np.mean(adv)), bootstrap95=ci,
                              passed=len(done) >= MIN_GATED and change >= MIN and headroom >= MIN and ci[0] > MIN),
                seedsAdvantageAtMostZero=int(sum(a <= 0 for a in adv)),
                meanAP={a: float(np.mean([s["metrics"][a]["ap"] for s in done])) for a in arms},
                meanRecallTop1={a: float(np.mean([s["metrics"][a]["recallTop1"] for s in done])) for a in arms},
                meanTaughtMinusStale=float(np.mean([s["metrics"]["taught"]["ap"] - s["metrics"]["stale"]["ap"] for s in done])),
                meanTeacherBitRecall=float(np.mean([s["teacherBitRecall"] for s in done if s["teacherBitRecall"] is not None])),
                adviceBytes=done[0]["adviceBytes"],
            )
        else:
            summary.update(mainTest=dict(passed=False, reason="fewer than 10 seeds passed the teacher gate"))
        if len(done) < MIN_GATED:
            summary["mainTest"]["passed"] = False
        (OUT / f"{name}_v2_summary.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or ["fraud", "sepsis"])
