"""Real-data advice after a change (fraud, sepsis). Locked protocol: docs/real-data-advice-protocol.md.

Run: python tools/fetch_real_data.py  then  python tools/run_real_data_advice.py [fraud|sepsis ...]
Writes data/real_data_advice/<dataset>/seed_<n>.json and data/real_data_advice/<dataset>_summary.json.
Uses ordinary ML models; no connectome, spikes, or flies are involved.
"""
import hashlib
import io
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "real_data_advice"
PROTOCOL = ROOT / "docs" / "real-data-advice-protocol.md"
SEEDS = range(15)
N_STREAM, N_LABELLED, OLD_WEIGHT = 5000, 500, 1000.0
BOOT_RNG = {"fraud": 2026111401, "sepsis": 2026111402}
CHANGE_MIN, PASS_MIN = 0.02, 0.02


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


# ---------- data ----------

def load_fraud():
    text = (RAW / "creditcard" / "creditcard.arff").read_text()
    header, body = text.split("@data", 1) if "@data" in text else text.split("@DATA", 1)
    cols = [line.split()[1].strip("'\"") for line in header.splitlines() if line.lower().startswith("@attribute")]
    df = pd.read_csv(io.StringIO(body), header=None, names=cols, quotechar="'")
    df["Class"] = df["Class"].astype(str).str.strip("'\"").astype(int)
    features = [c for c in cols if c not in ("Time", "Class")]
    old = df[df["Time"] < 86400]
    new = df[df["Time"] >= 86400]
    # one "unit" per transaction
    return dict(features=features, old_X=old[features].to_numpy(float), old_y=old["Class"].to_numpy(int),
                new_X=new[features].to_numpy(float), new_y=new["Class"].to_numpy(int),
                new_unit=np.arange(len(new)), old_unit=np.arange(len(old)))


SEPSIS_DROP = {"Unit1", "Unit2", "SepsisLabel"}


def load_sepsis_site(site):
    cache = RAW / "sepsis" / f"set{site}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    frames = []
    for i, path in enumerate(sorted((RAW / "sepsis" / f"training_set{site}").glob("p*.psv"))):
        f = pd.read_csv(path, sep="|")
        f = f.ffill()  # carry values forward within this patient only
        f["patient"] = int(path.stem[1:])
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(cache)
    return df


def load_sepsis():
    a, b = load_sepsis_site("A"), load_sepsis_site("B")
    features = [c for c in a.columns if c not in SEPSIS_DROP | {"patient"}]
    return dict(features=features, old_X=a[features].to_numpy(float), old_y=a["SepsisLabel"].to_numpy(int),
                new_X=b[features].to_numpy(float), new_y=b["SepsisLabel"].to_numpy(int),
                new_unit=b["patient"].to_numpy(), old_unit=a["patient"].to_numpy())


# ---------- models ----------

class Student:
    """Logistic regression with imputation and scaling fitted on its own old data."""

    def __init__(self, old_X):
        self.median = np.nanmedian(old_X, axis=0)
        self.median = np.where(np.isnan(self.median), 0.0, self.median)
        filled = self._fill(old_X)
        self.mean, self.std = filled.mean(0), filled.std(0)
        self.std[self.std == 0] = 1.0

    def _fill(self, X):
        return np.where(np.isnan(X), self.median, X)

    def transform(self, X):
        return np.hstack([(self._fill(X) - self.mean) / self.std, np.isnan(X).astype(float)])

    def fit(self, old_X, old_y, X=None, y=None):
        parts_X, parts_y = [self.transform(old_X)], [old_y]
        weights = [np.full(len(old_y), OLD_WEIGHT / len(old_y))]
        if X is not None and len(y):
            parts_X.append(self.transform(X)); parts_y.append(y); weights.append(np.ones(len(y)))
        self.model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
        self.model.fit(np.vstack(parts_X), np.concatenate(parts_y), sample_weight=np.concatenate(weights))
        return self

    def score(self, X):
        return self.model.predict_proba(self.transform(X))[:, 1]


def teacher(X, y, seed):
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed).fit(X, y)


def recall_top1(y, s):
    k = max(1, int(round(0.01 * len(y))))
    top = np.argsort(-s, kind="stable")[:k]
    return float(y[top].sum() / max(1, y.sum()))


def evaluate(y, s):
    return dict(ap=float(average_precision_score(y, s)), recallTop1=recall_top1(y, s))


# ---------- one seed ----------

def run_seed(d, seed):
    units = np.unique(d["new_unit"])
    order = np.random.default_rng(1000 + seed).permutation(units)
    n = len(order)
    t_units, s_units, e_units = order[: int(0.4 * n)], order[int(0.4 * n): int(0.6 * n)], order[int(0.6 * n):]
    unit = d["new_unit"]
    t_rows = np.flatnonzero(np.isin(unit, t_units))
    s_rows = np.flatnonzero(np.isin(unit, s_units))
    e_rows = np.flatnonzero(np.isin(unit, e_units))
    rng = np.random.default_rng(1000 + seed)
    stream = rng.choice(s_rows, size=N_STREAM, replace=False)
    labelled, advised = stream[:N_LABELLED], stream[N_LABELLED:]

    X, y = d["new_X"], d["new_y"]
    tX, ty, eX, ey = X[t_rows], y[t_rows], X[e_rows], y[e_rows]

    current = teacher(tX, ty, seed)
    # out-of-date teacher: old sample of the same size (rows for fraud, patients for sepsis)
    old_units = np.unique(d["old_unit"])
    pick = np.random.default_rng(4000 + seed).choice(old_units, size=min(len(t_units), len(old_units)), replace=False)
    o_rows = np.flatnonzero(np.isin(d["old_unit"], pick))
    stale_t = teacher(d["old_X"][o_rows], d["old_y"][o_rows], seed)

    bits = (current.predict_proba(X[advised])[:, 1] >= 0.5).astype(int)
    stale_bits = (stale_t.predict_proba(X[advised])[:, 1] >= 0.5).astype(int)
    shuffled = np.random.default_rng(2000 + seed).permutation(bits)
    rand_bits = (np.random.default_rng(3000 + seed).random(len(bits)) < bits.mean()).astype(int)

    oX, oy = d["old_X"], d["old_y"]
    base = Student(oX)
    lX, ly, aX = X[labelled], y[labelled], X[advised]

    def arm(extra_bits):
        if extra_bits is None:
            return base.fit(oX, oy, lX, ly).score(eX)
        return base.fit(oX, oy, np.vstack([lX, aX]), np.concatenate([ly, extra_bits])).score(eX)

    scores = {
        "alone": arm(None),
        "taught": arm(bits),
        "shuffled": arm(shuffled),
        "random": arm(rand_bits),
        "stale": arm(stale_bits),
        "fullLabels": arm(y[advised]),
        "oldOnlyStudent": base.fit(oX, oy).score(eX),
        "newSiteStudent": base.fit(oX, oy, tX, ty).score(eX),
        "teacher": current.predict_proba(eX)[:, 1],
        "staleTeacher": stale_t.predict_proba(eX)[:, 1],
    }
    result = {k: evaluate(ey, v) for k, v in scores.items()}
    ap = {k: v["ap"] for k, v in result.items()}
    return dict(
        status="completed", seed=seed, metrics=result,
        advantage=ap["taught"] - max(ap["alone"], ap["shuffled"], ap["random"]),
        change=ap["newSiteStudent"] - ap["oldOnlyStudent"],
        adviceBits=int(len(bits)), adviceBytes=int(len(bits) // 8), teacherFlagRate=float(bits.mean()),
        staleFlagRate=float(stale_bits.mean()), labelledPositives=int(ly.sum()), advisedTruePositives=int(y[advised].sum()),
        counts=dict(teacherUnits=int(len(t_units)), streamUnits=int(len(s_units)), testUnits=int(len(e_units)),
                    testCases=int(len(e_rows)), testPositives=int(ey.sum()), oldCases=int(len(oy))),
    )


def bootstrap(values, seed):
    rng = np.random.default_rng(seed)
    v = np.asarray(values)
    boots = np.sort(v[rng.integers(0, len(v), size=(10000, len(v)))].mean(1))
    return [float(boots[249]), float(boots[9749])]


def main(names):
    hashes = dict(protocol=sha(PROTOCOL), runner=sha(__file__))
    for name in names:
        d = load_fraud() if name == "fraud" else load_sepsis()
        folder = OUT / name
        folder.mkdir(parents=True, exist_ok=True)
        seeds = []
        for seed in SEEDS:
            path = folder / f"seed_{seed}.json"
            if path.exists():  # write-once: never rerun a finished seed
                seeds.append(json.loads(path.read_text()))
                continue
            r = run_seed(d, seed)
            r["hashes"] = hashes
            path.write_text(json.dumps(r, indent=1))
            seeds.append(r)
            m = r["metrics"]
            print(f"{name} seed {seed}: taught {m['taught']['ap']:.3f} alone {m['alone']['ap']:.3f} "
                  f"adv {r['advantage']:+.3f} change {r['change']:+.3f}", flush=True)
        done = [s for s in seeds if s["status"] == "completed"]
        adv = [s["advantage"] for s in done]
        ci = bootstrap(adv, BOOT_RNG[name])
        change = float(np.mean([s["change"] for s in done]))
        arms = ["alone", "taught", "shuffled", "random", "stale", "fullLabels", "oldOnlyStudent", "newSiteStudent", "teacher", "staleTeacher"]
        summary = dict(
            dataset=name, seeds=len(done), hashes=hashes,
            changeTest=dict(meanChange=change, passed=change >= CHANGE_MIN),
            mainTest=dict(meanAdvantage=float(np.mean(adv)), bootstrap95=ci, passed=len(done) == 15 and ci[0] > PASS_MIN),
            seedsAdvantageAtMostZero=int(sum(a <= 0 for a in adv)),
            meanAP={a: float(np.mean([s["metrics"][a]["ap"] for s in done])) for a in arms},
            meanRecallTop1={a: float(np.mean([s["metrics"][a]["recallTop1"] for s in done])) for a in arms},
            meanTaughtMinusStale=float(np.mean([s["metrics"]["taught"]["ap"] - s["metrics"]["stale"]["ap"] for s in done])),
            meanFullLabelsMinusTaught=float(np.mean([s["metrics"]["fullLabels"]["ap"] - s["metrics"]["taught"]["ap"] for s in done])),
            adviceBytes=done[0]["adviceBytes"],
        )
        (OUT / f"{name}_summary.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or ["fraud", "sepsis"])
