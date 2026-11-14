"""Download the two public datasets used by the real-data advice tests into data/raw/ (git-ignored).

  - Credit-card fraud (Worldline / ULB, Dal Pozzolo et al.), OpenML dataset 1597, public.
  - PhysioNet / CinC Challenge 2019 sepsis data, hospital systems A and B, CC BY 4.0.

Run: python tools/fetch_real_data.py      (re-running skips files already present)
"""
import concurrent.futures as cf
import pathlib
import re
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"
FRAUD_URL = "https://openml.org/data/v1/download/1673544/creditcard.arff"
SEPSIS_BASE = "https://physionet.org/files/challenge-2019/1.0.0/training/"
HEADERS = {"User-Agent": "synapse-research-fetch"}


def get(url, tries=5):
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=120) as r:
                return r.read()
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def fetch_fraud():
    out = ROOT / "creditcard" / "creditcard.arff"
    if out.exists() and out.stat().st_size > 100_000_000:
        print("fraud: already present")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    data = get(FRAUD_URL)
    out.write_bytes(data)
    print(f"fraud: {len(data):,} bytes")


def fetch_sepsis(hospital):
    folder = ROOT / "sepsis" / f"training_set{hospital}"
    folder.mkdir(parents=True, exist_ok=True)
    listing = get(f"{SEPSIS_BASE}training_set{hospital}/").decode()
    names = sorted(set(re.findall(r"p\d+\.psv", listing)))
    todo = [n for n in names if not (folder / n).exists()]
    print(f"sepsis {hospital}: {len(names)} patients, {len(todo)} to download")

    def one(name):
        (folder / name).write_bytes(get(f"{SEPSIS_BASE}training_set{hospital}/{name}"))

    with cf.ThreadPoolExecutor(16) as pool:
        for i, _ in enumerate(pool.map(one, todo), 1):
            if i % 2000 == 0:
                print(f"sepsis {hospital}: {i}/{len(todo)}", flush=True)
    print(f"sepsis {hospital}: done")


if __name__ == "__main__":
    fetch_fraud()
    fetch_sepsis("A")
    fetch_sepsis("B")
