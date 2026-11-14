"""Information audit for actual teacher neural messages on the switch task."""
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL = len(sys.argv) > 1 and sys.argv[1] == 'final'
study = json.loads((ROOT / ('data/rps_teaching_switch_final.json' if FINAL
                            else 'data/rps_teaching_switch.json')).read_text())
runs = study['runs']

def conditional_information(rows):
    groups = defaultdict(list)
    for row in rows:
        if row['previousOpponent'] is not None:
            groups[(row['phase'], row['previousOpponent'])].append(row)
    total = sum(len(group) for group in groups.values())
    def entropy(labels):
        counts = Counter(labels)
        return -sum((n / len(labels)) * math.log2(n / len(labels)) for n in counts.values())
    information = 0.0
    for group in groups.values():
        by_signal = defaultdict(list)
        for row in group:
            by_signal[row['label']].append(row['nextOpponent'])
        information += (len(group) / total) * (entropy([r['nextOpponent'] for r in group])
            - sum((len(moves) / len(group)) * entropy(moves) for moves in by_signal.values()))
    return information

def flatten(key):
    return [dict(sample, phase=0 if ep < study['switchEpisode'] - 1 else 1)
            for run in runs for ep, samples in enumerate(run[key]) for sample in samples]

rand = random.Random(73211)
out = {'seeds': [run['seed'] for run in runs], 'permutations': 500, 'conditions': {}}
for condition, key in [('trained', 'teacherSignals'), ('untrained', 'untrainedSignals')]:
    rows = flatten(key)
    observed = conditional_information(rows)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[(row['phase'], row['previousOpponent'])].append(i)
    null = []
    for _ in range(500):
        shuffled = [dict(row) for row in rows]
        for indices in groups.values():
            labels = [shuffled[i]['label'] for i in indices]
            rand.shuffle(labels)
            for i, label in zip(indices, labels):
                shuffled[i]['label'] = label
        null.append(conditional_information(shuffled))
    out['conditions'][condition] = {
        'n': len(rows), 'noninitialN': sum(r['previousOpponent'] is not None for r in rows),
        'labelCounts': dict(Counter(row['label'] for row in rows)),
        'conditionalNextMoveBits': observed,
        'permutationMedianBits': sorted(null)[len(null) // 2],
        'permutationP': (1 + sum(value >= observed for value in null)) / (len(null) + 1),
    }
out['trainedUntrainedAgreement'] = sum(a['label'] == b['label']
    for run in runs for ep_a, ep_b in zip(run['teacherSignals'], run['untrainedSignals'])
    for a, b in zip(ep_a, ep_b)) / sum(len(ep) for run in runs for ep in run['teacherSignals'])
(ROOT / ('data/rps_teaching_signal_information_final.json' if FINAL
         else 'data/rps_teaching_signal_information.json')).write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
