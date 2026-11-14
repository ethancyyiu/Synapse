"""Outcome-blind quantizer comparison on the saved raw seed-1 teacher spikes."""
import json
import math
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = json.loads((root / 'data/rps_channel_source_audit.json').read_text())
rows = source['rounds']

def summarize(values):
    counts = Counter(values)
    n = len(values)
    entropy = -sum((count / n) * math.log2(count / n) for count in counts.values())
    return {'patterns': len(counts), 'entropyBits': entropy,
            'modalShare': max(counts.values()) / n,
            'counts': {str(k): v for k, v in sorted(counts.items(), key=lambda x: str(x[0]))}}

methods = {
    'current_cap2_vector': [tuple(min(2, x) for x in row['raw']) for row in rows],
    'cap4_vector': [tuple(min(4, x) for x in row['raw']) for row in rows],
    'binary_nonzero_vector': [tuple(int(x > 0) for x in row['raw']) for row in rows],
    'binary_ge2_vector': [tuple(int(x >= 2) for x in row['raw']) for row in rows],
    'source0_nonzero_bit': [int(row['raw'][0] > 0) for row in rows],
    'source0_ge2_bit': [int(row['raw'][0] >= 2) for row in rows],
    'source0_vs_source1': [0 if row['raw'][0] > row['raw'][1]
                           else 1 if row['raw'][0] < row['raw'][1] else 'tie' for row in rows],
}
result = {'sourceSeed': source['seed'], 'rounds': len(rows),
          'sampledMs': source['sampledMs'],
          'note': 'Descriptive outcome-blind code statistics; no decoder or teaching outcome was tuned.',
          'methods': {name: summarize(values) for name, values in methods.items()}}
(root / 'data/rps_channel_normalization_audit.json').write_text(json.dumps(result, indent=2))
for name, stats in result['methods'].items():
    print(f"{name}: {stats['patterns']} patterns, entropy {stats['entropyBits']:.2f} bits, "
          f"modal share {stats['modalShare']:.1%}")
