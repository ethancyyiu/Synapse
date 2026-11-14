"""Scientific figure for the 40-episode held-out RPS plateau diagnosis."""
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
rows = json.loads((root / 'data/rps_plateau_diagnostics.json').read_text())['runs']
out = root / 'docs/rps-diagnostics/plateau_analysis.png'
fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)

for row in rows:
    axes[0].plot(range(1, 41), row['curve'], alpha=.22, color='#176b52')
axes[0].plot(range(1, 41),
             [statistics.mean(r['curve'][i] for r in rows) for i in range(40)],
             color='#176b52', lw=2, label='Mean of 3 seeds')
axes[0].set(xlabel='Training episode (24 rounds)', ylabel='Win fraction', ylim=(0, 1),
            title='RPS learning through 40 episodes')
axes[0].legend(fontsize=8)

names = ['Random', 'Majority', 'Untrained fly', 'Trained fly', 'No MBON noise', 'Lookup']
keys = [lambda r: r['baseline']['random'], lambda r: r['baseline']['majority'],
        lambda r: r['noLearning']['winRate'], lambda r: r['trained']['winRate'],
        lambda r: r['noNoise']['winRate'], lambda r: r['baseline']['lookup']]
axes[1].bar(range(len(keys)), [statistics.mean(f(r) for r in rows) for f in keys],
            color=['#858b91', '#858b91', '#b35c38', '#176b52', '#6b86a4', '#303f4f'])
axes[1].set_xticks(range(len(keys)), names, rotation=40, ha='right', fontsize=8)
axes[1].set(ylabel='Held-out win fraction', ylim=(0, 1), title='Same opponent rule, new sequences')

for obs in range(3):
    counts = [sum(next(x for x in r['trained']['byObservation']
                       if x['observation'] == obs)['actionCounts'][a] for r in rows)
              for a in range(3)]
    total = sum(counts)
    axes[2].bar([obs - .24, obs, obs + .24], [v / total for v in counts], width=.22,
                color=['#6484ac', '#a4628b', '#176b52'])
axes[2].set_xticks(range(3), ['Previous 0', 'Previous 1', 'Previous 2'])
axes[2].set(ylabel='Fraction of actions', ylim=(0, 1.05),
            title='Trained fly never chooses action 1')
axes[2].legend(['Action 0', 'Action 1', 'Action 2'], fontsize=8)
fig.savefig(out, dpi=170)
plt.close(fig)
