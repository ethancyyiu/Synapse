"""Round-level plots for neural-message strategy-switch controls."""
import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
final = len(sys.argv) > 1 and sys.argv[1] == 'final'
data = json.loads((root / ('data/rps_teaching_switch_postonly_final.json' if final
                          else 'data/rps_teaching_switch_postonly.json')).read_text())
runs = data['runs']
names = ['alone', 'trained', 'untrained', 'shuffled']
colors = {'alone': '#444c55', 'trained': '#176b52', 'untrained': '#6484ac', 'shuffled': '#b35c38'}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
for name in names:
    curves = []
    for run in runs:
        wins = [int(e['reward'] == 1) for e in run['conditions'][name]['events']]
        curves.append([statistics.mean(wins[max(0, i - 23):i + 1]) for i in range(len(wins))])
    means = [statistics.mean(c[i] for c in curves) for i in range(len(curves[0]))]
    axes[0].plot(range(1, 481), means, color=colors[name], label=name, lw=1.6)
axes[0].axvline(240.5, ls='--', color='black', lw=1)
axes[0].axhline(.5, ls=':', color='black', lw=1)
axes[0].set(xlabel='Individual RPS round', ylabel='Trailing 24-round win fraction',
            ylim=(0, 1), title='Signals enabled only after strategy switch')
axes[0].legend(fontsize=8)

xs = range(len(names))
means = [statistics.mean(r['conditions'][name]['postSwitchFirst3'] for r in runs) for name in names]
axes[1].bar(xs, means, color=[colors[name] for name in names], width=.55)
for i, name in enumerate(names):
    axes[1].scatter([i] * len(runs), [r['conditions'][name]['postSwitchFirst3'] for r in runs],
                    color='black', s=17, zorder=3)
axes[1].set_xticks(list(xs), names, rotation=25)
axes[1].set(ylabel='Win fraction, first 72 post-switch rounds', ylim=(0, .7),
            title='Early adaptation (3 seeds)')
fig.savefig(root / ('docs/rps-diagnostics/teaching_switch_final.png' if final
                    else 'docs/rps-diagnostics/teaching_switch.png'), dpi=170)
plt.close(fig)
