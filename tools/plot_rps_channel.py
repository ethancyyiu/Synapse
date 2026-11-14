"""Static scientific figures for the isolated communication calibration."""
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'docs' / 'rps-diagnostics'
screen = json.loads((DATA / 'rps_channel_calibration.json').read_text())
controls = json.loads((DATA / 'rps_channel_target_controls.json').read_text())
evaluations = [json.loads((DATA / name).read_text()) for name in (
    'rps_channel_evaluation.json',
    'rps_channel_evaluation_eight_120ms_300hz.json')]
confirmation = json.loads((DATA / 'rps_channel_confirmation.json').read_text())

fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)
configs = screen['configurations'] + controls['configurations']
names = [c['config']['name'].replace('_', ' ') for c in configs]
correct = [statistics.mean(r['evaluations']['correct']['accuracy'] for r in c['runs']) for c in configs]
shuffled = [statistics.mean(r['evaluations']['shuffled']['accuracy'] for r in c['runs']) for c in configs]
x = range(len(configs))
axes[0].bar([i - .19 for i in x], correct, width=.38, label='Correct signal', color='#176b52')
axes[0].bar([i + .19 for i in x], shuffled, width=.38, label='Shuffled', color='#b35c38')
axes[0].axhline(.5, ls='--', lw=1, color='black', label='Constant-action baseline')
axes[0].set_xticks(list(x), names, rotation=55, ha='right', fontsize=8)
axes[0].set_ylim(0, 1.08)
axes[0].set_ylabel('Frozen held-out accuracy')
axes[0].set_title('Calibration screen and target controls (2 seeds)')
axes[0].legend(fontsize=8)

for idx, evaluation in enumerate(evaluations):
    rows = evaluation['configuration']['runs']
    name = evaluation['selection']['name'].replace('_', ' ')
    offset = -.12 if idx == 0 else .12
    axes[1].scatter([r['seed'] + offset for r in rows],
                    [r['evaluations']['correct']['accuracy'] for r in rows],
                    label=name, s=35)
axes[1].axhline(.5, ls='--', lw=1, color='black')
axes[1].set_ylim(0, 1.08)
axes[1].set_xticks([r['seed'] for r in evaluations[0]['configuration']['runs']])
axes[1].set_xlabel('Independent evaluation seed')
axes[1].set_ylabel('Frozen held-out accuracy')
axes[1].set_title('Tied settings on 10 comparison seeds')
axes[1].legend(fontsize=8)
confirmed = confirmation['configuration']['runs']
for key, color in [('correct', '#176b52'), ('shuffled', '#b35c38'),
                   ('disconnected', '#6484ac')]:
    axes[2].scatter([r['seed'] for r in confirmed],
                    [r['evaluations'][key]['accuracy'] for r in confirmed],
                    label=key, s=30, color=color)
axes[2].axhline(.5, ls='--', lw=1, color='black')
axes[2].set(xlabel='Independent confirmation seed', ylabel='Frozen held-out accuracy',
            ylim=(0, 1.08), title='Locked 300 Hz channel: 10 final seeds')
axes[2].set_xticks([r['seed'] for r in confirmed])
axes[2].legend(fontsize=8)
fig.savefig(OUT / 'channel_calibration.png', dpi=170)
plt.close(fig)

chosen = confirmation['configuration']['runs']
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for row in chosen:
    axes[0].plot(range(1, 13), row['training'], color='#176b52', alpha=.18)
axes[0].plot(range(1, 13),
             [statistics.mean(r['training'][i] for r in chosen) for i in range(12)],
             color='#176b52', lw=2.5, label='Mean of 10 seeds')
axes[0].axhline(.5, ls='--', lw=1, color='black')
axes[0].set(xlabel='Training episode (24 rounds)', ylabel='Accuracy', ylim=(0, 1.05),
            title='Two-label learning with selected channel')
axes[0].legend(fontsize=8)

labels = ['Before', 'Trained + signal', 'Trained + shuffled', 'Disconnected', 'No-signal fly']
values = [[r['before']['accuracy'] for r in chosen]] + [
    [r['evaluations'][condition]['accuracy'] for r in chosen]
    for condition in ('correct', 'shuffled', 'disconnected', 'signalIndependent')]
axes[1].boxplot(values, tick_labels=labels, showmeans=True)
axes[1].axhline(.5, ls='--', lw=1, color='black')
axes[1].set(ylabel='Frozen held-out accuracy', ylim=(0, 1.08),
            title='10 independent evaluation seeds')
axes[1].tick_params(axis='x', labelrotation=40)
fig.savefig(OUT / 'channel_learning.png', dpi=170)
plt.close(fig)
