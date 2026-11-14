"""Plot 24-round trailing win rates at every individual RPS round.

Run after the diagnostic simulations: python tools/plot_rps_diagnostics.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUTPUT = ROOT / 'docs' / 'rps-diagnostics'
OUTPUT.mkdir(parents=True, exist_ok=True)


def load(mode):
    path = DATA / f'rps_diagnostic_{mode}.json'
    return json.loads(path.read_text()) if path.exists() else None


def rolling(rewards, width=24):
    wins = np.asarray(rewards) == 1
    cumulative = np.r_[0, np.cumsum(wins)]
    end = np.arange(1, len(wins) + 1)
    start = np.maximum(0, end - width)
    rates = (cumulative[end] - cumulative[start]) / (end - start)
    rates[:width - 1] = np.nan
    return rates


def plot_runs(ax, runs, label, color, width=24):
    curves = np.stack([rolling(rewards, width) for rewards in runs])
    rounds = np.arange(1, curves.shape[1] + 1)
    ax.plot(rounds, curves.mean(axis=0), label=label, color=color, linewidth=2)
    ax.fill_between(rounds, curves.min(axis=0), curves.max(axis=0), color=color, alpha=0.12)


def finish(ax, max_round):
    for boundary in range(24, max_round, 24):
        ax.axvline(boundary, color='0.85', linewidth=0.5, zorder=0)
    ax.axhline(1 / 3, color='0.3', linestyle='--', linewidth=1, label='chance')
    ax.set(xlim=(1, max_round), ylim=(0, 1), xlabel='Individual RPS round', ylabel='Trailing 24-round win rate')
    ax.grid(axis='y', alpha=0.2)


current = load('current')
if current:
    colors = {'alone': '#222222', 'trained': '#16833d',
              'untrained': '#2765b8', 'shuffled_signal': '#c12b87'}
    labels = {'alone': 'alone', 'trained': 'trained teacher',
              'untrained': 'untrained teacher', 'shuffled_signal': 'shuffled signal'}
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), layout='constrained')
    for condition in colors:
        plot_runs(axes[0], [run['conditions'][condition]['rewards'] for run in current['runs']],
                  labels[condition], colors[condition])
    finish(axes[0], 120)
    axes[0].set_title('First five episodes (three seeds)')
    axes[0].legend(loc='lower right', fontsize=8)
    plot_runs(axes[1], [run['conditions']['alone']['rewards'] for run in current['runs']],
              'alone', colors['alone'])
    finish(axes[1], 480)
    axes[1].set_title('Alone through 20 episodes (three seeds)')
    axes[1].legend(loc='lower right', fontsize=8)
    fig.suptitle('Fly B learning by RPS round; bands show the range across seeds')
    fig.savefig(OUTPUT / 'round_learning.png', dpi=180)
    plt.close(fig)

hard = load('hard')
if hard:
    colors = {'long_sequence': '#814fba', 'history2': '#1c8b9e',
              'history3': '#ca6a20', 'switch': '#a8263b', 'probabilistic': '#5a6b22'}
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), layout='constrained')
    for ax, (kind, color) in zip(axes.flat, colors.items()):
        plot_runs(ax, [run['rewards'] for run in hard['runs'][kind]],
                  kind.replace('_', ' '), color)
        finish(ax, 480)
        if kind == 'switch':
            ax.axvline(240, color='red', linestyle=':', linewidth=1.5)
        ax.set_title(kind.replace('_', ' ').title())
    axes.flat[-1].set_visible(False)
    fig.suptitle('Fly B on harder opponents; mean and seed range at each individual round')
    fig.savefig(OUTPUT / 'harder_tasks.png', dpi=180)
    plt.close(fig)

positive = load('positive')
if positive:
    fig, ax = plt.subplots(figsize=(12, 4.5), layout='constrained')
    for condition, label, color in [('no_signal', 'no input', '#333333'),
                                    ('oracle_signal', 'oracle signal', '#16833d')]:
        plot_runs(ax, [run['conditions'][condition]['rewards'] for run in positive['runs']],
                  label, color)
    finish(ax, 480)
    ax.set_title('Separate positive control: previous-move input masked; oracle sends winning move')
    ax.legend(loc='lower right')
    fig.savefig(OUTPUT / 'oracle_control.png', dpi=180)
    plt.close(fig)

observed_oracle = load('oracle_observed')
if observed_oracle and current:
    fig, ax = plt.subplots(figsize=(12, 4.5), layout='constrained')
    plot_runs(ax, [run['conditions']['alone']['rewards'] for run in current['runs']],
              'alone', '#333333')
    plot_runs(ax, [run['rewards'] for run in observed_oracle['runs']],
              '+ oracle', '#16833d')
    finish(ax, 480)
    ax.set_title('Separate oracle control with the normal previous-move observation present')
    ax.legend(loc='lower right')
    fig.savefig(OUTPUT / 'oracle_with_observation.png', dpi=180)
    plt.close(fig)

print(f'Plots written to {OUTPUT}')
