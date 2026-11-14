// Diagnostic replay: which action groups get chosen and plastic updates?
// Run: node tools/probe_rps_action_updates.mjs
import { readFileSync, writeFileSync } from 'node:fs';
import { Network, wiring } from '../js/sim.js';
import { Opponent, RpsBioRL, RpsEpisode } from '../js/rps.js';

const sub = JSON.parse(readFileSync(new URL('../data/subgraph.json', import.meta.url)));
const OUTPUT = new URL('../data/rps_plateau_action_updates.json', import.meta.url);
const seeds = [1, 2, 3];
const results = [];
for (const seed of seeds) {
  const studentSeed = seed + 500;
  const net = new Network(sub, wiring(sub, 'real', studentSeed), { seed: studentSeed });
  const agent = new RpsBioRL(net, { seed: studentSeed });
  const episodes = [];
  for (let ep = 0; ep < 40; ep++) {
    const run = new RpsEpisode(net, agent,
      new Opponent({ strategy: 'previous_opponent', offset: 1, noise: 0.1, seed: seed * 1009 + ep }),
      { rounds: 24 }).run();
    const byObservation = [0, 1, 2].map(obs => [0, 1, 2].map(action =>
      run.events.filter(e => e.observation.previousOpponent === obs && e.action === action).length));
    episodes.push({ episode: ep + 1, actionCounts: [0, 1, 2].map(action =>
      run.events.filter(e => e.action === action).length), byObservation,
      winRate: run.winRate });
  }
  const groupWeights = net.plastic.map((ids, group) => {
    const values = Array.from(ids, i => net.m[i]);
    return { group, edges: values.length, changed: values.filter(v => v !== 1).length,
      atZero: values.filter(v => v === 0).length, atMax: values.filter(v => v === 4).length,
      mean: values.reduce((sum, v) => sum + v, 0) / values.length };
  });
  results.push({ seed, episodes, groupWeights });
  writeFileSync(OUTPUT, JSON.stringify({ seeds, episodes: 40, roundsPerEpisode: 24, results }, null, 2));
  console.log(`seed ${seed}: actions ${[0, 1, 2].map(a => episodes.reduce((n, e) => n + e.actionCounts[a], 0))}, ` +
    `changed edges ${groupWeights.map(g => g.changed)}`);
}
