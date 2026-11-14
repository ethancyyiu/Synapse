// Fixed, label-free RPS MBON baseline from archived reward-free initial probes.
// node tools/calibrate_rps_rate_baseline.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';

const source = new URL('../data/rps_frozen_input.json', import.meta.url);
const output = new URL('../data/rps_rate_baseline.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite RPS rate calibration');
const data = JSON.parse(readFileSync(source));
const seeds = [651, 652, 653], states = [0, 1, 2];
const rows = data.trials;
const sourceValid = data.design?.reward === false && data.design?.teacher === false &&
  data.design?.weights?.startsWith('reproducible initial') && rows.length === 108 &&
  seeds.every(seed => states.every(state => rows.filter(r => r.seed === seed && r.state === state).length === 12)) &&
  rows.every(r => Array.isArray(r.rates) && r.rates.length === 3 &&
    r.rates.every(Number.isFinite) && !('reward' in r) && !('opponent' in r) && !('correct' in r));
if (!sourceValid) throw Error('Archived untrained, reward-free calibration provenance failed');
const baseline = Object.freeze(states.map(a =>
  rows.reduce((sum, r) => sum + r.rates[a], 0) / rows.length));
const winnerOpportunities = seeds.map(seed => {
  const sample = rows.filter(r => r.seed === seed);
  return { seed, counts: states.map(a => sample.filter(r => {
    const scores = r.rates.map((rate, k) => rate - baseline[k]);
    return scores[a] === Math.max(...scores);
  }).length) };
});
const valid = baseline.every(Number.isFinite) &&
  winnerOpportunities.every(row => row.counts.every(n => n > 0));
const result = { command: 'node tools/calibrate_rps_rate_baseline.mjs',
  source: 'data/rps_frozen_input.json',
  method: 'pooled mean per-action raw MBON rate over all 108 reward-free initial trials; ignore state, action, reward, and held-out results',
  baseline, sourceValid, winnerOpportunities, valid };
writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({ baseline, winnerOpportunities, valid }, null, 2));
