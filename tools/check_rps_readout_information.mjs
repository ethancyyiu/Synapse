// Prospective Path B information gate. No reward, learning, or action training.
// node tools/check_rps_readout_information.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { RpsBioRL } from '../js/rps.js';

const output = new URL('../data/rps_readout_information.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite information result');
const sub = JSON.parse(readFileSync(new URL('../data/subgraph.json', import.meta.url)));
const saved = JSON.parse(readFileSync(new URL('../data/rps_frozen_input.json', import.meta.url)));
const calibrationSeeds = [651, 652, 653], heldoutSeeds = [661, 662, 663], repeats = 12;
const calibration = saved.trials;
if (calibration.length !== 108 || calibration.some(r => !calibrationSeeds.includes(r.seed) ||
  !Number.isInteger(r.state) || r.state < 0 || r.state > 2 || r.rates.length !== 3 ||
  r.rates.some(x => !Number.isFinite(x) || x < 0))) throw Error('Invalid calibration source');
const feature = rates => rates.map(x => Math.log1p(x));
const centroids = [0, 1, 2].map(state => {
  const rows = calibration.filter(r => r.state === state);
  if (rows.length !== 36) throw Error('Unbalanced calibration source');
  return [0, 1, 2].map(j => rows.reduce((s, r) => s + feature(r.rates)[j], 0) / rows.length);
});
const predict = rates => {
  const x = feature(rates);
  const d = centroids.map(c => c.reduce((s, v, j) => s + (x[j] - v) ** 2, 0));
  return d.indexOf(Math.min(...d));
};
const trials = [];
for (const seed of heldoutSeeds) {
  const net = new Network(sub, wiring(sub, 'real', seed), { seed });
  const agent = new RpsBioRL(net, { seed });
  const frozen = net.m.slice();
  for (let repeat = 0; repeat < repeats; repeat++) for (let state = 0; state < 3; state++) {
    net.resetState();
    net.rand = rng(seed * 100000 + repeat + 1);
    agent.rand = rng(seed * 100000 + repeat + 70001);
    const input = net.laneVpn[state].slice(2);
    net.setDrive(input, 150);
    for (let t = 0; t < 240; t++) {
      if (t === 50) net.resetCounts();
      net.step();
    }
    net.setDrive(input, 0);
    agent.decide(net);
    const rates = [...agent.rates];
    trials.push({ seed, repeat, state, rates, predicted: predict(rates) });
  }
  if (net.m.some((v, i) => v !== frozen[i])) throw Error(`Frozen weights changed for ${seed}`);
}
if (trials.some(r => r.rates.some(x => !Number.isFinite(x) || x < 0))) throw Error('Invalid held-out rate');
const confusion = [0, 1, 2].map(s => [0, 1, 2].map(p =>
  trials.filter(r => r.state === s && r.predicted === p).length));
const correct = trials.filter(r => r.state === r.predicted).length;
const stateCorrect = confusion.map((row, s) => row[s]);
const seedCorrect = heldoutSeeds.map(seed => trials.filter(r => r.seed === seed && r.state === r.predicted).length);
const shuffleRand = rng(20261103), shuffledScores = [];
for (let rep = 0; rep < 1000; rep++) {
  let score = 0;
  for (const seed of heldoutSeeds) {
    const labels = trials.filter(r => r.seed === seed).map(r => r.state);
    for (let i = labels.length - 1; i > 0; i--) {
      const j = Math.floor(shuffleRand() * (i + 1));
      [labels[i], labels[j]] = [labels[j], labels[i]];
    }
    trials.filter(r => r.seed === seed).forEach((r, i) => { score += Number(r.predicted === labels[i]); });
  }
  shuffledScores.push(score);
}
const sorted = shuffledScores.toSorted((a, b) => a - b);
const shuffled95 = sorted[949];
const gates = { total: correct >= 87, eachState: stateCorrect.every(n => n >= 26),
  eachSeed: seedCorrect.every(n => n >= 27), shuffled: correct > shuffled95 };
const result = { command: 'node tools/check_rps_readout_information.mjs',
  design: { calibrationSeeds, heldoutSeeds, repeats, features: 'log1p of three action-group MBON rates at 240ms',
    classifier: 'pooled nearest Euclidean centroid; calibration only; tie order 0,1,2',
    balancedChance: 36, shuffleSeed: 20261103, shuffledPermutations: 1000 },
  centroids, confusion, correct, stateCorrect, seedCorrect, shuffled95,
  shuffledMean: shuffledScores.reduce((s, x) => s + x, 0) / shuffledScores.length,
  gates, passed: Object.values(gates).every(Boolean), trials };
writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({ correct, stateCorrect, seedCorrect, shuffled95, gates, passed: result.passed }));
