// Frozen input-layer fidelity check for the fixed graded four-MBON interface.
// node tools/check_two_cue_population_kc.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';

const output = new URL('../data/two_cue_population_kc_fixed_rng.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite KC diagnostic');
const sub = JSON.parse(readFileSync(new URL('../data/subgraph.json', import.meta.url)));
const ids = [2146, 2134, 2124, 2122];
const targets = [[506, 545], [113, 1421], [1319, 2016], [818, 1462]];
if (targets.flat().some(i => sub.neurons.role[i] !== 'kc')) throw Error('Target is not KC');
const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;
const sources = Object.fromEntries([501, 502, 505, 506].map(seed => [seed,
  JSON.parse(readFileSync(new URL(`../data/two_cue_source_02_${seed}_lanes02.json`, import.meta.url))).source]));
function rows(seed, trained, test = null) {
  const source = sources[seed], indices = ids.map(id => source.sourceIds.indexOf(id));
  if (indices.includes(-1)) throw Error(`Missing source neuron: ${seed}`);
  return [0, 1].flatMap(mapping => {
    const run = source.runs.find(r => r.mapping === mapping && r.trained === trained);
    if (!run || trained && mean(run.frozenAccuracy) < .75) throw Error(`Source gate: ${seed}`);
    return run.samples.filter(s => s.cue === 1 && (test === null || s.test === test))
      .map(s => ({ teacher: seed, mapping, test: s.test, sourceRound: s.round,
        trace: indices.map(i => s.trace[i]) }));
  });
}
const calibration = [501, 502].flatMap(seed => rows(seed, true));
const ceilings = ids.map((_, i) => {
  const sorted = calibration.map(r => r.trace[i]).sort((a, b) => a - b);
  return sorted[Math.ceil(.95 * sorted.length) - 1];
});
const archived = JSON.parse(readFileSync(new URL('../data/two_cue_graded_population_decode_02_203_lanes02.json', import.meta.url))).gradedPopulationDecode;
if (ceilings.some((x, i) => Math.abs(x - archived.calibrationCeilings[i]) > 1e-12) ||
    JSON.stringify(targets) !== JSON.stringify(archived.setting.messageTargets))
  throw Error('Fixed interface does not match archived runner');
const net = new Network(sub, wiring(sub, 'real', 703), { seed: 703 });
const weights = net.m.slice();
function measure(row, connected = true) {
  net.resetState();
  net.rand = rng(703000000 + row.teacher * 10000 + row.test * 100 + row.sourceRound);
  if (connected) targets.forEach((group, i) => net.setDrive(group,
    300 * Math.min(1, Math.max(0, row.trace[i] / ceilings[i]))));
  for (let t = 0; t < 40; t++) net.step();
  return targets.map(group => group.reduce((sum, i) => sum + net.count[i], 0));
}
const fit = calibration.map(r => ({ ...r, kc: measure(r) }));
const scale = ids.map((_, i) => {
  const xs = fit.map(r => r.kc[i]), center = mean(xs);
  return { mean: center, sd: Math.sqrt(mean(xs.map(x => (x - center) ** 2))) || 1 };
});
const vector = r => r.kc.map((x, i) => (x - scale[i].mean) / scale[i].sd);
const centroids = [0, 1].map(mapping => ids.map((_, i) =>
  mean(fit.filter(r => r.mapping === mapping).map(r => vector(r)[i]))));
function predict(row) {
  const x = vector(row);
  const distance = m => x.reduce((sum, value, i) => sum + (value - centroids[m][i]) ** 2, 0);
  return distance(0) <= distance(1) ? 0 : 1;
}
const assess = sample => ({ score: sample.filter(r => predict(r) === r.mapping).length,
  byMapping: [0, 1].map(m => sample.filter(r => r.mapping === m && predict(r) === m).length) });
const evaluation = Object.fromEntries([505, 506].map(seed => {
  const trained = rows(seed, true, 2).map(r => ({ ...r, kc: measure(r) }));
  const untrained = rows(seed, false, 2).map(r => ({ ...r, kc: measure(r) }));
  const disconnected = rows(seed, true, 2).map(r => ({ ...r, kc: measure(r, false) }));
  if ([trained, untrained, disconnected].some(sample => sample.length !== 24 ||
      [0, 1].some(m => sample.filter(r => r.mapping === m).length !== 12)))
    throw Error(`Unbalanced evaluation: ${seed}`);
  const labels = JSON.parse(readFileSync(new URL('../data/two_cue_population_fresh.json', import.meta.url)))
    .evaluation[seed].trials.map(r => r.shuffledLabel);
  const shuffled = trained.map((r, i) => ({ ...r, mapping: labels[i] }));
  return [seed, { trained: assess(trained), untrained: assess(untrained),
    shuffledLabels: assess(shuffled), disconnected: assess(disconnected),
    trials: trained.map((r, i) => ({ mapping: r.mapping, test: r.test,
      sourceRound: r.sourceRound, trace: r.trace, kc: r.kc,
      prediction: predict(r), shuffledLabel: labels[i],
      untrainedKc: untrained[i].kc, disconnectedKc: disconnected[i].kc })) }];
}));
if (net.m.some((weight, i) => weight !== weights[i])) throw Error('Weights changed');
const passed = [505, 506].every(seed => evaluation[seed].trained.score >= 18 &&
  evaluation[seed].trained.byMapping.every(n => n >= 8) && evaluation[seed].shuffledLabels.score <= 14);
const result = { command: 'node tools/check_two_cue_population_kc.mjs',
  hypothesis: 'The fixed graded input channel preserves mapping information in target-KC spike counts',
  calibrationTeachers: [501, 502], evaluationTeachers: [505, 506], studentSeed: 703,
  trainingEpisodes: 0, windowMs: [0, 40], targets, ceilings, scale, centroids,
  gate: 'each fresh teacher: trained >= 18/24, >= 8/12 per mapping, shuffled labels <= 14/24',
  evaluation, passed };
writeFileSync(output, JSON.stringify(result, null, 2));
console.log(JSON.stringify({ passed, evaluation: Object.fromEntries([505, 506].map(seed => {
  const { trained, untrained, shuffledLabels, disconnected } = evaluation[seed];
  return [seed, { trained, untrained, shuffledLabels, disconnected }];
})) }, null, 2));
