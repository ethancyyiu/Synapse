// One frozen, no-message decoder validation; protocol locked in docs/lab-notebook.md.
// node tools/check_rps_individual_mbon_decoder.mjs
import { open, readFile } from 'node:fs/promises';
import { Network, rng, wiring } from '../js/sim.js';
import { ChallengeCueHeadBandit, studentWeights } from '../js/rps_challenge_students.js';

const read = async name => JSON.parse(await readFile(new URL(`../data/${name}`, import.meta.url), 'utf8'));
const output = new URL('../data/rps_individual_mbon_decoder.json', import.meta.url);
const sub = await read('subgraph.json');
const { featureMean, featureStd } = (await read('rps_contextual_bandit.json')).design;
const oldCentroids = (await read('rps_readout_information.json')).centroids;
const calibrationSeed = 10931, evaluationSeeds = [10932, 10933], repeats = 12;
const mbonIds = Array.from({ length: 42 }, (_, i) => 2108 + i);
const wireCache = new Map();
const count = (xs, x) => xs.filter(y => y === x).length;
const sum = xs => xs.reduce((s, x) => s + x, 0);
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const protocol = { source: 'docs/lab-notebook.md prospective 42-MBON cue-decoder validation',
  calibrationSeed, evaluationSeeds, repeats, firstCues: [0, 1, 2],
  secondCue: '(first+2)%3', branches: ['no_message'],
  key: 's*10000000+(300+12*firstCue+repeat)*1000; second key +1',
  streams: ['rng(key+2)', 'rng(key+501)'], cueHz: 150, cueWindowMs: [0, 240],
  countResetMs: 50, decisionMs: 240, roundEndMs: 680, betweenRoundReset: true,
  mbonIds, countWindowMs: [50, 240], feature: 'log1p(individual count), no fitted scaling',
  centroid: 'arithmetic mean of 24 calibration features per cue',
  distance: 'squared Euclidean; tie chooses lowest cue',
  oldDecoder: 'unchanged rps_readout_information.json centroids',
  absoluteFloorPerCondition: 10, minimumOverallGainPerSeed: 4,
  minimumCue0GainPerSeed: 2, maximumConditionLoss: 0,
  rewards: false, learning: false, dnStimulation: false };

if (!mbonIds.every(i => sub.neurons.role[i] === 'mbon') ||
    sub.neurons.role.filter(x => x === 'mbon').length !== 42)
  throw Error('Locked 42-MBON identities changed');

function fresh(seed) {
  if (!wireCache.has(seed)) wireCache.set(seed, wiring(sub, 'real', seed));
  const net = new Network(sub, wireCache.get(seed), { seed });
  const agent = new ChallengeCueHeadBandit(net, seed, featureMean, featureStd);
  agent.explore = false; agent.readoutLearn = false;
  return { net, agent };
}
function round(net, agent, cue, key, firstCue, repeat, phase, seed) {
  const vpn = Array.from(net.laneVpn[cue].slice(2));
  net.rand = rng(key + 2); agent.rand = rng(key + 501);
  net.setDrive(vpn, 150);
  let row, dnDriven = false;
  for (let t = 0; t < 680; t++) {
    if (t === 50) net.resetCounts();
    if (t === 240) {
      net.setDrive(vpn, 0);
      agent.decide(net);
      const individual = mbonIds.map(i => net.count[i]);
      const pooled = net.groupMbon.map(ids => sum(Array.from(ids, i => net.count[i])) / ids.length);
      const laneCounts = [...agent.laneCounts];
      row = { seed, firstCue, repeat, phase, cue, key, individual,
        rates: [...agent.rates], pooled, laneCounts, estimatedCue: agent.activeHead };
    }
    if (t === 280) net.resetCounts();
    if (net.drive[3141] !== 0 || net.drive[3165] !== 0) dnDriven = true;
    net.step();
  }
  row.dnDriven = dnDriven;
  return row;
}
function collect(seed) {
  const rows = [], pairs = [];
  for (let firstCue = 0; firstCue < 3; firstCue++) for (let repeat = 0; repeat < repeats; repeat++) {
    const { net, agent } = fresh(seed), before = studentWeights(agent);
    const key = seed * 10000000 + (300 + 12 * firstCue + repeat) * 1000;
    net.resetState();
    rows.push(round(net, agent, firstCue, key, firstCue, repeat, 'first', seed));
    net.resetState();
    rows.push(round(net, agent, (firstCue + 2) % 3, key + 1,
      firstCue, repeat, 'second', seed));
    pairs.push({ firstCue, repeat,
      weightsUnchanged: studentWeights(agent) === before && net.m.every(x => x === 1) });
  }
  return { seed, rows, pairs };
}
const valid = batch => batch.rows.length === 72 && batch.pairs.length === 36 &&
  batch.pairs.every(p => p.weightsUnchanged) && batch.rows.every(r =>
    r.individual.length === 42 && r.individual.every(x => Number.isInteger(x) && x >= 0) &&
    r.rates.length === 3 && r.rates.every((x, j) => Number.isFinite(x) &&
      Math.abs(x - r.pooled[j]) <= 1e-12) && r.estimatedCue === r.cue && !r.dnDriven &&
    r.laneCounts[r.cue] > Math.max(...r.laneCounts.filter((_, i) => i !== r.cue))) &&
  [0, 1, 2].every(cue => ['first', 'second'].every(phase =>
    count(batch.rows.filter(r => r.phase === phase).map(r => r.cue), cue) === 12));
const feature = row => row.individual.map(Math.log1p);
function classify(values, centroids) {
  const distances = centroids.map(c => sum(c.map((v, i) => (values[i] - v) ** 2)));
  const ordered = [...distances].sort((a, b) => a - b);
  return { choice: distances.indexOf(ordered[0]), margin: ordered[1] - ordered[0], distances };
}
const confusion = rows => [0, 1, 2].map(cue => [0, 1, 2].map(choice =>
  rows.filter(r => r.cue === cue && r.choice === choice).length));
async function save(value) {
  const file = await open(output, 'wx');
  try { await file.writeFile(`${JSON.stringify(value, null, 2)}\n`); }
  finally { await file.close(); }
}

const calibration = collect(calibrationSeed);
if (!valid(calibration)) {
  await save({ protocol, status: 'stopped_calibration', calibration });
  console.log(JSON.stringify({ status: 'stopped_calibration' }));
  process.exit(0);
}
const centroids = [0, 1, 2].map(cue => {
  const xs = calibration.rows.filter(r => r.cue === cue).map(feature);
  if (xs.length !== 24) throw Error('Calibration cue count changed');
  return mbonIds.map((_, j) => sum(xs.map(x => x[j])) / xs.length);
});
const evaluation = evaluationSeeds.map(collect);
if (!evaluation.every(valid)) {
  await save({ protocol, status: 'stopped_evaluation_validity', calibration, centroids, evaluation });
  console.log(JSON.stringify({ status: 'stopped_evaluation_validity' }));
  process.exit(0);
}
for (const batch of evaluation) for (const row of batch.rows) {
  row.newDecoder = classify(feature(row), centroids);
  row.oldDecoder = classify(row.rates.map(Math.log1p), oldCentroids);
}
const conditions = evaluation.flatMap(batch => [0, 1, 2].flatMap(cue =>
  ['first', 'second'].map(phase => {
    const rows = batch.rows.filter(r => r.cue === cue && r.phase === phase);
    return { seed: batch.seed, cue, phase, n: rows.length,
      individualCorrect: count(rows.map(r => r.newDecoder.choice === cue), true),
      pooledCorrect: count(rows.map(r => r.oldDecoder.choice === cue), true) };
  })));
const seedResults = evaluation.map(batch => {
  const rows = batch.rows;
  return { seed: batch.seed, individualCorrect: count(rows.map(r => r.newDecoder.choice === r.cue), true),
    pooledCorrect: count(rows.map(r => r.oldDecoder.choice === r.cue), true),
    individualConfusion: confusion(rows.map(r => ({ cue: r.cue, choice: r.newDecoder.choice }))),
    pooledConfusion: confusion(rows.map(r => ({ cue: r.cue, choice: r.oldDecoder.choice }))) };
});
const gates = { validity: evaluation.every(valid) && valid(calibration),
  absolute: conditions.every(x => x.n === 12 && x.individualCorrect >= 10),
  noHarm: conditions.every(x => x.individualCorrect >= x.pooledCorrect),
  overallGain: seedResults.every(x => x.individualCorrect - x.pooledCorrect >= 4),
  cue0Gain: evaluationSeeds.every(seed => sum(conditions.filter(x => x.seed === seed && x.cue === 0)
    .map(x => x.individualCorrect - x.pooledCorrect)) >= 2) };
const passed = Object.values(gates).every(Boolean);
await save({ protocol, status: passed ? 'passed' : 'failed', passed, gates,
  calibration, centroids, evaluation, conditions, seedResults });
console.log(JSON.stringify({ status: passed ? 'passed' : 'failed', gates, seedResults, conditions }));
