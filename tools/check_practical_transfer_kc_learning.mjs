// Prospective isolated KC-input reward-learning study. Write-once results.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { access, open, readFile } from 'node:fs/promises';
import { Network, rng, wiring } from '../js/sim.js';
import { RecoveryRidge } from './practical_transfer_redesign_controller.mjs';

const root = new URL('../', import.meta.url);
const sha = createHash('sha256').update(await readFile(
  new URL('docs/practical-transfer-kc-learning-protocol.md', root))).digest('hex');
const mode = process.argv[2] ?? 'initial';
if (!['preflight', 'initial', 'confirmation'].includes(mode)) throw Error('Use preflight, initial, or confirmation');
const initial = mode !== 'confirmation';
const config = initial ? { neural: 19402, train: 20261109111, evaluation: 20261109112,
  controller: 20261109118 } : { neural: 19403, train: 20261109413,
  evaluation: 20261109414, controller: 20261109418 };
const out = new URL(`data/practical_transfer_kc_learning_${initial ? 'initial' : 'confirmation'}.json`, root);
const info = JSON.parse(await readFile(new URL('data/practical_transfer_kc_information.json', root)));
const oldRaw = JSON.parse(await readFile(new URL('data/practical_transfer_redesign_initial_raw.json', root)));
const sub = JSON.parse(await readFile(new URL('data/subgraph.json', root)));
const kcIds = sub.neurons.role.flatMap((role, id) => role === 'kc' ? [id] : []);
const kcPosition = new Map(kcIds.map((id, i) => [id, i]));
assert.equal(kcIds.length, 2045); assert.equal(info.calibration.rows.length, 96);
assert.equal(info.gates.allPassed, true, 'prior KC information gate');
if (mode === 'confirmation') {
  const prior = JSON.parse(await readFile(new URL(
    'data/practical_transfer_kc_learning_initial.json', root)));
  assert.equal(prior.protocolSha256, sha);
  assert.equal(prior.gates.passed, true, 'initial must pass before confirmation');
}
const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;
function bucketCounts(row) {
  const counts = new Float64Array(42);
  for (const [id, count] of row.kcActive) {
    const index = kcPosition.get(id);
    assert.ok(index !== undefined && Number.isInteger(count) && count > 0);
    counts[index % 42] += count;
  }
  return Array.from(counts, v => Math.log1p(v));
}
const calibration = info.calibration.rows.map(bucketCounts);
const scaler = Array.from({ length: 42 }, (_, j) => {
  const values = calibration.map(x => x[j]), m = mean(values);
  return { mean: m, sd: Math.max(Math.sqrt(values.reduce((s, v) =>
    s + (v - m) ** 2, 0) / (values.length - 1)), .1) };
});
const kcInput = row => [1, ...bucketCounts(row).map((v, j) =>
  (v - scaler[j].mean) / scaler[j].sd / Math.sqrt(42))];
const rawInput = spec => [1, ...spec.telemetry.map(v => (2 * v - 1) / Math.sqrt(3))];
const sigmoid = z => 1 / (1 + Math.exp(-z));
const cost = [.06, .12, .31], delay = [2, 4, 7];
const probabilities = ([o, s, b]) => [
  sigmoid(1.6 + 1.2 * o - 1.5 * s - 2 * b),
  sigmoid(1.3 - .6 * o + 2 * s - 1.7 * b),
  sigmoid(.4 - .3 * o - .5 * s + 3.3 * b),
];
const expected = (spec, action) => 1.4 * spec.probabilities[action] - .4 - cost[action];
const reward = (spec, action) => (spec.outcomes[action] ? 1 : -.4) - cost[action];
function shuffled(values, random) {
  const a = [...values];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}
function sample(random, octant, id) {
  const telemetry = [2, 1, 0].map(bit => ((octant >> bit) & 1) / 2 + random() / 2);
  const p = probabilities(telemetry);
  return { id, octant, telemetry, probabilities: p,
    outcomes: p.map(v => Number(random() < v)) };
}
function tape(seed, kind) {
  const random = rng(seed), rows = [];
  if (kind === 'train') {
    for (let block = 0; block < 12; block++)
      for (const octant of shuffled([0, 1, 2, 3, 4, 5, 6, 7], random))
        for (let position = 0; position < 3; position++) {
          const row = sample(random, octant, `${seed}-${rows.length}`);
          row.forcedAction = (position + block) % 3; rows.push(row);
        }
    for (const octant of shuffled([0, 1, 2, 3, 4, 5, 6, 7]
      .flatMap(o => Array(24).fill(o)), random))
      rows.push(sample(random, octant, `${seed}-${rows.length}`));
  } else for (const octant of shuffled([0, 1, 2, 3, 4, 5, 6, 7]
    .flatMap(o => Array(15).fill(o)), random))
    rows.push(sample(random, octant, `${seed}-${rows.length}`));
  return rows;
}
const training = initial ? oldRaw.tapes.training : tape(config.train, 'train');
const evaluation = initial ? oldRaw.tapes.evaluation : tape(config.evaluation, 'evaluation');
assert.equal(training.length, 480); assert.equal(evaluation.length, 120);
if (initial) {
  assert.equal(config.train, oldRaw.config.train);
  assert.equal(config.evaluation, oldRaw.config.evaluation);
  assert.equal(config.controller, oldRaw.config.controller);
  assert.equal(JSON.stringify(training), JSON.stringify(tape(config.train, 'train')));
  assert.equal(JSON.stringify(evaluation), JSON.stringify(tape(config.evaluation, 'evaluation')));
  assert.equal(oldRaw.gate.passed, true);
}
for (let octant = 0; octant < 8; octant++) for (let action = 0; action < 3; action++)
  assert.equal(training.slice(0, 288).filter(r =>
    r.octant === octant && r.forcedAction === action).length, 12);
assert.equal(kcInput(info.calibration.rows[0]).length, 43);
assert.ok(kcInput(info.calibration.rows[0]).every(Number.isFinite));
const rawBaseline = oldRaw.controller.checkpoints.map(c => c.summary.meanExpected);
if (mode === 'preflight') {
  console.log(JSON.stringify({ protocolSha256: sha, config, calibration: calibration.length,
    dimensions: [42, 43], training: training.length, heldOut: evaluation.length,
    forcedPerOctantAction: 12, archivedRawExpected: rawBaseline }));
  process.exit(0);
}
try { await access(out); throw Error('Write-once result already exists'); }
catch (error) { if (error.code !== 'ENOENT') throw error; }

const key = (phase, index) => config.neural * 1e7 + phase * 1e6 + index * 1000 + 1;
const keys = [...training.map((_, i) => key(2, i) >>> 0),
  ...evaluation.map((_, i) => key(3, i) >>> 0)];
assert.equal(new Set(keys).size, keys.length, 'effective RNG key collision');
function probe(net, spec, phase, index) {
  net.resetState(); const streamKey = key(phase, index); net.rand = rng(streamKey);
  const drives = spec.telemetry.map(v => 40 + 110 * v);
  for (let j = 0; j < 3; j++) net.setDrive(net.laneVpn[j].slice(2), drives[j]);
  for (let t = 0; t < 240; t++) { if (t === 50) net.resetCounts(); net.step(); }
  for (let j = 0; j < 3; j++) net.setDrive(net.laneVpn[j].slice(2), 0);
  const kcActive = [...net.idx.kc].filter(id => net.count[id] > 0)
    .map(id => [id, net.count[id]]);
  const row = { index, neuralSeed: config.neural, phase, streamKey,
    telemetry: spec.telemetry, octant: spec.octant, drives, windowMs: [50, 240],
    reset: 'resetState before drive; resetCounts at 50 ms',
    vpn: net.laneVpn.map(ids => ids.slice(2).reduce((s, id) => s + net.count[id], 0)),
    kcActive };
  row.features = kcInput(row);
  assert.equal(row.features.length, 43);
  assert.ok(row.features.every(Number.isFinite));
  return row;
}
const trainNet = new Network(sub, wiring(sub, 'real', config.neural), { seed: config.neural });
const evalNet = new Network(sub, wiring(sub, 'real', config.neural), { seed: config.neural });
const neuralTraining = training.map((spec, i) => probe(trainNet, spec, 2, i));
const neuralEvaluation = evaluation.map((spec, i) => probe(evalNet, spec, 3, i));
assert.deepEqual(probe(trainNet, training[0], 2, 0), neuralTraining[0]);
assert.deepEqual(probe(evalNet, evaluation[0], 3, 0), neuralEvaluation[0]);
assert.ok(trainNet.m.every(v => v === 1) && evalNet.m.every(v => v === 1));
console.log(JSON.stringify({ neuralSeed: config.neural,
  trainingProbes: neuralTraining.length, evaluationProbes: neuralEvaluation.length }));

const reference = { constant: Math.max(...[0, 1, 2].map(a =>
  mean(evaluation.map(spec => expected(spec, a))))),
  oracle: mean(evaluation.map(spec => Math.max(...[0, 1, 2].map(a => expected(spec, a))))) };
function summarize(rows) {
  return { n: rows.length, successes: rows.reduce((s, r) => s + r.success, 0),
    failures: rows.reduce((s, r) => s + 1 - r.success, 0),
    delaySeconds: rows.reduce((s, r) => s + r.delaySeconds, 0),
    escalations: rows.reduce((s, r) => s + r.escalation, 0),
    meanReward: mean(rows.map(r => r.reward)),
    meanExpected: mean(rows.map(r => r.expected)),
    actions: [0, 1, 2].map(a => rows.filter(r => r.action === a).length),
    byOctant: [0, 1, 2, 3, 4, 5, 6, 7].map(o => {
      const z = rows.filter(r => r.octant === o);
      return { octant: o, n: z.length,
        successes: z.reduce((s, r) => s + r.success, 0),
        actions: [0, 1, 2].map(a => z.filter(r => r.action === a).length),
        meanExpected: mean(z.map(r => r.expected)) };
    }) };
}
function frozen(core, cases, features, at) {
  const copy = new RecoveryRidge(core.d); copy.restore(core.snapshot());
  const before = JSON.stringify(copy.snapshot());
  const rows = cases.map((spec, i) => {
    const input = features[i], decision = copy.choose(input, { training: false });
    const action = decision.action;
    return { index: i, octant: spec.octant, telemetry: spec.telemetry,
      input, action, scores: decision.means, uncertainty: decision.uncertainties,
      success: spec.outcomes[action], reward: reward(spec, action),
      expected: expected(spec, action), delaySeconds: delay[action],
      escalation: Number(action === 2) };
  });
  assert.equal(JSON.stringify(copy.snapshot()), before, 'frozen controller changed');
  return { at, rows, summary: summarize(rows), controller: before, unchanged: true };
}
function learn(cases, evalCases, features, evalFeatures, d, controllerSeed) {
  const core = new RecoveryRidge(d), random = rng(controllerSeed);
  const initialState = core.snapshot(), rows = [];
  const checkpoints = [frozen(core, evalCases, evalFeatures, 0)];
  for (let i = 0; i < cases.length; i++) {
    const spec = cases[i], x = features[i];
    const before = [...core.n];
    const choice = core.choose(x, { training: true,
      forcedAction: spec.forcedAction ?? null, random });
    const action = choice.action, r = reward(spec, action);
    core.update(r);
    assert.equal(core.n[action], before[action] + 1);
    for (const other of [0, 1, 2]) if (other !== action) assert.equal(core.n[other], before[other]);
    rows.push({ index: i, octant: spec.octant, telemetry: spec.telemetry,
      input: x, forced: i < 288, action, scores: choice.means,
      uncertainty: choice.uncertainties, draws: choice.draws,
      success: spec.outcomes[action], reward: r, expected: expected(spec, action),
      delaySeconds: delay[action], escalation: Number(action === 2) });
    if ([288, 480].includes(i + 1)) checkpoints.push(frozen(core, evalCases, evalFeatures, i + 1));
  }
  return { initial: initialState, final: core.snapshot(), rows,
    training: summarize(rows), checkpoints };
}
const kc = learn(training, evaluation, neuralTraining.map(r => r.features),
  neuralEvaluation.map(r => r.features), 43, config.controller);
const raw = initial ? oldRaw.controller : learn(training, evaluation,
  training.map(rawInput), evaluation.map(rawInput), 4, config.controller);
if (initial) {
  assert.deepEqual(raw.checkpoints.map(c => c.summary.meanExpected), rawBaseline);
  assert.equal(raw.checkpoints[2].summary.meanExpected, oldRaw.controller.checkpoints[2].summary.meanExpected);
}
const rawFinal = raw.checkpoints[2].summary.meanExpected;
const first = kc.checkpoints[0].summary, last = kc.checkpoints[2].summary;
const gap = reference.oracle - reference.constant;
const rawFirst = raw.checkpoints[0].summary, rawLast = raw.checkpoints[2].summary;
const rawReferencePassed = gap > 0 && rawLast.meanExpected - rawFirst.meanExpected >= .05 &&
  rawLast.meanExpected - reference.constant >= .05 &&
  (rawLast.meanExpected - reference.constant) / gap >= .5 &&
  rawLast.actions.every(n => n >= 12);
const gates = { positiveGap: gap > 0,
  rawReferencePassed,
  ownGain: last.meanExpected - first.meanExpected,
  overConstant: last.meanExpected - reference.constant,
  closure: gap > 0 ? (last.meanExpected - reference.constant) / gap : null,
  actionCoverage: last.actions,
  versusRaw: last.meanExpected - rawFinal,
  controllerChanged: JSON.stringify(kc.initial) !== JSON.stringify(kc.final),
  weightsUnchanged: trainNet.m.every(v => v === 1) && evalNet.m.every(v => v === 1),
  frozenUnchanged: kc.checkpoints.every(c => c.unchanged) };
gates.passed = gates.positiveGap && gates.rawReferencePassed && gates.ownGain >= .05 &&
  gates.overConstant >= .05 && gates.closure >= .5 &&
  gates.actionCoverage.every(n => n >= 12) && gates.versusRaw >= -.05 &&
  gates.controllerChanged && gates.weightsUnchanged && gates.frozenUnchanged;
const active = neuralTraining.map(r => r.kcActive.length);
const featureSummary = { meanActiveKC: mean(active), minActiveKC: Math.min(...active),
  maxActiveKC: Math.max(...active), meanFeature: Array.from({ length: 42 }, (_, j) =>
    mean(neuralTraining.map(r => r.features[j + 1]))),
  minFeature: Math.min(...neuralTraining.flatMap(r => r.features.slice(1))),
  maxFeature: Math.max(...neuralTraining.flatMap(r => r.features.slice(1))) };
const artifact = { protocolSha256: sha, stage: initial ? 'initial' : 'confirmation',
  config, calibration: { neuralSeed: 19101, caseSeed: 20261109115, scaler },
  task: { costs: cost, delay, reference },
  tapes: { training, evaluation }, neuralTraining, neuralEvaluation,
  featureSummary, controller: { kc, raw }, gates,
  integrity: { keyedReplay: true, frozenNoUpdate: true, selectedArmOnly: true,
    weightsUnchanged: gates.weightsUnchanged } };
const file = await open(out, 'wx');
try { await file.writeFile(JSON.stringify(artifact) + '\n'); }
finally { await file.close(); }
console.log(JSON.stringify({ stage: artifact.stage, gates,
  kcCheckpoints: kc.checkpoints.map(c => ({ at: c.at, ...c.summary })),
  rawCheckpoints: raw.checkpoints.map(c => ({ at: c.at, ...c.summary })),
  training: kc.training, featureSummary: { ...featureSummary, meanFeature: undefined } }));
