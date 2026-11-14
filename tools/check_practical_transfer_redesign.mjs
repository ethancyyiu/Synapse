// Write-once, staged feasibility check. Stops on the first failed locked prerequisite.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { access, open, readFile } from 'node:fs/promises';
import { Network, rng, wiring } from '../js/sim.js';
import { RecoveryRidge } from './practical_transfer_redesign_controller.mjs';

const root = new URL('../', import.meta.url);
const protocolSha256 = createHash('sha256').update(await readFile(
  new URL('docs/practical-transfer-redesign-protocol.md', root))).digest('hex');
const mode = process.argv[2] ?? 'initial';
const preflight = mode === 'preflight';
const stage = preflight ? 'initial' : mode;
if (!['initial', 'confirmation'].includes(stage)) throw Error('Use initial or confirmation');
const config = stage === 'initial' ?
  { neural: 19102, train: 20261109111, evaluation: 20261109112,
    sensory: 20261109116, controller: 20261109118 } :
  { neural: 19103, train: 20261109113, evaluation: 20261109114,
    sensory: 20261109117, controller: 20261109119 };
const CAL = { neural: 19101, sensory: 20261109115 };
const path = part => new URL(`data/practical_transfer_redesign_${stage}_${part}.json`, root);
const exists = async url => { try { await access(url); return true; }
  catch (error) { if (error.code === 'ENOENT') return false; throw error; } };
const read = async url => JSON.parse(await readFile(url));
async function writeOnce(part, artifact) {
  const file = await open(path(part), 'wx');
  try { await file.writeFile(JSON.stringify({ protocolSha256, stage, part, ...artifact }, null, 2) + '\n'); }
  finally { await file.close(); }
}
if (stage === 'confirmation') {
  const previous = await read(path('teacher'));
  if (!previous.gate?.passed || previous.protocolSha256 !== protocolSha256)
    throw Error('Initial teacher prerequisite not passed under this exact protocol');
}
for (const part of ['raw', 'sensory', 'teacher']) if (await exists(path(part)))
  throw Error(`Write-once ${stage}/${part} result already exists`);

const sigmoid = z => 1 / (1 + Math.exp(-z));
const cost = [.06, .12, .31], delay = [2, 4, 7];
const probabilities = ([o, s, b]) => [
  sigmoid(1.6 + 1.2 * o - 1.5 * s - 2 * b),
  sigmoid(1.3 - .6 * o + 2 * s - 1.7 * b),
  sigmoid(.4 - .3 * o - .5 * s + 3.3 * b),
];
const expected = (spec, action) => 1.4 * spec.probabilities[action] - .4 - cost[action];
const reward = (spec, action) => (spec.outcomes[action] ? 1 : -.4) - cost[action];
const shuffled = (values, random) => {
  const a = [...values];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
};
function sample(random, octant, id) {
  const telemetry = [2, 1, 0].map(bit =>
    ((octant >> bit) & 1) / 2 + random() / 2);
  const p = probabilities(telemetry);
  const outcomes = p.map(v => Number(random() < v));
  return { id, octant, telemetry, probabilities: p, outcomes };
}
function tape(seed, kind) {
  const random = rng(seed), rows = [];
  if (kind === 'train') {
    for (let block = 0; block < 12; block++)
      for (const octant of shuffled([0, 1, 2, 3, 4, 5, 6, 7], random))
        for (let position = 0; position < 3; position++) {
          const row = sample(random, octant, `${seed}-${rows.length}`);
          row.forcedAction = (position + block) % 3;
          rows.push(row);
        }
    for (const octant of shuffled([0, 1, 2, 3, 4, 5, 6, 7]
      .flatMap(o => Array(24).fill(o)), random))
      rows.push(sample(random, octant, `${seed}-${rows.length}`));
  } else {
    const perOctant = kind === 'evaluation' ? 15 : 12;
    for (const octant of shuffled([0, 1, 2, 3, 4, 5, 6, 7]
      .flatMap(o => Array(perOctant).fill(o)), random))
      rows.push(sample(random, octant, `${seed}-${rows.length}`));
  }
  return rows;
}
const training = tape(config.train, 'train');
const evaluation = tape(config.evaluation, 'evaluation');
const calibrationCases = tape(CAL.sensory, 'sensory');
const sensoryCases = tape(config.sensory, 'sensory');
assert.equal(training.length, 480); assert.equal(evaluation.length, 120);
assert.equal(calibrationCases.length, 96); assert.equal(sensoryCases.length, 96);
for (let octant = 0; octant < 8; octant++) for (let a = 0; a < 3; a++)
  assert.equal(training.slice(0, 288).filter(r => r.octant === octant && r.forcedAction === a).length, 12);
const grid = [0, 1, 2, 3, 4].flatMap(i => [0, 1, 2, 3, 4].flatMap(j =>
  [0, 1, 2, 3, 4].map(k => ({ telemetry: [(i + .5) / 5, (j + .5) / 5, (k + .5) / 5],
    probabilities: probabilities([(i + .5) / 5, (j + .5) / 5, (k + .5) / 5]) }))));
const gridBest = grid.map(spec => [0, 1, 2].map(a => expected(spec, a)).indexOf(
  Math.max(...[0, 1, 2].map(a => expected(spec, a)))));
const gridCounts = [0, 1, 2].map(a => gridBest.filter(v => v === a).length);
const gridConstant = Math.max(...[0, 1, 2].map(a => grid.reduce((s, x) => s + expected(x, a), 0) / 125));
const gridOracle = grid.reduce((s, x) => s + Math.max(...[0, 1, 2].map(a => expected(x, a))), 0) / 125;
const taskGate = gridCounts.every(n => n >= 13) && gridOracle - gridConstant >= .08;
assert.deepEqual(gridCounts, [28, 53, 44]);
assert.ok(taskGate, 'Locked task geometry failed');
if (preflight) {
  console.log(JSON.stringify({ preflight: 'passed', training: training.length,
    forced: training.slice(0, 288).length, evaluation: evaluation.length,
    calibration: calibrationCases.length, sensory: sensoryCases.length,
    gridCounts, gridConstant, gridOracle, protocolSha256 }));
  process.exit(0);
}

const rawInput = spec => [1, ...spec.telemetry.map(v => (2 * v - 1) / Math.sqrt(3))];
const reference = rows => ({ constant: Math.max(...[0, 1, 2].map(a =>
  rows.reduce((s, r) => s + expected(r, a), 0) / rows.length)),
  oracle: rows.reduce((s, r) => s + Math.max(...[0, 1, 2].map(a => expected(r, a))), 0) / rows.length });
const ref = reference(evaluation);
function summary(rows) {
  return { n: rows.length, successes: rows.reduce((s, r) => s + r.success, 0),
    failures: rows.reduce((s, r) => s + 1 - r.success, 0),
    delaySeconds: rows.reduce((s, r) => s + r.delaySeconds, 0),
    escalations: rows.reduce((s, r) => s + r.escalation, 0),
    meanReward: rows.reduce((s, r) => s + r.reward, 0) / rows.length,
    meanExpected: rows.reduce((s, r) => s + r.expected, 0) / rows.length,
    actions: [0, 1, 2].map(a => rows.filter(r => r.action === a).length),
    byOctant: [0, 1, 2, 3, 4, 5, 6, 7].map(o => {
      const q = rows.filter(r => r.octant === o);
      return { octant: o, n: q.length, successes: q.reduce((s, r) => s + r.success, 0),
        actions: [0, 1, 2].map(a => q.filter(r => r.action === a).length),
        meanExpected: q.reduce((s, r) => s + r.expected, 0) / q.length };
    }) };
}
function frozen(core, cases, input, at) {
  const copy = new RecoveryRidge(core.d); copy.restore(core.snapshot());
  const before = JSON.stringify(copy.snapshot());
  const rows = cases.map(spec => {
    const x = input(spec), decision = copy.choose(x, { training: false });
    const a = decision.action;
    return { id: spec.id, octant: spec.octant, telemetry: spec.telemetry,
      probabilities: spec.probabilities, outcomes: spec.outcomes, input: x,
      action: a, scores: decision.means, uncertainty: decision.uncertainties,
      success: spec.outcomes[a], reward: reward(spec, a), expected: expected(spec, a),
      delaySeconds: delay[a], escalation: Number(a === 2) };
  });
  assert.equal(JSON.stringify(copy.snapshot()), before);
  return { at, rows, summary: summary(rows), parametersUnchanged: true,
    controller: before };
}
function learn(cases, evalCases, input, d, randomSeed) {
  const core = new RecoveryRidge(d), random = rng(randomSeed);
  const initial = core.snapshot(), checkpoints = [frozen(core, evalCases, input, 0)], rows = [];
  for (let i = 0; i < cases.length; i++) {
    const spec = cases[i], x = input(spec), decision = core.choose(x, {
      training: true, forcedAction: spec.forcedAction ?? null, random });
    const a = decision.action, y = spec.outcomes[a], r = reward(spec, a);
    core.update(r);
    rows.push({ index: i, id: spec.id, octant: spec.octant,
      telemetry: spec.telemetry, probabilities: spec.probabilities,
      outcomes: spec.outcomes, input: x, forced: i < 288,
      action: a, scores: decision.means, uncertainty: decision.uncertainties,
      draws: decision.draws, success: y, reward: r, expected: expected(spec, a),
      delaySeconds: delay[a], escalation: Number(a === 2) });
    if ([288, 480].includes(i + 1)) checkpoints.push(frozen(core, evalCases, input, i + 1));
  }
  return { rows, training: summary(rows), checkpoints, initial, final: core.snapshot() };
}
function learnGate(run) {
  const initial = run.checkpoints[0].summary, last = run.checkpoints.at(-1).summary;
  const gap = ref.oracle - ref.constant;
  return { improvement: last.meanExpected - initial.meanExpected,
    overConstant: last.meanExpected - ref.constant,
    regretClosure: gap > 0 ? (last.meanExpected - ref.constant) / gap : null,
    actionCoverage: last.actions,
    passed: gap > 0 && last.meanExpected - initial.meanExpected >= .05 &&
      last.meanExpected - ref.constant >= .05 &&
      (last.meanExpected - ref.constant) / gap >= .5 &&
      last.actions.every(n => n >= 12) };
}
const raw = learn(training, evaluation, rawInput, 4, config.controller);
const rawGate = learnGate(raw);
await writeOnce('raw', { status: rawGate.passed ? 'passed' : 'failed', config, task: {
  costs: cost, delay, gridCounts, gridConstant, gridOracle, taskGate, reference: ref },
  tapes: { training, evaluation }, controller: raw, gate: rawGate });
console.log(JSON.stringify({ stage, part: 'raw', gate: rawGate,
  checkpoints: raw.checkpoints.map(x => ({ at: x.at, ...x.summary })) }));
if (!rawGate.passed) process.exit(0);

// Independent sensory stage begins only after the write-once raw pass.
const sub = JSON.parse(await readFile(new URL('data/subgraph.json', root)));
const makeNet = seed => new Network(sub, wiring(sub, 'real', seed), { seed });
const key = (seed, phase, index) => seed * 1e7 + phase * 1e6 + index * 1000 + 1;
const allKeys = [
  ...calibrationCases.map((_, i) => key(CAL.neural, 0, i) >>> 0),
  ...sensoryCases.map((_, i) => key(config.neural, 1, i) >>> 0),
  ...training.map((_, i) => key(config.neural, 2, i) >>> 0),
  ...evaluation.map((_, i) => key(config.neural, 3, i) >>> 0),
];
assert.equal(new Set(allKeys).size, allKeys.length, 'Effective neural stream key overlap');
function probe(net, spec, seed, phase, index) {
  net.resetState(); const streamKey = key(seed, phase, index); net.rand = rng(streamKey);
  for (let j = 0; j < 3; j++) net.setDrive(net.laneVpn[j].slice(2), 40 + 110 * spec.telemetry[j]);
  for (let t = 0; t < 240; t++) { if (t === 50) net.resetCounts(); net.step(); }
  for (let j = 0; j < 3; j++) net.setDrive(net.laneVpn[j].slice(2), 0);
  const vpn = net.laneVpn.map(ids => [...ids.slice(2)]
    .reduce((sum, id) => sum + net.count[id], 0));
  const mbon = [...net.idx.mbon].map(id => ({ id, group: net.groupOf[id], count: net.count[id] }));
  const kcActive = [...net.idx.kc].filter(id => net.count[id] > 0)
    .map(id => [id, net.count[id]]);
  const pooled = net.groupMbon.map(ids => ids.reduce((sum, id) => sum + net.count[id], 0) / ids.length);
  return { ...spec, index, streamKey, vpn, kcActive, mbon, pooled };
}
const calNet = makeNet(CAL.neural);
const calRows = calibrationCases.map((spec, i) => probe(calNet, spec, CAL.neural, 0, i));
const firstNet = makeNet(config.neural);
const sensoryRows = sensoryCases.map((spec, i) => probe(firstNet, spec, config.neural, 1, i));
const repeat = probe(firstNet, sensoryCases[0], config.neural, 1, 0);
assert.deepEqual(repeat, sensoryRows[0], 'Keyed neural replay mismatch');
const logCount = row => row.mbon.map(m => Math.log1p(m.count));
const mean = x => x.reduce((s, v) => s + v, 0) / x.length;
const scaler = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
  20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
  .map(j => { const values = calRows.map(r => logCount(r)[j]), m = mean(values);
    return { mean: m, sd: Math.max(Math.sqrt(values.reduce((s, v) => s + (v - m) ** 2, 0) /
      (values.length - 1)), .1) }; });
const neuralInput = row => [1, ...logCount(row).map((v, j) =>
  (v - scaler[j].mean) / scaler[j].sd / Math.sqrt(42))];
// Analysis-only ridge reconstruction: intercept unpenalized, other features penalty 4.
function solve(A, b) {
  const d = b.length, M = A.map((row, i) => [...row, b[i]]);
  for (let k = 0; k < d; k++) {
    let pivot = k;
    for (let i = k + 1; i < d; i++) if (Math.abs(M[i][k]) > Math.abs(M[pivot][k])) pivot = i;
    if (Math.abs(M[pivot][k]) < 1e-12) throw Error('Singular sensory ridge');
    [M[k], M[pivot]] = [M[pivot], M[k]];
    const scale = M[k][k]; for (let j = k; j <= d; j++) M[k][j] /= scale;
    for (let i = 0; i < d; i++) if (i !== k) {
      const factor = M[i][k]; for (let j = k; j <= d; j++) M[i][j] -= factor * M[k][j];
    }
  }
  return M.map(row => row[d]);
}
const X = calRows.map(neuralInput), d = 43;
const gram = Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) =>
  X.reduce((s, x) => s + x[i] * x[j], 0) + (i === j && i > 0 ? 4 : 0)));
const recon = [0, 1, 2].map(j => solve(gram, Array.from({ length: d }, (_, k) =>
  X.reduce((s, x, i) => s + x[k] * calRows[i].telemetry[j], 0))));
const dot = (x, y) => x.reduce((s, v, i) => s + v * y[i], 0);
for (const row of sensoryRows) row.reconstructed = recon.map(beta => dot(neuralInput(row), beta));
const sensoryMetrics = [0, 1, 2].map(j => {
  const actual = sensoryRows.map(r => r.telemetry[j]), predicted = sensoryRows.map(r => r.reconstructed[j]);
  const avg = mean(actual), sse = actual.reduce((s, v, i) => s + (v - predicted[i]) ** 2, 0);
  const sst = actual.reduce((s, v) => s + (v - avg) ** 2, 0);
  const vpn = sensoryRows.map(r => r.vpn[j]), vAvg = mean(vpn);
  const cov = actual.reduce((s, v, i) => s + (v - avg) * (vpn[i] - vAvg), 0);
  const vsst = vpn.reduce((s, v) => s + (v - vAvg) ** 2, 0);
  return { dimension: j, r2: 1 - sse / sst,
    mae: mean(actual.map((v, i) => Math.abs(v - predicted[i]))),
    vpnCorrelation: cov / Math.sqrt(sst * vsst), vpnVariance: vsst / actual.length };
});
const sensoryGate = { replayPassed: true, weightsUnchanged: calNet.m.every(v => v === 1) &&
  firstNet.m.every(v => v === 1), metrics: sensoryMetrics,
  passed: sensoryMetrics.every(v => v.r2 >= .4 && v.mae <= .2 &&
    v.vpnCorrelation >= .8 && Number.isFinite(v.vpnVariance) && v.vpnVariance > 0) &&
    calNet.m.every(v => v === 1) && firstNet.m.every(v => v === 1) };
await writeOnce('sensory', { status: sensoryGate.passed ? 'passed' : 'failed',
  calibrationSeed: CAL, config, scaler, analyticalReconstruction: recon,
  calibrationRows: calRows, validationRows: sensoryRows, gate: sensoryGate });
console.log(JSON.stringify({ stage, part: 'sensory', gate: sensoryGate }));
if (!sensoryGate.passed) process.exit(0);

// Independent Fly A learning only after raw and sensory gates.
const trainNet = makeNet(config.neural), evalNet = makeNet(config.neural);
const neuralTraining = training.map((spec, i) => probe(trainNet, spec, config.neural, 2, i));
const neuralEvaluation = evaluation.map((spec, i) => probe(evalNet, spec, config.neural, 3, i));
const teacher = learn(neuralTraining, neuralEvaluation, neuralInput, 43, config.controller);
const baseGate = learnGate(teacher), rawFinal = raw.checkpoints.at(-1).summary.meanExpected;
const teacherGate = { ...baseGate, withinRaw: teacher.checkpoints.at(-1).summary.meanExpected - rawFinal,
  parametersChanged: JSON.stringify(teacher.initial) !== JSON.stringify(teacher.final),
  neuralWeightsUnchanged: trainNet.m.every(v => v === 1) && evalNet.m.every(v => v === 1) };
teacherGate.passed = baseGate.passed && teacherGate.withinRaw >= -.05 &&
  teacherGate.parametersChanged && teacherGate.neuralWeightsUnchanged;
await writeOnce('teacher', { status: teacherGate.passed ? 'passed' : 'failed', config,
  rawResult: path('raw').pathname, sensoryResult: path('sensory').pathname,
  reference: ref, neuralTraining, neuralEvaluation, controller: teacher, gate: teacherGate });
console.log(JSON.stringify({ stage, part: 'teacher', gate: teacherGate,
  checkpoints: teacher.checkpoints.map(x => ({ at: x.at, ...x.summary })) }));
