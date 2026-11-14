// One write-once synthetic recovery learnability diagnostic; no teacher/student study.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { access, open, readFile } from 'node:fs/promises';
import { Network, rng, wiring } from '../js/sim.js';
import { LinUcbCore } from './package_recovery_linucb.mjs';

const root = new URL('../', import.meta.url);
const read = async path => JSON.parse(await readFile(new URL(path, root)));
const output = new URL('data/package_recovery_learnability.json', root);
try { await access(output); throw Error('Write-once learnability result already exists'); }
catch (error) { if (error.code !== 'ENOENT') throw error; }
const protocolSha256 = createHash('sha256').update(
  await readFile(new URL('docs/package-recovery-learnability-protocol.md', root))).digest('hex');
const prior = await read('data/package_recovery_initial.json');
const priorUcb = await read('data/package_recovery_uncertainty_initial.json');
assert.equal(prior.status, 'stopped_teacher');
assert.equal(priorUcb.status, 'failed');
const { cueProb, successProb, delays } = prior.task;
const sub = await read('data/subgraph.json');
const { featureMean, featureStd } = (await read('data/rps_contextual_bandit.json')).design;
const seeds = [
  { neural: 16001, train: 2026110901, eval: 2026110902 },
  { neural: 16002, train: 2026110903, eval: 2026110904 },
];
const checkpoints = [0, 24, 48, 96, 144];
const expected = (mode, action) =>
  1.4 * successProb[mode][action] - .4 - .03 * delays[action] - (action === 2 ? .20 : 0);
const reward = (spec, action) =>
  (spec.outcomes[action] ? 1 : -.4) - .03 * delays[action] - (action === 2 ? .20 : 0);
const cueExpected = [0, 1, 2].map(cue => [0, 1, 2].map(action =>
  cueProb.reduce((sum, row, mode) => sum + row[cue] * expected(mode, action), 0) /
  cueProb.reduce((sum, row) => sum + row[cue], 0)));
const best = cueExpected.map(row => row.indexOf(Math.max(...row)));
assert.deepEqual(best, [0, 1, 2]);
const population = {
  raw: cueProb.reduce((sum, row, mode) =>
    sum + row.reduce((s, p, cue) => s + p * expected(mode, best[cue]), 0) / 4, 0),
  oracle: [0, 1, 2, 3].reduce((sum, mode) =>
    sum + Math.max(...[0, 1, 2].map(action => expected(mode, action))) / 4, 0),
};
function cases(seed, perMode) {
  const random = rng(seed), rows = [];
  for (let mode = 0; mode < 4; mode++) for (let j = 0; j < perMode; j++) {
    const u = random(); let cue = 0, cumulative = 0;
    while (cue < 2 && u >= (cumulative += cueProb[mode][cue])) cue++;
    const outcomes = successProb[mode].map(p => Number(random() < p));
    rows.push({ id: `${seed}-${mode}-${j}`, mode, cue, outcomes });
  }
  for (let i = rows.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [rows[i], rows[j]] = [rows[j], rows[i]];
  }
  return rows;
}
function probe(net, spec, key) {
  net.resetState(); net.rand = rng(key);
  const vpn = net.laneVpn[spec.cue].slice(2);
  net.setDrive(vpn, 150);
  for (let t = 0; t < 240; t++) {
    if (t === 50) net.resetCounts();
    net.step();
  }
  net.setDrive(vpn, 0);
  const rates = net.groupMbon.map(ids => ids.reduce((sum, id) => sum + net.count[id], 0) / ids.length);
  const f = rates.map((rate, j) => (Math.log1p(rate) - featureMean[j]) / featureStd[j]);
  const lanes = net.laneVpn.map(ids => [...ids.slice(2)].reduce((sum, id) => sum + net.count[id], 0));
  return { ...spec, key, rates, f, lanes };
}
const artifact = { protocolSha256, status: 'started', seeds, task: { cueProb, successProb, delays,
  cueExpected, best, population, checkpoints }, stages: [], gates: null };
async function finish(status) {
  artifact.status = status;
  const file = await open(output, 'wx');
  try { await file.writeFile(JSON.stringify(artifact, null, 2) + '\n'); }
  finally { await file.close(); }
  console.log(JSON.stringify({ status, stages: artifact.stages.map(s => ({
    seed: s.seed.neural, information: s.information?.summary,
    raw: s.arms?.raw?.frozen?.at(-1)?.summary,
    neural: s.arms?.neural?.frozen?.at(-1)?.summary,
  })), gates: artifact.gates }));
}
const onehot = cue => [0, 1, 2].map(i => Number(i === cue));
function summarize(rows) {
  return { n: rows.length, successes: rows.reduce((sum, row) => sum + row.success, 0),
    meanReward: rows.reduce((sum, row) => sum + row.reward, 0) / rows.length,
    meanExpected: rows.reduce((sum, row) => sum + row.expected, 0) / rows.length,
    actions: [0, 1, 2].map(a => rows.filter(row => row.action === a).length),
    byCue: [0, 1, 2].map(cue => {
      const q = rows.filter(row => row.cue === cue);
      return { cue, n: q.length, successes: q.reduce((sum, row) => sum + row.success, 0),
        actions: [0, 1, 2].map(a => q.filter(row => row.action === a).length),
        meanExpected: q.reduce((sum, row) => sum + row.expected, 0) / q.length };
    }) };
}
function information(train, evaluation) {
  const centroids = [0, 1, 2].map(cue => [0, 1, 2].map(j => {
    const q = train.filter(row => row.cue === cue);
    return q.reduce((sum, row) => sum + row.f[j], 0) / q.length;
  }));
  const rows = evaluation.map(row => {
    const distances = centroids.map(center =>
      center.reduce((sum, x, j) => sum + (x - row.f[j]) ** 2, 0));
    const decodedCue = distances.indexOf(Math.min(...distances));
    const action = best[decodedCue];
    return { id: row.id, cue: row.cue, mode: row.mode, decodedCue,
      action, expected: expected(row.mode, action) };
  });
  const summary = { recognized: rows.filter(row => row.cue === row.decodedCue).length,
    byCue: [0, 1, 2].map(cue => {
      const q = rows.filter(row => row.cue === cue);
      return { cue, recognized: q.filter(row => row.decodedCue === cue).length, n: q.length };
    }),
    decodedExpected: rows.reduce((sum, row) => sum + row.expected, 0) / rows.length,
    rawExpected: evaluation.reduce((sum, row) => sum + expected(row.mode, best[row.cue]), 0) / evaluation.length,
    oracleExpected: evaluation.reduce((sum, row) =>
      sum + Math.max(...[0, 1, 2].map(a => expected(row.mode, a))), 0) / evaluation.length };
  summary.passed = summary.recognized >= 30 && summary.decodedExpected >= .16;
  return { centroids, rows, summary };
}
function evaluate(core, features, attempt) {
  const frozen = new LinUcbCore(); frozen.restore(core.snapshot());
  const before = JSON.stringify(frozen.snapshot());
  const rows = features.map((spec, index) => {
    const { action, means, bonuses } = frozen.choose(spec.input, false);
    assert.ok(bonuses.every(x => x === 0));
    return { index, id: spec.id, cue: spec.cue, mode: spec.mode,
      action, scores: means, success: spec.outcomes[action],
      reward: reward(spec, action), expected: expected(spec.mode, action) };
  });
  assert.equal(JSON.stringify(frozen.snapshot()), before);
  return { attempt, rows, summary: summarize(rows), weightsUnchanged: true };
}
function trainArm(name, training, evaluation) {
  const feature = spec => name === 'raw' ? onehot(spec.cue) : spec.f;
  const trainRows = training.map(spec => ({ ...spec, input: feature(spec) }));
  const evalRows = evaluation.map(spec => ({ ...spec, input: feature(spec) }));
  const core = new LinUcbCore(), initialWeights = core.snapshot(), frozen = [evaluate(core, evalRows, 0)];
  const rows = [];
  for (let i = 0; i < trainRows.length; i++) {
    const spec = trainRows[i], decision = core.choose(spec.input, true);
    const { action } = decision, chosenReward = reward(spec, action);
    rows.push({ index: i, id: spec.id, cue: spec.cue, mode: spec.mode,
      action, success: spec.outcomes[action], reward: chosenReward,
      expected: expected(spec.mode, action), scores: decision.means,
      selectionScores: decision.selectionScores, uncertainty: decision.uncertainty });
    core.update(chosenReward);
    if (checkpoints.includes(i + 1)) frozen.push(evaluate(core, evalRows, i + 1));
  }
  return { initialWeights, finalWeights: core.snapshot(), training: { rows, summary: summarize(rows) }, frozen };
}
for (const seed of seeds) {
  const trainingCases = cases(seed.train, 36), evaluationCases = cases(seed.eval, 9);
  assert.equal(trainingCases.length, 144); assert.equal(evaluationCases.length, 36);
  assert.equal(new Set([...trainingCases, ...evaluationCases].map(row => row.id)).size, 180);
  const net = new Network(sub, wiring(sub, 'real', seed.neural), { seed: seed.neural });
  const training = trainingCases.map((spec, i) => probe(net, spec, seed.neural * 10000000 + i * 1000 + 1));
  const evaluation = evaluationCases.map((spec, i) =>
    probe(net, spec, seed.neural * 10000000 + 900000 + i * 1000 + 1));
  const neuralWeightsUnchanged = net.m.every(x => x === 1);
  assert.ok(neuralWeightsUnchanged && [...training, ...evaluation].every(row => row.f.every(Number.isFinite)));
  const info = information(training, evaluation);
  artifact.stages.push({ seed, training, evaluation, information: info, neuralWeightsUnchanged, arms: null });
}
artifact.gates = { information: artifact.stages.every(stage => stage.information.summary.passed),
  validity: null, raw: null, neural: null };
if (!artifact.gates.information) { await finish('stopped_information'); process.exit(0); }
for (const stage of artifact.stages) {
  stage.arms = { raw: trainArm('raw', stage.training, stage.evaluation),
    neural: trainArm('neural', stage.training, stage.evaluation) };
}
artifact.gates.validity = artifact.stages.every(stage => stage.neuralWeightsUnchanged &&
  Object.values(stage.arms).every(arm => arm.training.rows.length === 144 &&
    arm.finalWeights.n.reduce((sum, n) => sum + n, 0) === 144 &&
    arm.frozen.length === 5 && arm.frozen.every(point => point.weightsUnchanged &&
      point.rows.length === 36) &&
    arm.initialWeights.n.every(n => n === 0)));
if (!artifact.gates.validity) { await finish('stopped_validity'); process.exit(0); }
artifact.gates.raw = artifact.stages.every(stage => {
  const final = stage.arms.raw.frozen.at(-1).summary;
  return final.meanExpected >= .20 && final.actions.every(n => n >= 4);
});
artifact.gates.neural = artifact.stages.every(stage => {
  const raw = stage.arms.raw.frozen.at(-1).summary;
  const final = stage.arms.neural.frozen.at(-1).summary;
  const c2 = final.byCue[2];
  return final.meanExpected >= .20 && final.successes >= 22 &&
    final.actions.every(n => n >= 4) && c2.actions[2] / c2.n >= .60 &&
    raw.meanExpected - final.meanExpected <= .05;
});
await finish(artifact.gates.raw && artifact.gates.neural ? 'passed' : 'failed');
