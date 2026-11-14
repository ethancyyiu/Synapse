// One locked, write-once synthetic package-recovery study. No existing experiment is edited.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { access, open, readFile } from 'node:fs/promises';
import { Network, rng, wiring } from '../js/sim.js';
import { LinUcbCore } from './package_recovery_linucb.mjs';
import { RecoveryBayesAgent, decodeDn, nearestCue, netReward } from './practical_transfer_controller.mjs';

const stage = process.argv[2];
if (!['initial', 'confirmation'].includes(stage)) throw Error('Use initial or confirmation');
const root = new URL('../', import.meta.url);
const read = async name => JSON.parse(await readFile(new URL(name, root)));
const output = new URL(`data/practical_transfer_${stage}.json`, root);
try { await access(output); throw Error('Write-once result already exists'); }
catch (e) { if (e.code !== 'ENOENT') throw e; }
if (stage === 'confirmation' && (await read('data/practical_transfer_initial.json')).status !== 'passed')
  throw Error('Initial stage did not pass');
const protocolSha256 = createHash('sha256').update(
  await readFile(new URL('docs/practical-transfer-protocol.md', root))).digest('hex');
const old = await read('data/package_recovery_initial.json');
const { cueProb, successProb, delays } = old.task;
const sub = await read('data/subgraph.json');
const { featureMean, featureStd } = (await read('data/rps_contextual_bandit.json')).design;
const config = stage === 'initial' ?
  { teacher: 17001, student: 17002, teacherTrain: 2026110951, teacherEval: 2026110952,
    studentTrain: 2026110953, studentEval: 2026110954, generic: 2026110959 } :
  { teacher: 17003, student: 17004, teacherTrain: 2026110955, teacherEval: 2026110956,
    studentTrain: 2026110957, studentEval: 2026110958, generic: 2026110960 };
const DN = [3141, 3165], CODES = [[1, 0], [0, 1], [1, 1]];
const teacherPoints = [0, 24, 48, 96, 144];
const studentPoints = [0, 6, 12, 24, 48, 96, 144];
const armNames = ['alone', 'untrained', 'shuffled', 'generic', 'trained'];
const artifact = { stage, protocolSha256, config, status: 'started', task: null,
  sensory: null, teacher: null, source: null, receiver: null, independent: null,
  comparison: null };
async function finish(status) {
  artifact.status = status;
  const file = await open(output, 'wx');
  try { await file.writeFile(JSON.stringify(artifact, null, 2) + '\n'); }
  finally { await file.close(); }
  console.log(JSON.stringify({ status, sensory: artifact.sensory?.gate,
    teacher: artifact.teacher?.gate, source: artifact.source?.gate,
    receiver: artifact.receiver?.gate, independent: artifact.independent?.gate,
    comparison: artifact.comparison?.gate }));
}
const key = (seed, phaseOffset, index) => seed * 10000000 + (phaseOffset + index) * 1000 + 1;
const expected = (mode, action) =>
  1.4 * successProb[mode][action] - .4 - .03 * delays[action] - (action === 2 ? .2 : 0);
const cueExpected = [0, 1, 2].map(cue => [0, 1, 2].map(action =>
  cueProb.reduce((s, row, mode) => s + row[cue] * expected(mode, action), 0) /
  cueProb.reduce((s, row) => s + row[cue], 0)));
const best = cueExpected.map(row => row.indexOf(Math.max(...row)));
function cases(seed, perMode) {
  const random = rng(seed), rows = [];
  for (let mode = 0; mode < 4; mode++) for (let j = 0; j < perMode; j++) {
    const u = random(); let cue = 0, cumulative = 0;
    while (cue < 2 && u >= (cumulative += cueProb[mode][cue])) cue++;
    rows.push({ id: `${seed}-${mode}-${j}`, mode, cue,
      outcomes: successProb[mode].map(p => Number(random() < p)) });
  }
  for (let i = rows.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [rows[i], rows[j]] = [rows[j], rows[i]];
  }
  return rows;
}
const teacherTrain = cases(config.teacherTrain, 36), teacherEval = cases(config.teacherEval, 9);
const studentTrain = cases(config.studentTrain, 36), studentEval = cases(config.studentEval, 9);
artifact.task = { cueProb, successProb, delays, cueExpected, best, teacherTrain, teacherEval,
  studentTrain, studentEval, gate: best.join() === '0,1,2' &&
    cueExpected.every(q => { const x = [...q].sort((a, b) => b - a); return x[0] - x[1] >= .03; }) &&
    cueProb.every(row => Math.abs(row.reduce((s, x) => s + x, 0) - 1) < 1e-12) &&
    teacherTrain.length === 144 && teacherEval.length === 36 &&
    studentTrain.length === 144 && studentEval.length === 36 };
if (!artifact.task.gate) { await finish('stopped_task'); process.exit(0); }
const makeNet = seed => new Network(sub, wiring(sub, 'real', seed), { seed });
function featureProbe(net, cue, streamKey) {
  net.resetState(); net.rand = rng(streamKey);
  const vpn = net.laneVpn[cue].slice(2); net.setDrive(vpn, 150);
  for (let t = 0; t < 240; t++) { if (t === 50) net.resetCounts(); net.step(); }
  net.setDrive(vpn, 0);
  const rates = net.groupMbon.map(ids => ids.reduce((s, id) => s + net.count[id], 0) / ids.length);
  const f = rates.map((v, j) => (Math.log1p(v) - featureMean[j]) / featureStd[j]);
  const lanes = net.laneVpn.map(ids => [...ids.slice(2)].reduce((s, id) => s + net.count[id], 0));
  const m = Math.max(...lanes);
  const routed = m > 0 && lanes.filter(x => x === m).length === 1 ? lanes.indexOf(m) : -1;
  return { cue, key: streamKey, rates, f, lanes, routed };
}
function sensory(seed) {
  const net = makeNet(seed), calibration = [], heldout = [];
  for (let cue = 0; cue < 3; cue++) for (let repeat = 0; repeat < 12; repeat++) {
    const row = { ...featureProbe(net, cue, key(seed, 2700, cue * 12 + repeat)), repeat };
    (repeat < 6 ? calibration : heldout).push(row);
  }
  const centroids = [0, 1, 2].map(cue => [0, 1, 2].map(j =>
    calibration.filter(r => r.cue === cue).reduce((s, r) => s + r.f[j], 0) / 6));
  for (const row of heldout) row.decodedCue = nearestCue(row.f, centroids);
  const byCue = [0, 1, 2].map(cue => ({ cue,
    routed: heldout.filter(r => r.cue === cue && r.routed === cue).length,
    recognized: heldout.filter(r => r.cue === cue && r.decodedCue === cue).length }));
  return { seed, centroids, calibration, heldout, byCue,
    gate: byCue.every(x => x.routed === 6 && x.recognized >= 4) &&
      byCue.reduce((s, x) => s + x.recognized, 0) >= 14 &&
      heldout.every(r => r.f.every(Number.isFinite)) && net.m.every(x => x === 1) };
}
const teacherSensory = sensory(config.teacher), studentSensory = sensory(config.student);
artifact.sensory = { teacher: teacherSensory, student: studentSensory,
  gate: teacherSensory.gate && studentSensory.gate };
if (!artifact.sensory.gate) { await finish('stopped_sensory'); process.exit(0); }

function agent(seed, centroids, state = null, frozen = false) {
  const net = makeNet(seed);
  const a = new RecoveryBayesAgent(net, { seed, featureMean, featureStd, centroids,
    readoutLearn: !frozen });
  if (state) a.core.restore(state);
  a.explore = !frozen;
  return { net, a, seed };
}
function round(s, spec, index, target, learn, isStudent = false, evaluation = false) {
  const { net, a, seed } = s, streamKey = key(seed, evaluation ? 900 : 0, index);
  const end = isStudent ? 680 : 400, counts = [0, 0];
  net.resetState(); net.rand = rng(streamKey);
  const vpn = net.laneVpn[spec.cue].slice(2); net.setDrive(vpn, 150);
  let row;
  for (let t = 0; t < end; t++) {
    if (t === 50) net.resetCounts();
    if (t === 240) {
      net.setDrive(vpn, 0);
      const action = a.decide(net);
      row = { index, id: spec.id, mode: spec.mode, cue: spec.cue, key: streamKey,
        decodedCue: a.decodedCue, rates: [...a.rates], f: [...a.f],
        action, scores: [...a.scores], selectionScores: [...a.selectionScores],
        coverage: a.coverage, target, counts: null, decoded: -1, imitation: 0,
        success: spec.outcomes[action], reward: netReward(spec.outcomes[action], action),
        expected: expected(spec.mode, action), delay: delays[action],
        escalation: action === 2 ? .2 : 0 };
      if (!isStudent) { a.reward(net, row.reward); net.resetCounts(); }
      else if (target !== null) DN.forEach((id, j) => {
        if (CODES[target][j]) net.setDrive([id], 300);
      });
    }
    if (t === 280 && isStudent) {
      net.setDrive(DN, 0);
      row.counts = [...counts]; row.decoded = decodeDn(counts);
      a.reward(net, row.reward); net.resetCounts();
    }
    if (t === 340 && !isStudent) a.afterDopamine(net, learn, row.success);
    if (t === 620 && isStudent) {
      a.afterDopamine(net, learn, row.success);
      row.imitation = a.imitate(row.decoded, learn);
    }
    net.step();
    if (isStudent && t >= 240 && t < 280) for (let i = 0; i < net.nSpikes; i++) {
      if (net.spikes[i] === DN[0]) counts[0]++;
      else if (net.spikes[i] === DN[1]) counts[1]++;
    }
  }
  if (isStudent) assert.equal(row.decoded, target ?? -1);
  assert.ok(net.m.every(x => x === 1));
  return row;
}
function summary(rows) {
  const byCue = [0, 1, 2].map(cue => {
    const q = rows.filter(r => r.cue === cue);
    return { cue, n: q.length, successes: q.filter(r => r.success).length,
      actions: [0, 1, 2].map(a => q.filter(r => r.action === a).length),
      meanExpected: q.length ? q.reduce((s, r) => s + r.expected, 0) / q.length : null };
  });
  return { n: rows.length, successes: rows.filter(r => r.success).length,
    failures: rows.filter(r => !r.success).length,
    actions: [0, 1, 2].map(a => rows.filter(r => r.action === a).length),
    meanReward: rows.reduce((s, r) => s + r.reward, 0) / rows.length,
    meanExpected: rows.reduce((s, r) => s + r.expected, 0) / rows.length,
    totalDelay: rows.reduce((s, r) => s + r.delay, 0),
    escalations: rows.filter(r => r.escalation > 0).length, byCue };
}
function frozen(s, specs, at, isStudent) {
  const copy = agent(s.seed, s.a.centroids, s.a.core.snapshot(), true);
  const before = JSON.stringify(copy.a.core.snapshot());
  const rows = specs.map((spec, i) => round(copy, spec, i, null, false, isStudent, true));
  assert.equal(JSON.stringify(copy.a.core.snapshot()), before);
  assert.ok(rows.every(r => r.imitation === 0 && (isStudent ?
    r.decoded === -1 && r.counts.every(x => x === 0) : true)));
  return { at, rows, summary: summary(rows), disconnected: isStudent,
    explorationOff: true, weightsUnchanged: true };
}
function train(s, specs, evaluation, points, isStudent, targets = null) {
  const initial = s.a.core.snapshot(), checkpoints = [frozen(s, evaluation, 0, isStudent)], rows = [];
  for (let i = 0; i < specs.length; i++) {
    rows.push(round(s, specs[i], i, targets ? targets[i] : null, true, isStudent));
    if (points.includes(i + 1)) checkpoints.push(frozen(s, evaluation, i + 1, isStudent));
  }
  return { initial, final: s.a.core.snapshot(), rows, training: summary(rows), checkpoints,
    neuralWeightsUnchanged: s.net.m.every(x => x === 1) };
}
const t = agent(config.teacher, teacherSensory.centroids);
const teacher = train(t, teacherTrain, teacherEval, teacherPoints, false);
const teacherFirst = teacher.checkpoints[0].summary, teacherLast = teacher.checkpoints.at(-1).summary;
// Historical LinUCB core on the exact neural input and case/outcome tapes; no extra neural run.
const ucb = new LinUcbCore(), ucbRows = [], ucbPoints = [];
const evalFeatures = teacher.checkpoints[0].rows.map(r => r.f);
function evalUcb(at) {
  const copy = new LinUcbCore(); copy.restore(ucb.snapshot());
  const before = JSON.stringify(copy.snapshot());
  const rows = teacherEval.map((spec, i) => {
    const d = copy.choose(evalFeatures[i], false), action = d.action;
    return { index: i, id: spec.id, cue: spec.cue, mode: spec.mode, action,
      success: spec.outcomes[action], reward: netReward(spec.outcomes[action], action),
      expected: expected(spec.mode, action), delay: delays[action], escalation: action === 2 ? .2 : 0,
      scores: d.means, f: evalFeatures[i] };
  });
  assert.equal(JSON.stringify(copy.snapshot()), before);
  return { at, rows, summary: summary(rows), weightsUnchanged: true };
}
ucbPoints.push(evalUcb(0));
for (let i = 0; i < 144; i++) {
  const spec = teacherTrain[i], d = ucb.choose(teacher.rows[i].f, true), action = d.action;
  const success = spec.outcomes[action], reward = netReward(success, action);
  ucbRows.push({ index: i, id: spec.id, cue: spec.cue, mode: spec.mode, action,
    f: teacher.rows[i].f, success, reward, expected: expected(spec.mode, action),
    delay: delays[action], escalation: action === 2 ? .2 : 0, scores: d.means,
    uncertainty: d.uncertainty });
  ucb.update(reward);
  if (teacherPoints.includes(i + 1)) ucbPoints.push(evalUcb(i + 1));
}
const teacherGate = teacherLast.meanExpected >= .20 &&
  teacherLast.meanExpected - teacherFirst.meanExpected >= .10 &&
  teacherLast.successes >= 22 && teacherLast.actions.every(n => n >= 4) &&
  teacherLast.byCue[2].actions[2] / teacherLast.byCue[2].n >= .60 &&
  JSON.stringify(teacher.initial) !== JSON.stringify(teacher.final) && teacher.neuralWeightsUnchanged;
artifact.teacher = { model: teacher, matchedLinUcb: { rows: ucbRows,
  training: summary(ucbRows), checkpoints: ucbPoints, final: ucb.snapshot() },
  gate: { passed: teacherGate, first: teacherFirst.meanExpected,
    final: teacherLast.meanExpected, successes: teacherLast.successes,
    actions: teacherLast.actions, c2Clear: teacherLast.byCue[2].actions[2],
    c2Total: teacherLast.byCue[2].n } };
if (!teacherGate) { await finish('stopped_teacher'); process.exit(0); }

const trainedSourceAgent = agent(config.teacher, teacherSensory.centroids, t.a.core.snapshot(), true);
const untrainedSourceAgent = agent(config.teacher, teacherSensory.centroids, null, true);
function sourcePreference(s, cue, index) {
  const probe = featureProbe(s.net, cue, key(config.teacher, 1800, index));
  const action = s.a.decide(s.net);
  return { ...probe, decodedCue: s.a.decodedCue, preference: action, scores: [...s.a.scores] };
}
const source = {
  probes: [0, 1, 2].flatMap(cue => Array.from({ length: 12 }, (_, i) =>
    sourcePreference(trainedSourceAgent, cue, 144 + cue * 12 + i))),
  trained: studentTrain.map((spec, i) => sourcePreference(trainedSourceAgent, spec.cue, i)),
  untrained: studentTrain.map((spec, i) => sourcePreference(untrainedSourceAgent, spec.cue, i)),
};
source.byCue = [0, 1, 2].map(cue => ({ cue,
  correct: source.probes.filter(r => r.cue === cue && r.preference === best[cue]).length }));
source.gate = source.byCue.every(x => x.correct >= 9) &&
  [0, 1, 2].every(a => source.trained.some(r => r.preference === a));
artifact.source = source;
if (!source.gate) { await finish('stopped_source'); process.exit(0); }

function receiverProbe(cue, repeat, target) {
  const s = agent(config.student, studentSensory.centroids, null, true);
  const net = s.net, counts = [0, 0], streamKey = key(config.student, 3600, cue * 6 + repeat);
  net.resetState(); net.rand = rng(streamKey);
  const vpn = net.laneVpn[cue].slice(2); net.setDrive(vpn, 150);
  const before = JSON.stringify(s.a.core.snapshot()); let pre;
  for (let time = 0; time < 680; time++) {
    if (time === 50) net.resetCounts();
    if (time === 240) {
      net.setDrive(vpn, 0); s.a.decide(net);
      const lanes = net.laneVpn.map(ids => [...ids.slice(2)].reduce((sum, id) => sum + net.count[id], 0));
      pre = { rates: [...s.a.rates], f: [...s.a.f], decodedCue: s.a.decodedCue,
        lanes, routed: lanes.indexOf(Math.max(...lanes)) };
      if (target !== null) DN.forEach((id, j) => {
        if (CODES[target][j]) net.setDrive([id], 300);
      });
    }
    if (time === 280) net.setDrive(DN, 0);
    net.step();
    if (time >= 240 && time < 280) for (let i = 0; i < net.nSpikes; i++) {
      if (net.spikes[i] === DN[0]) counts[0]++;
      else if (net.spikes[i] === DN[1]) counts[1]++;
    }
  }
  return { cue, repeat, target, key: streamKey, ...pre, counts, decoded: decodeDn(counts),
    weightsUnchanged: before === JSON.stringify(s.a.core.snapshot()) && net.m.every(x => x === 1) };
}
const receiverRows = [];
for (let cue = 0; cue < 3; cue++) for (let repeat = 0; repeat < 6; repeat++)
  for (const target of [null, 0, 1, 2]) receiverRows.push(receiverProbe(cue, repeat, target));
const receiverConditions = [0, 1, 2].flatMap(cue => [null, 0, 1, 2].map(target => {
  const rows = receiverRows.filter(r => r.cue === cue && r.target === target);
  const none = receiverRows.filter(r => r.cue === cue && r.target === null);
  const exact = rows.filter(r => {
    const n = none.find(x => x.repeat === r.repeat);
    return JSON.stringify([r.rates, r.f, r.decodedCue]) ===
      JSON.stringify([n.rates, n.f, n.decodedCue]);
  }).length;
  const passed = rows.length === 6 && exact === 6 && rows.every(r => r.routed === cue &&
    r.lanes[cue] > Math.max(...r.lanes.filter((_, i) => i !== cue))) &&
    rows.every(r => r.decoded === (target ?? -1) && r.weightsUnchanged &&
      (target === null ? r.counts.every(x => x === 0) :
        CODES[target].every((v, j) => v ? r.counts[j] > 0 : r.counts[j] === 0)));
  return { cue, target, n: rows.length, exact, decoded: rows.filter(r =>
    r.decoded === (target ?? -1)).length, passed };
}));
artifact.receiver = { rows: receiverRows, conditions: receiverConditions,
  gate: receiverConditions.every(x => x.passed) };
if (!artifact.receiver.gate) { await finish('stopped_receiver'); process.exit(0); }

function trainStudent(name, targets = null) {
  const s = agent(config.student, studentSensory.centroids);
  return train(s, studentTrain, studentEval, studentPoints, true, targets);
}
const alone = trainStudent('alone');
const a0 = alone.checkpoints[0].summary, af = alone.checkpoints.at(-1).summary;
const independentGate = af.meanExpected >= .20 && af.meanExpected - a0.meanExpected >= .10 &&
  af.successes >= 22 && af.actions.every(n => n >= 4) && alone.neuralWeightsUnchanged;
artifact.independent = { alone, gate: { passed: independentGate, first: a0.meanExpected,
  final: af.meanExpected, successes: af.successes, actions: af.actions } };
if (!independentGate) { await finish('stopped_independent'); process.exit(0); }

const genericRng = rng(config.generic);
const targets = {
  trained: source.trained.map(x => x.preference),
  untrained: source.untrained.map(x => x.preference),
  shuffled: source.trained.map((_, i) => source.trained[(i + 37) % 144].preference),
  generic: Array.from({ length: 144 }, () => Math.floor(3 * genericRng())),
};
const targetQuality = Object.fromEntries(Object.entries(targets).map(([name, tape]) =>
  [name, { actions: [0, 1, 2].map(a => tape.filter(x => x === a).length),
    aligned: tape.filter((a, i) => a === best[studentTrain[i].cue]).length,
    moved: name === 'shuffled' ? tape.filter((a, i) => a !== targets.trained[i]).length : null }]));
const arms = { alone };
for (const name of armNames.slice(1)) arms[name] = trainStudent(name, targets[name]);
const matched = armNames.every(name =>
  JSON.stringify(arms[name].initial) === JSON.stringify(alone.initial) &&
  arms[name].rows.length === 144 && arms[name].neuralWeightsUnchanged &&
  arms[name].rows.every((r, i) => r.id === alone.rows[i].id &&
    JSON.stringify(r.f) === JSON.stringify(alone.rows[i].f) &&
    r.decoded === (name === 'alone' ? -1 : targets[name][i])) &&
  arms[name].checkpoints.every(p => p.weightsUnchanged && p.explorationOff && p.disconnected));
const early = Object.fromEntries(armNames.map(name => [name,
  [24, 48, 96].reduce((sum, at) => sum +
    arms[name].checkpoints.find(p => p.at === at).summary.meanExpected, 0) / 3]));
const first20 = Object.fromEntries(armNames.map(name => [name,
  arms[name].checkpoints.find(p => p.summary.meanExpected >= .20)?.at ?? null]));
const trainedFinal = arms.trained.checkpoints.at(-1).summary;
const gate = matched && armNames.filter(n => n !== 'trained').every(name =>
  early.trained - early[name] >= .04 &&
  (first20[name] === null || first20.trained < first20[name])) &&
  first20.trained !== null && first20.trained <= 96 &&
  trainedFinal.meanExpected >= .20 && trainedFinal.successes >= 22 &&
  trainedFinal.actions.every(n => n >= 4);
artifact.comparison = { targets, targetQuality, arms,
  gate: { passed: gate, matched, early, first20, trainedFinal } };
await finish(gate ? 'passed' : 'failed');
