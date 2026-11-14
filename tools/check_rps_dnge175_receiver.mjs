// Locked 680 ms teacher-first, frozen DNge175 neural receiving diagnostic.
// node tools/check_rps_dnge175_receiver.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { RpsContextualBandit } from '../js/rps.js';

const output = new URL('../data/rps_dnge175_receiver.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite DNge175 receiver result');
const read = path => JSON.parse(readFileSync(new URL(path, import.meta.url)));
const sub = read('../data/subgraph.json');
const context = read('../data/rps_contextual_bandit.json');
const source = read('../data/rps_hybrid_message_source.json');
const calibration = read('../data/rps_readout_information.json');
const threshold = 1.098235868125541;
if (!context.allValid || context.passed || source.passed || source.threshold !== threshold ||
    !calibration.passed) throw Error('Archived calibration or source result changed');

const targets = [3141, 3165];
if (targets.some(i => sub.neurons.role[i] !== 'dn' || sub.neurons.type[i] !== 'DNge175'))
  throw Error('DNge175 receiving targets changed');
const conditions = ['none', 'bit0', 'bit1'];
const featureMean = context.design.featureMean, featureStd = context.design.featureStd;
const archived = seed => {
  const student = context.students.find(x => x.seed === seed);
  const run = student?.runs.find(x => x.arm === 'contextual_bandit');
  if (!run || run.kcMbonChangedWeights !== 0 || !run.initialNeuralAllOne) throw Error('Archived trained state invalid');
  return run.finalController;
};
const restore = seed => {
  const net = new Network(sub, wiring(sub, 'real', seed), { seed });
  const agent = new RpsContextualBandit(net, { seed, featureMean, featureStd, readoutLearn: false });
  const saved = archived(seed);
  agent.W.forEach((w, a) => w.set(saved.W[a]));
  agent.b.set(saved.b);
  agent.baseline = saved.baseline;
  agent.explore = false;
  return { net, agent, saved };
};
const teacher = restore(711), student = restore(712);
if (teacher.net === student.net) throw Error('Teacher and student networks are not separate');
const snapshot = agent => ({ W: agent.W.map(w => [...w]), b: [...agent.b], baseline: agent.baseline });
const initial = { teacher: snapshot(teacher.agent), student: snapshot(student.agent) };
if (JSON.stringify(initial.teacher) !== JSON.stringify(teacher.saved) ||
    JSON.stringify(initial.student) !== JSON.stringify(student.saved)) throw Error('Controller reconstruction failed');
const mean = xs => xs.reduce((s, x) => s + x, 0) / xs.length;
const statePrediction = rates => {
  const x = rates.map(Math.log1p);
  const distances = calibration.centroids.map(c => c.reduce((d, v, j) => d + (x[j] - v) ** 2, 0));
  return distances.indexOf(Math.min(...distances));
};
const streamKey = (state, repeat) => 2026120400 + 1000 * state + 10 * repeat;

function teacherProbe(state, key) {
  const { net, agent } = teacher;
  net.resetState();
  net.rand = rng(key + 1);
  const cue = state === null ? [] : Array.from(net.laneVpn[state].slice(2));
  if (cue.length) net.setDrive(cue, 150);
  for (let t = 0; t < 240; t++) {
    if (t === 50) net.resetCounts();
    net.step();
  }
  if (cue.length) net.setDrive(cue, 0);
  if (state === null) return { bit: null, cue: null, scores: null, rates: null };
  const action = agent.decide(net);
  const scores = [...agent.scores];
  const spread = Math.max(...scores) - Math.min(...scores);
  return { bit: Number(spread > threshold), cue: state, scores, spread,
    rates: [...agent.rates], action };
}

function studentProbe(state, repeat, condition, key) {
  const { net, agent } = student;
  net.resetState();
  net.rand = rng(key + 2);
  const cue = state === null ? [] : Array.from(net.laneVpn[state].slice(2));
  if (cue.some(i => targets.includes(i))) throw Error('Cue and DN targets overlap');
  const stimulatedId = state === null || condition === 'none' ? null :
    targets[condition === 'bit0' ? 0 : 1];
  const dnWindowCounts = [0, 0]; // Actual spikes observed during 240-280 ms only.
  let windowCountsAt280, windowCountsAtDecision, decisionTargetCounts;
  let rates, features, scores, action, stateGuess, cueActiveAtDecision;
  let decidedAt = null, updateAt = null;
  for (let t = 0; t < 680; t++) {
    if (t === 240 && stimulatedId !== null) net.setDrive([stimulatedId], 300);
    if (t === 280) {
      windowCountsAt280 = [...dnWindowCounts];
      if (stimulatedId !== null) net.setDrive([stimulatedId], 0);
      net.rand = rng(key + 3); // Match cue-period random draws; retain all neural state.
      if (cue.length) net.setDrive(cue, 150);
    }
    if (t === 330) net.resetCounts();
    if (t === 520) {
      cueActiveAtDecision = cue.every(i => net.drive[i] === 150);
      if (cue.length) net.setDrive(cue, 0);
      windowCountsAtDecision = [...dnWindowCounts];
      decisionTargetCounts = targets.map(i => net.count[i]);
      action = agent.decide(net);
      rates = [...agent.rates]; features = [...agent.f]; scores = [...agent.scores];
      stateGuess = statePrediction(rates);
      decidedAt = t;
      net.resetCounts();
    }
    if (t === 620) { agent.afterDopamine(net, false); updateAt = t; }
    net.step();
    if (t >= 240 && t < 280) for (let k = 0; k < net.nSpikes; k++) {
      const i = net.spikes[k];
      if (i === targets[0]) dnWindowCounts[0]++;
      else if (i === targets[1]) dnWindowCounts[1]++;
    }
  }
  return { state, repeat, condition, key, stimulatedId, windowCountsAt280,
    windowCountsAtDecision, decisionTargetCounts, rates, features, scores, action,
    statePrediction: stateGuess, cueActiveAtDecision, decidedAt, updateAt };
}

const firstKey = 2026120390;
const firstTeacher = teacherProbe(null, firstKey);
const firstStudent = studentProbe(null, -1, 'none', firstKey);
const rows = [];
for (const state of [0, 1, 2]) for (let repeat = 0; repeat < 18; repeat++) {
  const key = streamKey(state, repeat);
  const observedTeacher = teacherProbe(state, key);
  for (const condition of conditions) {
    const row = studentProbe(state, repeat, condition, key);
    rows.push({ ...row, teacher: observedTeacher });
  }
}

const calibrationBitRows = rows.filter(r => r.condition !== 'none' && r.repeat < 6);
const heldoutRows = rows.filter(r => r.repeat >= 6);
const bitTestRows = heldoutRows.filter(r => r.condition !== 'none' && r.repeat < 12);
const bitCentroids = [0, 1].map(bit => {
  const xs = calibrationBitRows.filter(r => r.condition === `bit${bit}`);
  return [0, 1].map(j => mean(xs.map(r => r.windowCountsAtDecision[j])));
});
const byState = [0, 1, 2].map(state => {
  const all = rows.filter(r => r.state === state);
  const byCondition = Object.fromEntries(conditions.map(condition => {
    const xs = all.filter(r => r.condition === condition && r.repeat >= 6);
    return [condition, { cueCorrect: xs.filter(r => r.statePrediction === state).length,
      meanMbonRates: [0, 1, 2].map(j => mean(xs.map(r => r.rates[j]))),
      actionCounts: [0, 1, 2].map(a => xs.filter(r => r.action === a).length) }];
  }));
  const targetExtra = [0, 1].map(bit => {
    const driven = all.filter(r => r.condition === `bit${bit}` && r.repeat >= 6);
    const none = all.filter(r => r.condition === 'none' && r.repeat >= 6);
    return mean(driven.map(r => r.windowCountsAtDecision[bit])) -
      mean(none.map(r => r.windowCountsAtDecision[bit]));
  });
  const bitTest = bitTestRows.filter(r => r.state === state)
    .map(r => {
      const distances = bitCentroids.map(c => c.reduce((d, x, j) =>
        d + (r.windowCountsAtDecision[j] - x) ** 2, 0));
      const bit = r.condition === 'bit0' ? 0 : 1;
      return { repeat: r.repeat, bit, predicted: distances.indexOf(Math.min(...distances)), distances };
    });
  const bitCorrect = bitTest.filter(x => x.bit === x.predicted).length;
  const meanMbonDelta = [0, 1].map(bit => [0, 1, 2].map(j => mean(all
    .filter(r => r.condition === `bit${bit}` && r.repeat >= 6)
    .map(r => r.rates[j] - all.find(n => n.condition === 'none' && n.repeat === r.repeat).rates[j]))));
  const actionChanged = [0, 1].map(bit => all.filter(r => r.condition === `bit${bit}` && r.repeat >= 6)
    .filter(r => r.action !== all.find(n => n.condition === 'none' && n.repeat === r.repeat).action).length);
  return { state, byCondition, targetExtra, bitTest, bitCorrect,
    meanMbonDelta, actionChanged, sourceBitCounts: [0, 1].map(bit =>
      all.filter(r => r.condition === 'none' && r.teacher.bit === bit).length) };
});

const frozen = JSON.stringify(initial.teacher) === JSON.stringify(snapshot(teacher.agent)) &&
  JSON.stringify(initial.student) === JSON.stringify(snapshot(student.agent)) &&
  teacher.net.m.every(x => x === 1) && student.net.m.every(x => x === 1);
const validity = rows.length === 162 && firstTeacher.bit === null &&
  firstStudent.stimulatedId === null && firstStudent.cueActiveAtDecision &&
  calibrationBitRows.length === 36 && heldoutRows.length === 108 &&
  bitTestRows.length === 36 && [0, 1].every(b =>
    calibrationBitRows.filter(r => r.condition === `bit${b}`).length === 18) &&
  [0, 1, 2].every(s => conditions.every(c =>
    rows.filter(r => r.state === s && r.condition === c && r.repeat < 6).length === 6 &&
    rows.filter(r => r.state === s && r.condition === c && r.repeat >= 6).length === 12) &&
    [0, 1].every(b => bitTestRows.filter(r => r.state === s &&
      r.condition === `bit${b}`).length === 6)) &&
  rows.every(r => r.cueActiveAtDecision && r.decidedAt === 520 && r.updateAt === 620 &&
    r.windowCountsAt280.every((n, j) => n === r.windowCountsAtDecision[j]) &&
    r.rates.every(Number.isFinite) && r.action === r.scores.indexOf(Math.max(...r.scores)) &&
    r.teacher.bit === Number(r.teacher.spread > threshold) &&
    r.stimulatedId === (r.condition === 'none' ? null : targets[r.condition === 'bit0' ? 0 : 1])) && frozen;
const gates = {
  validity,
  physicalReach: byState.every(x => x.targetExtra.every(n => n >= 1)),
  decisionBit: byState.every(x => x.bitCorrect >= 10),
  cueRecognition: byState.every(x => ['bit0', 'bit1'].every(c => x.byCondition[c].cueCorrect >= 10)),
  cueNoLargeDrop: byState.every(x => ['bit0', 'bit1'].every(c =>
    x.byCondition.none.cueCorrect - x.byCondition[c].cueCorrect <= 2)),
  noWeightUpdates: frozen
};
const result = { command: 'node tools/check_rps_dnge175_receiver.mjs',
  design: { teacherSeed: 711, studentSeed: 712, repeats: 18, states: [0, 1, 2], conditions,
    streamFormula: 'key=2026120400+1000*state+10*repeat; teacher rng(key+1); student rng(key+2); student cue-period rng(key+3)',
    threshold, targets, messageHz: 300, messageWindowMs: [240, 280],
    teacherCueMs: [0, 240], studentCueMs: [280, 520], cueHz: 150,
    teacherCountResetMs: 50, studentCountResetMs: 330, decisionMs: 520,
    noLearningUpdateMs: 620, roundEndMs: 680,
    bitDecoder: 'one global nearest Euclidean centroid on raw two-DN 240-280 ms spike counts; pooled calibration repeats 0-5; held-out repeats 6-11, six/bit/cue',
    cueDecoder: 'prior fixed reward-free log1p three-MBON centroids; held-out repeats 6-17',
    gates: 'target +1 mean spike/cue/bit; bit >=10/12/cue; cue >=10/12/cue/bit and drop <=2/12; no weights change',
    sampleCounts: { studentProbes: 162, teacherResponses: 54, calibrationBitProbes: 36,
      heldoutStudentProbes: 108, heldoutBitProbes: 36, bitProbesPerCue: 12,
      cueProbesPerCueCondition: 12 } },
  initial, firstRound: { teacher: firstTeacher, student: firstStudent },
  bitCentroids, byState, gates, passed: Object.values(gates).every(Boolean), rows };
writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({ byState: byState.map(x => ({ state: x.state,
  targetExtra: x.targetExtra, bitCorrect: x.bitCorrect,
  cueCorrect: Object.fromEntries(conditions.map(c => [c, x.byCondition[c].cueCorrect])),
  actionChanged: x.actionChanged, meanMbonDelta: x.meanMbonDelta })), gates,
  passed: result.passed }, null, 2));
