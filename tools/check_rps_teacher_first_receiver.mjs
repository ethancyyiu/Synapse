// Locked 680 ms teacher-first, frozen RPS neural receiving diagnostic.
// node tools/check_rps_teacher_first_receiver.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { Opponent, RpsContextualBandit } from '../js/rps.js';

const output = new URL('../data/rps_teacher_first_receiver.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite teacher-first receiver result');
const read = path => JSON.parse(readFileSync(new URL(path, import.meta.url)));
const sub = read('../data/subgraph.json');
const context = read('../data/rps_contextual_bandit.json');
const source = read('../data/rps_hybrid_message_source.json');
const calibration = read('../data/rps_readout_information.json');
const threshold = 1.098235868125541;
if (!context.allValid || context.passed || source.passed || source.threshold !== threshold ||
    !calibration.passed) throw Error('Archived calibration or source result changed');

const targets = [[506, 545], [113, 1421]];
if (targets.flat().some(i => sub.neurons.role[i] !== 'kc')) throw Error('Message target is not a KC');
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
const sumIds = (net, ids) => ids.reduce((s, i) => s + net.count[i], 0);
const statePrediction = rates => {
  const x = rates.map(Math.log1p);
  const distances = calibration.centroids.map(c => c.reduce((d, v, j) => d + (x[j] - v) ** 2, 0));
  return distances.indexOf(Math.min(...distances));
};
const streamKey = (state, repeat) => 2026110400 + 1000 * state + 10 * repeat;

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
  if (cue.some(i => targets.flat().includes(i))) throw Error('Cue and KC targets overlap');
  const opponent = new Opponent({ strategy: 'previous_opponent', seed: 51000, offset: 1, noise: 0 });
  // Isolated observed-state fixture: the next move follows the unchanged opponent rule.
  if (state !== null) { opponent.previousOpponent = state; opponent.round = 1; }
  const opponentRoundBeforeDecision = opponent.round;
  let beforeTarget, windowTargetCounts, decisionTargetCounts, activeKcSpikes;
  let rates, features, scores, action, stateGuess, cueActiveAtDecision, opponentResult;
  for (let t = 0; t < 680; t++) {
    if (t === 240) {
      beforeTarget = targets.map(ids => sumIds(net, ids));
      if (condition !== 'none' && state !== null)
        net.setDrive(targets[condition === 'bit0' ? 0 : 1], 300);
    }
    if (t === 280) {
      windowTargetCounts = targets.map((ids, b) => sumIds(net, ids) - beforeTarget[b]);
      if (condition !== 'none' && state !== null)
        net.setDrive(targets[condition === 'bit0' ? 0 : 1], 0);
      net.rand = rng(key + 3); // Match cue-period random draws; retain all neural state.
      if (cue.length) net.setDrive(cue, 150);
    }
    if (t === 330) net.resetCounts();
    if (t === 520) {
      cueActiveAtDecision = cue.every(i => net.drive[i] === 150);
      if (cue.length) net.setDrive(cue, 0);
      decisionTargetCounts = targets.map(ids => sumIds(net, ids));
      activeKcSpikes = Array.from(net.idx.kc).filter(i => net.count[i] > 0)
        .map(i => [i, net.count[i]]);
      action = agent.decide(net);
      rates = [...agent.rates]; features = [...agent.f]; scores = [...agent.scores];
      stateGuess = statePrediction(rates);
      if (opponent.round !== opponentRoundBeforeDecision) throw Error('Opponent advanced before decision');
      opponentResult = opponent.play(action);
      agent.reward(net, opponentResult.reward);
      net.resetCounts();
    }
    if (t === 620) agent.afterDopamine(net, false);
    net.step();
  }
  return { state, repeat, condition, key, targetIds: condition === 'none' || state === null ? [] :
      targets[condition === 'bit0' ? 0 : 1], windowTargetCounts,
    decisionTargetCounts, activeKcSpikes, rates, features, scores, action,
    statePrediction: stateGuess, cueActiveAtDecision, opponentRoundBeforeDecision,
    opponentRoundAfterDecision: opponent.round, opponentResult };
}

const firstKey = 2026110390;
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

const byState = [0, 1, 2].map(state => {
  const all = rows.filter(r => r.state === state);
  const byCondition = Object.fromEntries(conditions.map(condition => {
    const xs = all.filter(r => r.condition === condition && r.repeat >= 6);
    return [condition, { cueCorrect: xs.filter(r => r.statePrediction === state).length,
      meanMbonRates: [0, 1, 2].map(j => mean(xs.map(r => r.rates[j]))),
      actionCounts: [0, 1, 2].map(a => xs.filter(r => r.action === a).length) }];
  }));
  const targetExtra = [0, 1].map(bit => {
    const driven = all.filter(r => r.condition === `bit${bit}`);
    const none = all.filter(r => r.condition === 'none');
    return mean(driven.map(r => r.windowTargetCounts[bit])) -
      mean(none.map(r => r.windowTargetCounts[bit]));
  });
  const bitCentroids = [0, 1].map(bit => {
    const xs = all.filter(r => r.condition === `bit${bit}` && r.repeat < 6);
    return [0, 1, 2].map(j => mean(xs.map(r => r.features[j])));
  });
  const bitTest = all.filter(r => r.condition !== 'none' && r.repeat >= 6 && r.repeat < 12)
    .map(r => {
      const distances = bitCentroids.map(c => c.reduce((d, x, j) => d + (r.features[j] - x) ** 2, 0));
      const bit = r.condition === 'bit0' ? 0 : 1;
      return { repeat: r.repeat, bit, predicted: distances.indexOf(Math.min(...distances)), distances };
    });
  const bitCorrect = bitTest.filter(x => x.bit === x.predicted).length;
  const meanMbonDelta = [0, 1].map(bit => [0, 1, 2].map(j => mean(all
    .filter(r => r.condition === `bit${bit}` && r.repeat >= 6)
    .map(r => r.rates[j] - all.find(n => n.condition === 'none' && n.repeat === r.repeat).rates[j]))));
  const actionChanged = [0, 1].map(bit => all.filter(r => r.condition === `bit${bit}` && r.repeat >= 6)
    .filter(r => r.action !== all.find(n => n.condition === 'none' && n.repeat === r.repeat).action).length);
  return { state, byCondition, targetExtra, bitCentroids, bitTest, bitCorrect,
    meanMbonDelta, actionChanged, sourceBitCounts: [0, 1].map(bit =>
      all.filter(r => r.condition === 'none' && r.teacher.bit === bit).length) };
});

const frozen = JSON.stringify(initial.teacher) === JSON.stringify(snapshot(teacher.agent)) &&
  JSON.stringify(initial.student) === JSON.stringify(snapshot(student.agent)) &&
  teacher.net.m.every(x => x === 1) && student.net.m.every(x => x === 1);
const validity = rows.length === 162 && firstTeacher.bit === null &&
  firstStudent.targetIds.length === 0 && firstStudent.cueActiveAtDecision &&
  rows.every(r => r.cueActiveAtDecision && r.opponentRoundAfterDecision ===
    r.opponentRoundBeforeDecision + 1 && r.rates.every(Number.isFinite) &&
    r.action === r.scores.indexOf(Math.max(...r.scores))) && frozen;
const gates = {
  validity,
  physicalReach: byState.every(x => x.targetExtra.every(n => n >= 1)),
  decisionBit: byState.every(x => x.bitCorrect >= 10),
  cueRecognition: byState.every(x => ['bit0', 'bit1'].every(c => x.byCondition[c].cueCorrect >= 10)),
  cueNoLargeDrop: byState.every(x => ['bit0', 'bit1'].every(c =>
    x.byCondition.none.cueCorrect - x.byCondition[c].cueCorrect <= 2)),
  noWeightUpdates: frozen
};
const result = { command: 'node tools/check_rps_teacher_first_receiver.mjs',
  design: { teacherSeed: 711, studentSeed: 712, repeats: 18, states: [0, 1, 2], conditions,
    streamFormula: 'key=2026110400+1000*state+10*repeat; teacher rng(key+1); student rng(key+2); student cue-period rng(key+3)',
    threshold, targets, messageHz: 300, messageWindowMs: [240, 280],
    teacherCueMs: [0, 240], studentCueMs: [280, 520], cueHz: 150,
    teacherCountResetMs: 50, studentCountResetMs: 330, decisionMs: 520,
    noLearningUpdateMs: 620, roundEndMs: 680,
    bitDecoder: 'per-cue nearest standardized-feature centroids; calibration repeats 0-5; held-out repeats 6-11, six/bit',
    cueDecoder: 'prior fixed reward-free log1p three-MBON centroids; held-out repeats 6-17',
    gates: 'target +1 mean spike/cue/bit; bit >=10/12/cue; cue >=10/12/cue/bit and drop <=2/12; no weights change' },
  initial, firstRound: { teacher: firstTeacher, student: firstStudent },
  byState, gates, passed: Object.values(gates).every(Boolean), rows };
writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({ byState: byState.map(x => ({ state: x.state,
  targetExtra: x.targetExtra, bitCorrect: x.bitCorrect,
  cueCorrect: Object.fromEntries(conditions.map(c => [c, x.byCondition[c].cueCorrect])),
  actionChanged: x.actionChanged, meanMbonDelta: x.meanMbonDelta })), gates,
  passed: result.passed }, null, 2));
