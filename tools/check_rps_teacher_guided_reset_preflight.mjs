// New, opt-in student-only between-round reset preflight. Refuses overwrite.
// node tools/check_rps_teacher_guided_reset_preflight.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { Opponent, RpsContextualBandit, result } from '../js/rps.js';

const output = new URL('../data/rps_teacher_guided_reset_preflight.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite reset preflight');
const read = path => JSON.parse(readFileSync(new URL(path, import.meta.url)));
const sub = read('../data/subgraph.json');
const calibration = read('../data/rps_contextual_bandit.json');
const cueCalibration = read('../data/rps_readout_information.json');
const prior = read('../data/rps_teacher_guided_exploration_preflight.json');
const resetFixture = read('../data/rps_teacher_guided_between_round_reset.json');
if (!calibration.allValid || !cueCalibration.passed) throw Error('Archived calibration invalid');
const { featureMean, featureStd } = calibration.design;
const teacherSeed = 830, studentSeeds = [831, 832], targets = [3141, 3165];
const teacherTrainSeeds = Array.from({ length: 12 }, (_, i) => 61000 + i);
const evalSeeds = [52000, 52003, 52004], studentTrainSeed = 41000;
const arms = ['correct', 'shuffled', 'untrained', 'disconnected', 'lookup'];
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const close = (a, b) => Math.abs(a - b) < 1e-10;
const counts = xs => [0, 1, 2].map(a => xs.filter(x => x === a).length);
const snapshot = a => ({ W: a.W.map(w => [...w]), b: [...a.b], baseline: a.baseline });
const netFor = seed => new Network(sub, wiring(sub, 'real', seed), { seed });
const agentFor = (net, seed) => new RpsContextualBandit(net,
  { seed, featureMean, featureStd, readoutLearn: true });
const gameFor = seed => new Opponent({ strategy: 'previous_opponent', offset: 2, noise: 0, seed });
const winningAction = opponent => (opponent + 1) % 3;
const approve = (scores, candidate) => Number(scores[candidate] >
  Math.max(...scores.filter((_, a) => a !== candidate)));
const cueGuess = rates => {
  const x = rates.map(Math.log1p);
  return cueCalibration.centroids.map(c =>
    c.reduce((s, v, j) => s + (x[j] - v) ** 2, 0)).indexOf(
      Math.min(...cueCalibration.centroids.map(c =>
        c.reduce((s, v, j) => s + (x[j] - v) ** 2, 0))));
};
const summarize = rows => {
  const observed = rows.filter(r => r.cue !== null);
  return { wins: rows.filter(r => r.reward === 1).length,
    byCue: [0, 1, 2].map(cue => {
      const x = observed.filter(r => r.cue === cue);
      return { cue, n: x.length, wins: x.filter(r => r.reward === 1).length,
        actions: counts(x.map(r => r.action)) };
    }), actions: counts(observed.map(r => r.action)) };
};
const protocol = {
  command: 'node tools/check_rps_teacher_guided_reset_preflight.mjs',
  teacherSeed, studentSeeds, teacherTrainSeeds, studentTrainSeed, evalSeeds, arms,
  mapping: 'fixed B: next=(previous+2)%3; no label supplied to a fly',
  teacherBudget: 'load archived reward-trained teacher 830; no teacher retraining',
  source: 'frozen 680ms real MBON Q; approve iff proposed Q exceeds both others',
  receiver: 'DNge175 3141 reject/0, 3165 approve/1; 300Hz 240-280ms; measured spike counts',
  student: 'one 24-round training episode, three MBON features and original alpha .03/epsilon .20 Q update',
  schedule: 'both cue 0-240ms at 150Hz, count reset 50ms, proposal 240ms, DN 240-280ms, final action/reward 280ms, Q update 620ms, end 680ms; student Network.resetState() only between rounds',
  streams: 'teacher base=830*1e7+episodeKey*1000+round; neural key+1, action key+501; student base=seed*1e7+episodeKey*1000+round; neural key+2, proposal key+501, fallback key+701; probe keys 200-235; evaluation keys 100-102; shuffle rng(2026110601+seed)',
  gates: { teacherWins: 58, teacherCueWins: 17, teacherEachAction: 12,
    sourceBalancedPerCue: .8, dnDominancePer12: 11, dnMeanGain: 1,
    cueRecognitionPer12: 10, maxCueLossPer12: 2, initialWinsMaximum: 36,
    shufflePositionsMinimum: 4, shuffleBalancedMaximum: .6,
    extraRewardedTargetUpdates: 4, ownBaselineGainPer72: 8,
    advantageOverEachRequiredControlPer72: 4,
    requiredControls: ['shuffled', 'untrained', 'disconnected'],
    lookup: 'benchmark only; not a formal target to beat' }
};
const data = { protocol, stage: 0, passed: false, status: 'not_started' };
function finish(stage, status, gates) {
  data.stage = stage; data.status = status; data.gates = gates;
  data.passed = status === 'passed';
  writeFileSync(output, `${JSON.stringify(data, null, 2)}\n`);
  console.log(JSON.stringify({ stage, status, gates,
    teacher: data.teacher && { policy: data.teacher.frozen.summary,
      source: data.teacher.source?.summary },
    receiver: data.receiver && data.receiver.summary,
    students: data.students && data.students.map(s => ({ seed: s.seed,
      initialWins: s.initialEvaluation?.summary.wins,
      training: s.training && Object.fromEntries(Object.entries(s.training).map(([arm, v]) =>
        [arm, { wins: v.summary.wins, targetRewardedUpdates: v.targetRewardedUpdates,
          adviceEffect: v.adviceEffect }])),
      final: s.final && Object.fromEntries(Object.entries(s.final).map(([arm, v]) =>
        [arm, v.summary])) })) }, null, 2));
  process.exit(0);
}
function load(agent, params) {
  agent.W.forEach((w, a) => w.set(params.W[a]));
  agent.b.set(params.b); agent.baseline = params.baseline;
}
function teacherSourceEpisode(net, agent, seed, episodeKey) {
  net.resetState(); agent.explore = false;
  const game = gameFor(seed), rows = [];
  for (let round = 0; round < 24; round++) {
    const cue = game.observe().previousOpponent;
    const vpn = cue === null ? [] : Array.from(net.laneVpn[cue].slice(2));
    const key = teacherSeed * 10000000 + episodeKey * 1000 + round;
    net.rand = rng(key + 1);
    if (vpn.length) net.setDrive(vpn, 150);
    let row;
    for (let t = 0; t < 680; t++) {
      if (t === 50) net.resetCounts();
      if (t === 240) {
        if (vpn.length) net.setDrive(vpn, 0);
        const action = agent.decide(net);
        row = { round, cue, rates: [...agent.rates], features: [...agent.f],
          scores: [...agent.scores], action,
          advice: cue === null ? null : [0, 1, 2].map(a => approve(agent.scores, a)) };
      }
      if (t === 280) {
        game.play(row.action); row.opponent = game.previousOpponent;
        row.correctAction = winningAction(row.opponent);
        net.resetCounts();
      }
      net.step();
    }
    rows.push(row);
  }
  if (rows.length !== 24 || rows.some(r => r.opponent !==
      (r.cue === null ? game.initial : (r.cue + 2) % 3)))
    throw Error('Teacher 680ms source mismatch');
  return rows;
}
function balancedAdvice(rows) {
  return [0, 1, 2].map(cue => {
    const x = rows.filter(r => r.cue === cue), positive = [], negative = [];
    for (const r of x) for (let a = 0; a < 3; a++)
      (a === r.correctAction ? positive : negative).push(r.advice[a]);
    const sensitivity = positive.reduce((s, v) => s + v, 0) / positive.length;
    const specificity = negative.filter(v => v === 0).length / negative.length;
    return { cue, events: x.length, sensitivity, specificity,
      balanced: (sensitivity + specificity) / 2,
      approvedWinning: positive.filter(Boolean).length,
      rejectedLosing: negative.filter(v => v === 0).length };
  });
}
function countDn(net, dnCounts) {
  for (let i = 0; i < net.nSpikes; i++) {
    if (net.spikes[i] === targets[0]) dnCounts[0]++;
    else if (net.spikes[i] === targets[1]) dnCounts[1]++;
  }
}
function lookupFrom(rows) {
  const table = [0, 1, 2].map(cue => counts(rows.filter(r => r.cue === cue).map(r => r.opponent)));
  if (table.some(row => row.every(v => v === 0))) throw Error('Lookup missing a cue');
  return { table, predictedOpponent: table.map(row => row.indexOf(Math.max(...row))),
    predictedAction: table.map(row => (row.indexOf(Math.max(...row)) + 1) % 3) };
}
function studentRound(net, agent, cue, key, game, learn, mode, sourceRow, scheduledBit, lookup) {
  const vpn = cue === null ? [] : Array.from(net.laneVpn[cue].slice(2));
  net.rand = rng(key + 2); agent.rand = rng(key + 501);
  if (vpn.length) net.setDrive(vpn, 150);
  const dnCounts = [0, 0];
  let row, target = null;
  for (let t = 0; t < 680; t++) {
    if (t === 50) net.resetCounts();
    if (t === 240) {
      if (vpn.length) net.setDrive(vpn, 0);
      const candidate = agent.decide(net);
      const teacherAdvice = cue === null || !sourceRow ? null : approve(sourceRow.scores, candidate);
      let emittedBit = null;
      if (cue !== null) {
        if (mode === 'correct') emittedBit = teacherAdvice;
        else if (mode === 'shuffled' || mode === 'force') emittedBit = scheduledBit;
        else if (mode === 'untrained') emittedBit = 0;
        else if (mode === 'lookup') emittedBit = Number(candidate === lookup.predictedAction[cue]);
      }
      if (emittedBit !== null) {
        target = targets[emittedBit]; net.setDrive([target], 300);
      }
      row = { cue, key, candidate, rates: [...agent.rates], features: [...agent.f],
        scores: [...agent.scores], cueGuess: cueGuess(agent.rates),
        teacherScores: sourceRow?.scores ?? null, teacherAdvice,
        emittedBit, dnCounts: null, decodedBit: null, action: null,
        reward: null, opponent: null, update: null };
    }
    if (t === 280) {
      if (target !== null) net.setDrive([target], 0);
      let decodedBit = null;
      if (target !== null) {
        if (dnCounts[0] !== dnCounts[1]) decodedBit = Number(dnCounts[1] > dnCounts[0]);
      }
      const alternative = [0, 1, 2].filter(a => a !== row.candidate);
      const action = decodedBit === 0 ? alternative[Math.floor(rng(key + 701)() * 2)] : row.candidate;
      agent.action = action;
      agent.V = Math.tanh(agent.rates[action] / agent.o.cRef);
      agent.elig = Float64Array.from(agent.plasticPre[action], p => net.trace[p]);
      row.dnCounts = [...dnCounts]; row.decodedBit = decodedBit; row.action = action;
      if (game) {
        const observed = game.play(action);
        agent.reward(net, observed.reward);
        row.opponent = observed.opponent; row.reward = observed.reward;
      }
      net.resetCounts();
    }
    if (t === 620 && game) {
      const before = snapshot(agent);
      agent.afterDopamine(net, learn);
      const after = snapshot(agent), a = row.action;
      const step = learn ? agent.alpha * (row.reward - row.scores[a]) : 0;
      if (!close(after.b[a], before.b[a] + step) ||
          row.features.some((v, j) => !close(after.W[a][j], before.W[a][j] + step * v)) ||
          [0, 1, 2].some(k => k !== a &&
            (!same(after.W[k], before.W[k]) || after.b[k] !== before.b[k])) ||
          after.baseline !== before.baseline) throw Error('Student selected-action update mismatch');
      row.update = { selectedAction: a, step,
        rewardedWeightChange: row.reward === 1 && Math.abs(step) > 1e-12 };
    }
    net.step();
    if (t >= 240 && t < 280) countDn(net, dnCounts);
  }
  if (!row || row.features.length !== 3 ||
      row.features.some((v, j) => !close(v, (Math.log1p(row.rates[j]) - featureMean[j]) / featureStd[j])) ||
      game && (row.reward !== result(row.action, row.opponent) ||
        row.opponent !== (cue === null ? game.initial : (cue + 2) % 3)) ||
      net.m.some(v => v !== 1)) throw Error('Student cue, advice, reward, or weight invalid');
  return row;
}
function studentEpisode(seed, params, sequenceSeed, episodeKey, learn, mode, sourceRows, bits, lookup) {
  const net = netFor(seed), agent = agentFor(net, seed);
  if (params) load(agent, params);
  const before = snapshot(agent);
  net.resetState(); agent.explore = learn;
  const game = gameFor(sequenceSeed), rows = [];
  for (let round = 0; round < 24; round++) {
    if (round > 0) net.resetState(); // Validated student-only boundary after the prior 680 ms round.
    const cue = game.observe().previousOpponent, sourceRow = sourceRows?.[round] ?? null;
    if (sourceRow && sourceRow.cue !== cue) throw Error('Teacher/student cue mismatch');
    const key = seed * 10000000 + episodeKey * 1000 + round;
    rows.push({ round, ...studentRound(net, agent, cue, key, game, learn,
      mode, sourceRow, bits?.[round] ?? null, lookup) });
  }
  const after = snapshot(agent);
  if (!learn && !same(before, after)) throw Error('Frozen student weights changed');
  return { before, after, rows, resets: 23, summary: summarize(rows),
    synapsesUnchanged: net.m.every(v => v === 1) };
}
function communicationSmoke(seed, sourceRows) {
  const net = netFor(seed), agent = agentFor(net, seed), before = snapshot(agent);
  net.resetState(); agent.explore = false;
  const rows = [];
  for (let round = 0; round < 24; round++) {
    if (round > 0) net.resetState();
    const key = seed * 10000000 + round;
    rows.push({ round, ...studentRound(net, agent, sourceRows[round].cue, key,
      null, false, 'correct', sourceRows[round], null, null) });
  }
  const valid = rows.length === 24 && rows.every((r, i) =>
    r.cue === sourceRows[i].cue &&
    r.emittedBit === (r.cue === null ? null : approve(sourceRows[i].scores, r.candidate)) &&
    (r.emittedBit === null ? r.action === r.candidate :
      r.decodedBit === r.emittedBit &&
      (r.decodedBit === 1 ? r.action === r.candidate : r.action !== r.candidate))) &&
    same(snapshot(agent), before) && net.m.every(v => v === 1);
  return { rows, resets: 23, valid };
}
function frozenStudent(seed, params) {
  const sequences = evalSeeds.map((sequenceSeed, i) =>
    studentEpisode(seed, params, sequenceSeed, 100 + i, false, 'disconnected', null, null, null));
  const rows = sequences.flatMap((v, i) => v.rows.map(r => ({ sequenceSeed: evalSeeds[i], ...r })));
  if (rows.length !== 72 || !sequences.every(v => same(v.before, v.after) && v.synapsesUnchanged) ||
      [0, 1, 2].some(c => rows.filter(r => r.cue === c).length !== 23))
    throw Error('Frozen student evaluation invalid');
  return { rows, summary: summarize(rows), weightsUnchanged: true };
}
function shuffleBits(bits, seed) {
  const shuffled = bits.slice(1), random = rng(2026110601 + seed);
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return [null, ...shuffled];
}
function adviceQuality(rows) {
  const positive = rows.filter(r => r.cue !== null && r.candidate === winningAction(r.opponent));
  const negative = rows.filter(r => r.cue !== null && r.candidate !== winningAction(r.opponent));
  if (!positive.length || !negative.length) return null;
  const sensitivity = positive.filter(r => r.emittedBit === 1).length / positive.length;
  const specificity = negative.filter(r => r.emittedBit === 0).length / negative.length;
  return { sensitivity, specificity, balanced: (sensitivity + specificity) / 2,
    positive: positive.length, negative: negative.length };
}
function targetRewardedUpdates(rows) {
  const x = rows.filter(r => r.cue !== null && [1, 2].includes(r.cue) &&
    r.action === r.cue && r.update?.rewardedWeightChange);
  return { total: x.length, byCue: [1, 2].map(cue => x.filter(r => r.cue === cue).length) };
}
function adviceEffect(rows) {
  const observed = rows.filter(r => r.cue !== null);
  const changed = observed.filter(r => r.action !== r.candidate);
  return { changed: changed.length,
    helpful: changed.filter(r => r.reward === 1 &&
      r.candidate !== winningAction(r.opponent)).length,
    harmful: changed.filter(r => r.reward !== 1 &&
      r.candidate === winningAction(r.opponent)).length,
    rewardedByCueAction: [0, 1, 2].map(cue => [0, 1, 2].map(action =>
      observed.filter(r => r.cue === cue && r.action === action &&
        r.update?.rewardedWeightChange).length)) };
}

// Stage 1: restore the already trained teacher; no teacher retraining.
const teacherNet = netFor(teacherSeed), teacherAgent = agentFor(teacherNet, teacherSeed);
const teacherInitial = snapshot(teacherAgent), teacherTraining = prior.teacher.training;
load(teacherAgent, prior.teacher.final);
const teacherFinal = snapshot(teacherAgent);
const teacherFrozenRows = prior.teacher.frozen.rows;
const teacherPolicySummary = summarize(teacherFrozenRows);
const teacherPolicyGates = {
  archived: prior.status === 'stopped_receiver' && prior.stage === 2 &&
    Object.values(prior.teacher.gates).every(Boolean),
  complete: teacherTraining.length === 12 && teacherTraining.every(v => v.rows.length === 24) &&
    teacherFrozenRows.length === 72,
  policyWins: teacherPolicySummary.wins >= 58,
  policyEachCue: teacherPolicySummary.byCue.every(v => v.n === 23 && v.wins >= 17),
  policyEachAction: teacherPolicySummary.actions.every(v => v >= 12),
  weights: !same(teacherInitial, teacherFinal) &&
    same(teacherFinal, prior.teacher.final) && teacherNet.m.every(v => v === 1)
};
data.teacher = { initial: teacherInitial, final: teacherFinal, training: teacherTraining,
  frozen: { rows: teacherFrozenRows, summary: teacherPolicySummary },
  policyGates: teacherPolicyGates };
if (!Object.values(teacherPolicyGates).every(Boolean))
  finish(1, 'stopped_teacher_policy', teacherPolicyGates);
const teacherSourceNet = netFor(teacherSeed), teacherSourceAgent = agentFor(teacherSourceNet, teacherSeed);
load(teacherSourceAgent, teacherFinal);
const teacherBeforeSource = snapshot(teacherSourceAgent);
const sourceRows = prior.teacher.source.rows;
const sourceSummary = balancedAdvice(sourceRows);
const teacherGates = {
  ...teacherPolicyGates,
  complete: teacherPolicyGates.complete && sourceRows.length === 72,
  sourceEachCue: sourceSummary.every(v => v.events === 23 && v.balanced >= .8),
  sameArchivedScores: same(teacherPolicySummary, prior.teacher.frozen.summary) &&
    same(sourceSummary, prior.teacher.source.summary),
  weights: teacherPolicyGates.weights && same(teacherBeforeSource, snapshot(teacherSourceAgent)) &&
    teacherSourceNet.m.every(v => v === 1)
};
data.teacher.source = { rows: sourceRows, summary: sourceSummary };
data.teacher.gates = teacherGates;
if (!Object.values(teacherGates).every(Boolean)) finish(1, 'stopped_teacher_readiness', teacherGates);

// Stage 2: reuse the write-once frozen receiver test, then check new-schedule headroom.
const receiverSummary = resetFixture.summary;
const receiverGates = {
  ...resetFixture.gates,
  archived: resetFixture.passed && resetFixture.rows.length === 216,
  cue: receiverSummary.every(s => s.branches.every(b => b.resetFirst >= 10 &&
    b.resetSecond >= 10 && b.resetFirst >= s.branches[0].resetFirst &&
    b.resetSecond >= s.branches[0].resetSecond)),
  receipt: receiverSummary.every(s => s.branches.slice(1).every(b =>
    b.decoded === 12 && b.targetDominates === 12 &&
    b.intendedMean - b.noMessageTargetMean >= 1))
};
data.receiver = { source: 'data/rps_teacher_guided_between_round_reset.json',
  summary: receiverSummary, gates: receiverGates };
if (!Object.values(receiverGates).every(Boolean)) finish(2, 'stopped_receiver', receiverGates);
data.students = studentSeeds.map(seed => {
  const net = netFor(seed), agent = agentFor(net, seed), initial = snapshot(agent);
  return { seed, initial, initialEvaluation: frozenStudent(seed, initial) };
});
const headroomGates = { initialScores: data.students.every(s =>
  s.initialEvaluation.summary.wins <= 36),
  initialMatched: data.students.every(s => s.initialEvaluation.rows.length === 72 &&
    s.initial.W.flat().every(v => v === 0) && s.initial.b.every(v => v === 0)) };
if (!Object.values(headroomGates).every(Boolean)) finish(2, 'stopped_headroom',
  { ...receiverGates, ...headroomGates });

// Real teacher Q and measured DN smoke, with no reward or student learning.
const sourceTrain = teacherSourceEpisode(teacherSourceNet, teacherSourceAgent,
  studentTrainSeed, 0);
if (!same(teacherBeforeSource, snapshot(teacherSourceAgent))) throw Error('Teacher source learned');
data.teacher.source.training = sourceTrain;
data.smoke = data.students.map(s => ({ seed: s.seed,
  ...communicationSmoke(s.seed, sourceTrain) }));
const smokeGates = { complete: data.smoke.every(s => s.rows.length === 24 && s.resets === 23),
  communication: data.smoke.every(s => s.valid && s.rows.filter(r => r.emittedBit !== null).length === 23),
  matchedCueSchedule: same(data.smoke[0].rows.map(r => r.cue),
    data.smoke[1].rows.map(r => r.cue)),
  teacherFrozen: same(teacherBeforeSource, snapshot(teacherSourceAgent)) &&
    teacherSourceNet.m.every(v => v === 1) };
if (!Object.values(smokeGates).every(Boolean)) finish(2, 'stopped_communication_smoke',
  { receiver: receiverGates, headroom: headroomGates, smoke: smokeGates });

// Stage 3: one matched, 24-round training episode per arm and student.
const lookup = lookupFrom(teacherTraining.flatMap(v => v.rows));
data.lookup = lookup;
for (const student of data.students) {
  const correct = studentEpisode(student.seed, student.initial, studentTrainSeed, 0,
    true, 'correct', sourceTrain, null, lookup);
  student.training = { correct: { ...correct,
    targetRewardedUpdates: targetRewardedUpdates(correct.rows),
    adviceEffect: adviceEffect(correct.rows) } };
  if (correct.rows.some(r => r.emittedBit !== null && r.decodedBit !== r.emittedBit))
    finish(3, 'stopped_invalid_dn_advice', { seed: student.seed, arm: 'correct' });
  const correctBits = correct.rows.map(r => r.emittedBit), shuffledBits = shuffleBits(correctBits, student.seed);
  const shuffled = studentEpisode(student.seed, student.initial, studentTrainSeed, 0,
    true, 'shuffled', sourceTrain, shuffledBits, lookup);
  student.training.shuffled = { ...shuffled,
    targetRewardedUpdates: targetRewardedUpdates(shuffled.rows),
    adviceEffect: adviceEffect(shuffled.rows) };
  if (shuffled.rows.some(r => r.emittedBit !== null && r.decodedBit !== r.emittedBit))
    finish(3, 'stopped_invalid_dn_advice', { seed: student.seed, arm: 'shuffled' });
  const moved = correctBits.filter((v, i) => v !== shuffledBits[i]).length;
  const quality = adviceQuality(shuffled.rows);
  student.shuffle = { correctBits, shuffledBits, moved, quality,
    valid: moved >= 4 && quality !== null && quality.balanced <= .6 &&
      counts(correctBits.slice(1)).every((v, i) => v === counts(shuffledBits.slice(1))[i]) };
  if (!student.shuffle.valid) finish(3, 'stopped_invalid_shuffle',
    { ...teacherGates, ...receiverGates, ...headroomGates, shuffle: false });
  for (const mode of ['untrained', 'disconnected', 'lookup']) {
    const arm = studentEpisode(student.seed, student.initial, studentTrainSeed, 0,
      true, mode, sourceTrain, null, lookup);
    student.training[mode] = { ...arm,
      targetRewardedUpdates: targetRewardedUpdates(arm.rows),
      adviceEffect: adviceEffect(arm.rows) };
    if (arm.rows.some(r => r.emittedBit !== null && r.decodedBit !== r.emittedBit))
      finish(3, 'stopped_invalid_dn_advice', { seed: student.seed, arm: mode });
  }
}
const trainingGates = {
  complete: data.students.every(s => arms.every(arm => s.training[arm]?.rows.length === 24)),
  shuffle: data.students.every(s => s.shuffle.valid),
  integrity: data.students.every(s => arms.every(arm => s.training[arm].synapsesUnchanged &&
    s.training[arm].resets === 23 &&
    !same(s.training[arm].before, s.training[arm].after) &&
    s.training[arm].rows.every(r => r.features.length === 3 &&
      (r.emittedBit === null || r.decodedBit === r.emittedBit)))),
  rewardedOpportunities: data.students.every(s => {
    const correct = s.training.correct.targetRewardedUpdates;
    const disconnected = s.training.disconnected.targetRewardedUpdates;
    return correct.total - disconnected.total >= 4 && correct.byCue.every(v => v >= 1);
  }) };
if (!Object.values(trainingGates).every(Boolean)) finish(3, 'stopped_training_mechanism',
  trainingGates);

// Stage 4: all frozen scores are disconnected and use the original cue-only Q heads.
for (const student of data.students) {
  student.final = Object.fromEntries(arms.map(arm =>
    [arm, frozenStudent(student.seed, student.training[arm].after)]));
}
const finalGates = {
  complete: data.students.every(s => arms.every(arm =>
    s.final[arm]?.rows.length === 72 && s.final[arm].weightsUnchanged)),
  ownImprovement: data.students.every(s => s.final.correct.summary.wins -
    s.initialEvaluation.summary.wins >= 8),
  advantageEveryControl: data.students.every(s =>
    ['shuffled', 'untrained', 'disconnected'].every(arm =>
      s.final.correct.summary.wins - s.final[arm].summary.wins >= 4)) };
finish(4, Object.values(finalGates).every(Boolean) ? 'passed' : 'failed_retained_learning',
  { teacher: teacherGates, receiver: receiverGates,
    headroom: headroomGates, training: trainingGates, final: finalGates });
