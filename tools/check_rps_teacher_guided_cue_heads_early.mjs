// Prospectively locked early retained-learning comparison; original runner untouched.
// node tools/check_rps_teacher_guided_cue_heads_early.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { Opponent, RpsContextualBandit, result } from '../js/rps.js';

const output = new URL('../data/rps_teacher_guided_cue_heads_early.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite early comparison');
const read = path => JSON.parse(readFileSync(new URL(path, import.meta.url)));
const sub = read('../data/subgraph.json');
const prior = read('../data/rps_teacher_guided_reset_preflight.json');
const routing = read('../data/rps_teacher_guided_cue_head_route_corrected.json');
const receiver = read('../data/rps_teacher_guided_between_round_reset.json');
const calibration = read('../data/rps_contextual_bandit.json').design;
const { featureMean, featureStd } = calibration;
const seeds = [841, 842], modes = ['correct', 'shuffled', 'untrained', 'disconnected', 'lookup'];
const evalSeeds = [52000, 52003, 52004], targets = [3141, 3165];
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const close = (a, b) => Math.abs(a - b) < 1e-10;
const counts = xs => [0, 1, 2].map(a => xs.filter(x => x === a).length);
const dot = (a, b) => a.reduce((s, x, j) => s + x * b[j], 0);
const correctAction = opponent => (opponent + 1) % 3;
const approve = (scores, candidate) => Number(scores[candidate] >
  Math.max(...scores.filter((_, a) => a !== candidate)));
const netFor = seed => new Network(sub, wiring(sub, 'real', seed), { seed });
const gameFor = seed => new Opponent({ strategy: 'previous_opponent', offset: 2, noise: 0, seed });
const makeHead = () => ({ W: [0, 1, 2].map(() => new Float64Array(3)), b: new Float64Array(3) });
const headCopy = h => ({ W: h.W.map(w => [...w]), b: [...h.b] });
const q = (h, f) => h.W.map((w, a) => h.b[a] + dot(w, f));

// The only new learner: route on own pre-message VPN spikes, then invoke the
// original epsilon-greedy decision and selected-action reward-prediction update.
class RpsCueHeadBandit extends RpsContextualBandit {
  constructor(net, seed) {
    super(net, { seed, featureMean, featureStd, readoutLearn: true });
    this.heads = [makeHead(), makeHead(), makeHead(), makeHead()]; // last: cue-free
    this.activeHead = null;
    this.laneCounts = null;
    this.kind = 'rps_contextual_bandit_cue_heads_opt_in';
  }
  decide(net) {
    this.laneCounts = net.laneVpn.map(ids => Array.from(ids.slice(2))
      .reduce((sum, i) => sum + net.count[i], 0));
    const maximum = Math.max(...this.laneCounts);
    if (maximum > 0 && this.laneCounts.filter(v => v === maximum).length !== 1)
      throw Error('Ambiguous student sensory-lane activity');
    this.activeHead = maximum === 0 ? 3 : this.laneCounts.indexOf(maximum);
    const h = this.heads[this.activeHead];
    this.W = h.W; this.b = h.b;
    return super.decide(net);
  }
}
const snapshot = a => ({ heads: a.heads.map(headCopy), baseline: a.baseline });
function load(a, params) {
  for (let h = 0; h < 4; h++) {
    for (let action = 0; action < 3; action++) a.heads[h].W[action].set(params.heads[h].W[action]);
    a.heads[h].b.set(params.heads[h].b);
  }
  a.baseline = params.baseline;
}
function summarize(rows) {
  const observed = rows.filter(r => r.cue !== null);
  return { wins: rows.filter(r => r.reward === 1).length,
    byCue: [0, 1, 2].map(cue => {
      const x = observed.filter(r => r.cue === cue);
      return { cue, n: x.length, wins: x.filter(r => r.reward === 1).length,
        actions: counts(x.map(r => r.action)),
        winsByAction: [0, 1, 2].map(a => x.filter(r => r.action === a && r.reward === 1).length),
        correctMarginMean: x.reduce((s, r) => s +
          r.scores[correctAction(r.opponent)] -
          Math.max(...r.scores.filter((_, a) => a !== correctAction(r.opponent))), 0) / x.length };
    }), actions: counts(observed.map(r => r.action)),
    winsByAction: [0, 1, 2].map(a => rows.filter(r => r.action === a && r.reward === 1).length),
    routed: observed.filter(r => r.selectedHead === r.cue).length,
    cueFree: rows[0]?.selectedHead === 3 };
}
function countDn(net, dnCounts) {
  for (let i = 0; i < net.nSpikes; i++) {
    if (net.spikes[i] === targets[0]) dnCounts[0]++;
    else if (net.spikes[i] === targets[1]) dnCounts[1]++;
  }
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
        else if (mode === 'shuffled') emittedBit = scheduledBit;
        else if (mode === 'untrained') emittedBit = 0;
        else if (mode === 'lookup') emittedBit = Number(candidate === lookup.predictedAction[cue]);
      }
      if (emittedBit !== null) {
        target = targets[emittedBit]; net.setDrive([target], 300);
      }
      row = { cue, key, laneCounts: [...agent.laneCounts], selectedHead: agent.activeHead,
        candidate, rates: [...agent.rates], features: [...agent.f], scores: [...agent.scores],
        teacherAdvice, emittedBit, dnCounts: null, decodedBit: null,
        action: null, reward: null, opponent: null, update: null };
    }
    if (t === 280) {
      if (target !== null) net.setDrive([target], 0);
      const decodedBit = target === null ? null :
        dnCounts[0] === dnCounts[1] ? 'tie' : Number(dnCounts[1] > dnCounts[0]);
      const alternative = [0, 1, 2].filter(a => a !== row.candidate);
      const action = decodedBit === 0 ? alternative[Math.floor(rng(key + 701)() * 2)] : row.candidate;
      agent.action = action;
      agent.V = Math.tanh(agent.rates[action] / agent.o.cRef);
      agent.elig = Float64Array.from(agent.plasticPre[action], p => net.trace[p]);
      row.dnCounts = [...dnCounts]; row.decodedBit = decodedBit; row.action = action;
      if (game) {
        const played = game.play(action);
        agent.reward(net, played.reward);
        row.opponent = played.opponent; row.reward = played.reward;
      }
      net.resetCounts();
    }
    if (t === 620 && game) {
      const before = snapshot(agent), active = row.selectedHead;
      const otherScores = [0, 1, 2, 3].filter(h => h !== active).map(h => q(agent.heads[h], row.features));
      agent.afterDopamine(net, learn);
      const after = snapshot(agent), action = row.action;
      const step = learn ? agent.alpha * (row.reward - row.scores[action]) : 0;
      if (before.heads.some((h, i) => i !== active && !same(h, after.heads[i])) ||
          !close(after.heads[active].b[action], before.heads[active].b[action] + step) ||
          row.features.some((v, j) => !close(after.heads[active].W[action][j],
            before.heads[active].W[action][j] + step * v)) ||
          [0, 1, 2].some(a => a !== action &&
            (!same(after.heads[active].W[a], before.heads[active].W[a]) ||
              after.heads[active].b[a] !== before.heads[active].b[a])) ||
          before.baseline !== after.baseline ||
          otherScores.some((scores, i) => !same(scores,
            q(agent.heads[[0, 1, 2, 3].filter(h => h !== active)[i]], row.features))))
        throw Error('Cue-head update deviated from the unchanged selected-action equation');
      row.update = { selectedHead: active, selectedAction: action, step,
        rewardedWeightChange: row.reward === 1 && Math.abs(step) > 1e-12,
        otherHeadScoreMaxChange: 0 };
    }
    net.step();
    if (t >= 240 && t < 280) countDn(net, dnCounts);
  }
  const expected = cue === null ? 3 : cue;
  if (row.selectedHead !== expected || row.features.length !== 3 ||
      row.features.some((v, j) => !close(v, (Math.log1p(row.rates[j]) - featureMean[j]) / featureStd[j])) ||
      game && (row.reward !== result(row.action, row.opponent) ||
        row.opponent !== (cue === null ? game.initial : (cue + 2) % 3)) ||
      net.m.some(v => v !== 1)) throw Error('Cue routing, sensory feature, reward, or synapse invalid');
  if (row.emittedBit !== null && (row.decodedBit !== row.emittedBit ||
      row.dnCounts[row.emittedBit] <= row.dnCounts[1 - row.emittedBit]))
    throw Error('Physical DN delivery failed');
  return row;
}
function liveState(net, agent, game) {
  return JSON.stringify({ net: { t: net.t, nSpikes: net.nSpikes,
    v: [...net.v], g: [...net.g], refr: [...net.refr], trace: [...net.trace],
    count: [...net.count], lastSpike: [...net.lastSpike], ring: net.ring.map(x => [...x]),
    drive: [...net.drive], noise: [...net.noise], noiseKick: [...net.noiseKick],
    spikes: [...net.spikes], m: [...net.m] },
    agent: { params: snapshot(agent), W: agent.W.map(x => [...x]), b: [...agent.b],
      action: agent.action, activeHead: agent.activeHead, laneCounts: agent.laneCounts,
      rates: agent.rates, f: agent.f, scores: agent.scores, V: agent.V, r: agent.r,
      delta: agent.delta, D: agent.D, elig: agent.elig && [...agent.elig],
      explore: agent.explore },
    game: { round: game.round, previousOpponent: game.previousOpponent,
      previousPlayer: game.previousPlayer, previousResult: game.previousResult } });
}
function episode(seed, params, sequenceSeed, episodeKey, learn, mode, sourceRows, bits, lookup,
  checkpoint = false) {
  const net = netFor(seed), agent = new RpsCueHeadBandit(net, seed);
  if (params) load(agent, params);
  const before = snapshot(agent), game = gameFor(sequenceSeed), rows = [];
  net.resetState(); agent.explore = learn;
  let middle = null, checkpointNoninvasive = null, middleParams = null;
  for (let round = 0; round < 24; round++) {
    if (round) net.resetState();
    const cue = game.observe().previousOpponent, sourceRow = sourceRows?.[round] ?? null;
    if (sourceRow && sourceRow.cue !== cue) throw Error('Teacher/student schedule mismatch');
    const key = seed * 10000000 + episodeKey * 1000 + round;
    rows.push({ round, ...studentRound(net, agent, cue, key, game, learn,
      mode, sourceRow, bits?.[round] ?? null, lookup) });
    if (checkpoint && round === 11) {
      middleParams = snapshot(agent);
      const stateBefore = liveState(net, agent, game);
      const randomBefore = [net.rand, agent.rand, game.rand];
      middle = frozen(seed, middleParams);
      checkpointNoninvasive = stateBefore === liveState(net, agent, game) &&
        randomBefore[0] === net.rand && randomBefore[1] === agent.rand &&
        randomBefore[2] === game.rand && same(middleParams, snapshot(agent));
      if (!checkpointNoninvasive) throw Error('Frozen midpoint changed live training state');
    }
  }
  const after = snapshot(agent);
  if (!learn && !same(before, after)) throw Error('Frozen Q weights changed');
  return { before, middleParams, middle, checkpointNoninvasive, after, rows,
    resets: 23, summary: summarize(rows),
    synapsesUnchanged: net.m.every(v => v === 1) };
}
function frozen(seed, params) {
  const sequences = evalSeeds.map((sequenceSeed, i) =>
    episode(seed, params, sequenceSeed, 100 + i, false, 'disconnected', null, null, null));
  const rows = sequences.flatMap((x, i) => x.rows.map(r => ({ sequenceSeed: evalSeeds[i], ...r })));
  if (rows.length !== 72 || !sequences.every(x => x.synapsesUnchanged && same(x.before, x.after)) ||
      [0, 1, 2].some(cue => rows.filter(r => r.cue === cue).length !== 23))
    throw Error('Frozen evaluation invalid');
  return { rows, summary: summarize(rows), weightsUnchanged: true };
}
function shuffledBits(bits, seed) {
  const x = bits.slice(1), random = rng(2026110601 + seed);
  for (let i = x.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [x[i], x[j]] = [x[j], x[i]];
  }
  return [null, ...x];
}
function shuffleQuality(rows) {
  const positive = rows.filter(r => r.cue !== null && r.candidate === correctAction(r.opponent));
  const negative = rows.filter(r => r.cue !== null && r.candidate !== correctAction(r.opponent));
  if (!positive.length || !negative.length) return null;
  return (positive.filter(r => r.emittedBit === 1).length / positive.length +
    negative.filter(r => r.emittedBit === 0).length / negative.length) / 2;
}
function targetUpdates(rows) {
  const byCue = [1, 2].map(cue => rows.filter(r => r.cue === cue && r.action === cue &&
    r.update.rewardedWeightChange).length);
  return { byCue, total: byCue[0] + byCue[1] };
}
const data = { protocol: {
  command: 'node tools/check_rps_teacher_guided_cue_heads_early.mjs',
  source: 'saved reset-preflight real-neural teacher 830 Q scores, approval of public proposal',
  students: seeds, modes, training: 'one 24-round Strategy B episode at opponent seed 41000',
  evaluation: '0/12/24 frozen disconnected 72 rounds, opponent seeds 52000/52003/52004',
  schedule: '150Hz VPN 0-240ms; count reset 50ms; VPN-spike head route and provisional action 240ms; 300Hz DN 240-280ms; action/reward 280ms; Q update 620ms; student-only reset between 680ms rounds',
  heads: 'three cue-specific three-action MBON-Q heads plus cue-free head; alpha .03, epsilon .20; original selected-action reward-prediction update',
  streams: 'archived key schedule; neural key+2, proposal key+501, rejection key+701, shuffle rng(2026110601+seed)',
  gates: { routeTraining: '23/23 plus cue-free', routeFrozen: '69/69 plus cue-free each sequence',
    receiver: 'archived passed receiving fixture; every actual message DN-decoded with target dominance',
    shuffle: '>=4/23 positions, bit counts preserved, <=60% balanced quality',
    initial: '<=36/72 and identical across arms',
    checkpoint: 'frozen in independent objects; live neural, controller, game, reward, and RNG state exactly unchanged',
    early: 'at round 12 correct-disconnected >=8/72, correct-shuffled/untrained >=4/72 each; all 3 mean correct Q margins >0; >=2 extra correctly rewarded target updates and >=1 for each target cue; both students' }
}, prerequisite: { source: 'data/rps_teacher_guided_cue_head_route_corrected.json',
  routingGates: routing.gates, receiverGates: receiver.gates }, students: [], stage: 0,
  status: 'not_started', passed: false };
function finish(stage, status, gates) {
  data.stage = stage; data.status = status; data.gates = gates;
  data.passed = status === 'passed_early_teaching';
  writeFileSync(output, `${JSON.stringify(data, null, 2)}\n`);
  console.log(JSON.stringify({ stage, status, gates,
    students: data.students.map(s => ({ seed: s.seed, initial: s.initial?.summary.wins,
      training: s.training && Object.fromEntries(Object.entries(s.training).map(([a, x]) =>
        [a, { wins: x.summary.wins, targetUpdates: x.targetUpdates }])),
      middle: s.middle && Object.fromEntries(Object.entries(s.middle).map(([a, x]) =>
        [a, { wins: x.summary.wins, cues: x.summary.byCue.map(c => c.wins) }])),
      final: s.final && Object.fromEntries(Object.entries(s.final).map(([a, x]) =>
        [a, { wins: x.summary.wins, cues: x.summary.byCue.map(c => c.wins) }])) })) }, null, 2));
}
const preflightGates = {
  archivedNegativeIntact: prior.status === 'failed_retained_learning' && prior.stage === 4 &&
    prior.students.length === 2,
  routing: routing.passed && routing.students.every(s => s.summary.observedCorrect === 23 &&
    s.summary.cueFreeCorrect && s.summary.archivedMatched),
  receiver: receiver.passed && receiver.summary.every(s => s.branches.every(b =>
    b.resetFirst >= 10 && b.resetSecond >= 10 &&
    b.resetFirst >= s.branches[0].resetFirst && b.resetSecond >= s.branches[0].resetSecond)),
  teacher: prior.teacher.source.training.length === 24 &&
    prior.teacher.frozen.summary.wins === 64 && prior.teacher.source.summary.length === 3,
  sourceSchedule: prior.teacher.source.training.every((r, i) =>
    r.cue === prior.students[0].training.correct.rows[i].cue),
  sourceFrozen: prior.teacher.source.training.every(r => r.scores.length === 3)
};
if (!Object.values(preflightGates).every(Boolean)) {
  finish(0, 'stopped_invalid_prerequisite', preflightGates);
  process.exit(0);
}
const sourceRows = prior.teacher.source.training, lookup = prior.lookup;
for (const seed of seeds) {
  const blankNet = netFor(seed), blank = new RpsCueHeadBandit(blankNet, seed);
  const initialParams = snapshot(blank), initial = frozen(seed, initialParams);
  const student = { seed, initialParams, initial, training: {}, middle: {}, final: {} };
  data.students.push(student);
}
const initialGates = {
  ...preflightGates,
  headroom: data.students.every(s => s.initial.summary.wins <= 36 &&
    s.initial.summary.routed === 69 && s.initial.summary.cueFree),
  frozenWeights: data.students.every(s => s.initial.weightsUnchanged),
  zeroHeads: data.students.every(s => s.initialParams.heads.every(h =>
    h.W.flat().every(v => v === 0) && h.b.every(v => v === 0)))
};
if (!Object.values(initialGates).every(Boolean)) {
  finish(1, 'stopped_initial', initialGates);
  process.exit(0);
}
for (const student of data.students) {
  const seed = student.seed;
  const correct = episode(seed, student.initialParams, 41000, 0, true, 'correct', sourceRows, null, lookup, true);
  student.training.correct = { ...correct, targetUpdates: targetUpdates(correct.rows),
    earlyTargetUpdates: targetUpdates(correct.rows.slice(0, 12)) };
  student.middle.correct = correct.middle;
  const bits = correct.rows.map(r => r.emittedBit), shuffled = shuffledBits(bits, seed);
  const shuffle = episode(seed, student.initialParams, 41000, 0, true, 'shuffled', sourceRows, shuffled, lookup, true);
  student.training.shuffled = { ...shuffle, targetUpdates: targetUpdates(shuffle.rows),
    earlyTargetUpdates: targetUpdates(shuffle.rows.slice(0, 12)) };
  student.middle.shuffled = shuffle.middle;
  const moved = bits.filter((v, i) => v !== shuffled[i]).length, quality = shuffleQuality(shuffle.rows);
  student.shuffle = { correctBits: bits, shuffledBits: shuffled, moved, quality,
    valid: moved >= 4 && same(counts(bits.slice(1)), counts(shuffled.slice(1))) &&
      quality !== null && quality <= .6 };
  if (!student.shuffle.valid) {
    finish(2, 'stopped_invalid_shuffle', { ...initialGates, shuffle: false });
    process.exit(0);
  }
  for (const mode of ['untrained', 'disconnected', 'lookup']) {
    const arm = episode(seed, student.initialParams, 41000, 0, true, mode, sourceRows, null, lookup, true);
    student.training[mode] = { ...arm, targetUpdates: targetUpdates(arm.rows),
      earlyTargetUpdates: targetUpdates(arm.rows.slice(0, 12)) };
    student.middle[mode] = arm.middle;
  }
}
const trainingGates = {
  complete: data.students.every(s => modes.every(mode => s.training[mode]?.rows.length === 24)),
  checkpointNoninvasive: data.students.every(s => modes.every(mode =>
    s.training[mode].checkpointNoninvasive && s.middle[mode]?.rows.length === 72 &&
    s.middle[mode].weightsUnchanged)),
  route: data.students.every(s => modes.every(mode =>
    s.training[mode].summary.routed === 23 && s.training[mode].summary.cueFree)),
  matching: data.students.every(s => modes.every(mode => same(
    s.training[mode].rows.map(r => [r.cue, r.opponent, r.rates, r.features, r.laneCounts]),
    s.training.correct.rows.map(r => [r.cue, r.opponent, r.rates, r.features, r.laneCounts])))),
  integrity: data.students.every(s => modes.every(mode => s.training[mode].synapsesUnchanged &&
    s.training[mode].resets === 23 && !same(s.training[mode].before, s.training[mode].after) &&
    s.training[mode].rows.every(r => r.update.selectedHead === r.selectedHead &&
      r.update.otherHeadScoreMaxChange === 0 &&
      (r.emittedBit === null || r.decodedBit === r.emittedBit)))),
  shuffle: data.students.every(s => s.shuffle.valid),
  midpointRoute: data.students.every(s => modes.every(mode =>
    s.middle[mode].summary.routed === 69 && s.middle[mode].summary.cueFree))
};
if (!Object.values(trainingGates).every(Boolean)) {
  finish(2, 'stopped_training_validity', { ...initialGates, ...trainingGates });
  process.exit(0);
}
for (const student of data.students) {
  for (const mode of modes) student.final[mode] = frozen(student.seed, student.training[mode].after);
}
const frozenGates = {
  complete: data.students.every(s => modes.every(mode =>
    s.final[mode]?.rows.length === 72 && s.final[mode].weightsUnchanged)),
  routing: data.students.every(s => modes.every(mode =>
    s.final[mode].summary.routed === 69 && s.final[mode].summary.cueFree)),
  matchedStreams: data.students.every(s => modes.every(mode => same(
    s.final[mode].rows.map(r => [r.sequenceSeed, r.round, r.cue, r.opponent,
      r.rates, r.features, r.laneCounts]),
    s.final.correct.rows.map(r => [r.sequenceSeed, r.round, r.cue, r.opponent,
      r.rates, r.features, r.laneCounts])))),
  midpointMatchedStreams: data.students.every(s => modes.every(mode => same(
    s.middle[mode].rows.map(r => [r.sequenceSeed, r.round, r.cue, r.opponent,
      r.rates, r.features, r.laneCounts]),
    s.middle.correct.rows.map(r => [r.sequenceSeed, r.round, r.cue, r.opponent,
      r.rates, r.features, r.laneCounts]))))
};
const earlyByStudent = data.students.map(s => {
  const wins = Object.fromEntries(modes.map(mode => [mode, s.middle[mode].summary.wins]));
  const correct = s.training.correct.earlyTargetUpdates;
  const disconnected = s.training.disconnected.earlyTargetUpdates;
  const comparisons = { disconnected: wins.correct - wins.disconnected,
    shuffled: wins.correct - wins.shuffled, untrained: wins.correct - wins.untrained };
  return { seed: s.seed, wins, comparisons,
    correctMargins: s.middle.correct.summary.byCue.map(x => x.correctMarginMean),
    targetUpdates: { correct, disconnected, difference: correct.total - disconnected.total },
    passed: comparisons.disconnected >= 8 && comparisons.shuffled >= 4 &&
      comparisons.untrained >= 4 &&
      s.middle.correct.summary.byCue.every(x => x.correctMarginMean > 0) &&
      correct.total - disconnected.total >= 2 &&
      correct.byCue.every((v, i) => v - disconnected.byCue[i] >= 1) };
});
data.earlyByStudent = earlyByStudent;
const technical = Object.values(preflightGates).every(Boolean) &&
  Object.values(initialGates).every(Boolean) && Object.values(trainingGates).every(Boolean) &&
  Object.values(frozenGates).every(Boolean);
finish(3, !technical ? 'stopped_frozen_validity' :
  earlyByStudent.every(x => x.passed) ? 'passed_early_teaching' : 'failed_early_teaching',
  { prerequisite: preflightGates, initial: initialGates, training: trainingGates,
    frozen: frozenGates, earlyByStudent });
