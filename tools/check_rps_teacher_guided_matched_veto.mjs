// Prospectively locked matched generic-veto comparison; original runner untouched.
// node tools/check_rps_teacher_guided_matched_veto.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { Opponent, RpsContextualBandit, result } from '../js/rps.js';

const output = new URL('../data/rps_teacher_guided_matched_veto.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite matched-veto comparison');
const read = path => JSON.parse(readFileSync(new URL(path, import.meta.url)));
const sub = read('../data/subgraph.json');
const prior = read('../data/rps_teacher_guided_reset_preflight.json');
const routing = read('../data/rps_teacher_guided_cue_head_route_corrected.json');
const receiver = read('../data/rps_teacher_guided_between_round_reset.json');
const calibration = read('../data/rps_contextual_bandit.json').design;
const { featureMean, featureStd } = calibration;
const seeds = [851, 852], modes = ['correct', 'generic', 'untrained', 'shuffled', 'disconnected'];
const evalSeeds = [52000, 52003, 52004], targets = [3141, 3165];
const vetoCueN = [3, 4, 4], vetoQuota = [1, 3, 3];
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
function genericPlan(seed) {
  const positions = vetoCueN.map((n, cue) => {
    const ordinals = Array.from({ length: n }, (_, i) => i + 1);
    const random = rng(2026110701 + 10 * seed + cue);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [ordinals[i], ordinals[j]] = [ordinals[j], ordinals[i]];
    }
    return ordinals.slice(0, vetoQuota[cue]).sort((a, b) => a - b);
  });
  return { positions, seen: [0, 0, 0] };
}

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
function studentRound(net, agent, cue, key, game, learn, mode, sourceRow, scheduledBit, genericState) {
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
      const teacherAdvice = mode === 'correct' && cue !== null && sourceRow ?
        approve(sourceRow.scores, candidate) : null;
      let emittedBit = null, genericOrdinal = null;
      if (mode === 'generic' && agent.activeHead !== 3) {
        const head = agent.activeHead;
        genericOrdinal = ++genericState.seen[head];
        if (genericOrdinal > vetoCueN[head]) throw Error('Generic cue-occurrence count exceeded calibration');
        emittedBit = Number(!genericState.positions[head].includes(genericOrdinal));
      } else if (cue !== null) {
        if (mode === 'correct') emittedBit = teacherAdvice;
        else if (mode === 'shuffled') emittedBit = scheduledBit;
        else if (mode === 'untrained') emittedBit = 0;
      }
      if (emittedBit !== null) {
        target = targets[emittedBit]; net.setDrive([target], 300);
      }
      row = { cue, key, laneCounts: [...agent.laneCounts], selectedHead: agent.activeHead,
        candidate, rates: [...agent.rates], features: [...agent.f], scores: [...agent.scores],
        teacherAdvice, emittedBit, genericOrdinal, dnCounts: null, decodedBit: null,
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
function episode(seed, params, sequenceSeed, episodeKey, learn, mode, sourceRows, bits,
  checkpoint = false) {
  const net = netFor(seed), agent = new RpsCueHeadBandit(net, seed);
  if (params) load(agent, params);
  const before = snapshot(agent), game = gameFor(sequenceSeed), rows = [];
  const genericState = mode === 'generic' ? genericPlan(seed) : null;
  const rounds = learn ? 12 : 24;
  net.resetState(); agent.explore = learn;
  let middle = null, checkpointNoninvasive = null, middleParams = null;
  for (let round = 0; round < rounds; round++) {
    if (round) net.resetState();
    const cue = game.observe().previousOpponent, sourceRow = sourceRows?.[round] ?? null;
    if (sourceRow && sourceRow.cue !== cue) throw Error('Teacher/student schedule mismatch');
    const key = seed * 10000000 + episodeKey * 1000 + round;
    rows.push({ round, ...studentRound(net, agent, cue, key, game, learn,
      mode, sourceRow, bits?.[round] ?? null, genericState) });
    if (checkpoint && round === 5) {
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
    genericPlan: genericState && { positions: genericState.positions, seen: genericState.seen },
    resets: rounds - 1, summary: summarize(rows),
    synapsesUnchanged: net.m.every(v => v === 1) };
}
function frozen(seed, params) {
  const sequences = evalSeeds.map((sequenceSeed, i) =>
    episode(seed, params, sequenceSeed, 100 + i, false, 'disconnected', null, null));
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

const calibrationArms = read('../data/rps_teacher_guided_cue_heads.json');
const calibrationCounts = calibrationArms.students.map(s => [0, 1, 2].map(cue => {
  const rows = s.training.correct.rows.slice(0, 12).filter(r => r.cue === cue);
  return { cue, n: rows.length, rejections: rows.filter(r => r.emittedBit === 0).length };
}));
const expectedPlans = [
  [[2], [1, 2, 3], [1, 3, 4]],
  [[3], [1, 2, 3], [1, 2, 4]]
];
const data = { protocol: {
  command: 'node tools/check_rps_teacher_guided_matched_veto.mjs',
  teacher: 830, students: seeds, opponent: 'Strategy B, seed 41000',
  arms: modes, trainingRounds: 12, checkpoints: [0, 6, 12],
  heldOut: { rounds: 72, opponentSeeds: evalSeeds, keys: [100, 101, 102], disconnected: true },
  source: 'archived continuous-neural-state teacher 830 Q scores; first 12 of 24 training rows',
  generic: 'non-teacher bits from fixed cue-occurrence veto positions, delivered through same measured DN spike path',
  vetoCueN, vetoQuota, vetoPositions: seeds.map(seed => genericPlan(seed).positions),
  streams: 'key=studentSeed*10000000+round; neural key+2, proposal key+501, alternative key+701; shuffle rng(2026110601+studentSeed); veto-position rng(2026110701+10*studentSeed+cue)',
  validity: { generic: 'exact 1/3/3 own-VPN-cue veto quota, correct/generic rejection difference <=1 per cue at 6 and 12, <=2 total at 12',
    actionDistribution: 'correct/generic played-action counts differ <=3 per action across 11 observed training rounds',
    shuffle: '>=2/11 bits moved, counts preserved, <=60% balanced quality',
    source: 'archived teacher 830 64/72 frozen, 24 saved real-neural Q rows, prior reset-receiver gate passed',
    route: '11/11 observed training, 69/69 frozen, cue-free first round',
    receiver: 'all actual DN bits decode with intended target dominance',
    learning: 'selected-head own-reward Q equation, no other head/synapse updates, frozen weights unchanged',
    checkpoint: 'independent simulator objects; exact live neural, controller, game, reward and RNG-reference state unchanged' },
  primary: 'at round 6 each correct arm exceeds generic/shuffled/untrained >=4/72, disconnected >=8/72, and has >=1 extra correctly rewarded cue1/paper or cue2/scissors update; both students plus all validity gates'
}, calibration: { students: [831, 832], counts: calibrationCounts },
  prerequisites: {}, students: [], gates: {}, status: 'not_started', passed: false };

function finish(status, gates) {
  data.status = status; data.gates = gates; data.passed = status === 'passed_primary';
  writeFileSync(output, `${JSON.stringify(data, null, 2)}\n`);
  console.log(JSON.stringify({ status, gates,
    students: data.students.map(s => ({ seed: s.seed, initial: s.initial?.summary.wins,
      veto: s.veto,
      arms: Object.fromEntries(modes.filter(mode => s.training[mode]).map(mode => {
        const t = s.training[mode];
        return [mode, { trainWins: t.summary.wins,
          wins6: s.middle[mode]?.summary.wins, wins12: s.final[mode]?.summary.wins,
          cueWins6: s.middle[mode]?.summary.byCue.map(x => x.wins),
          margins6: s.middle[mode]?.summary.byCue.map(x => x.correctMarginMean),
          targets6: t.targetUpdates6, targets12: t.targetUpdates12 }];
      })) })) }, null, 2));
}

const sourceRows = prior.teacher.source.training;
const prerequisite = data.prerequisites = {
  priorNegativeIntact: prior.status === 'failed_retained_learning' && prior.stage === 4,
  teacher: prior.teacher.frozen.summary.wins === 64 && sourceRows.length === 24 &&
    sourceRows.every(r => r.scores.length === 3),
  receiver: receiver.passed,
  routeCalibration: routing.passed,
  quotaCalibration: same(calibrationArms.students.map(s => s.seed), [831, 832]) &&
    same(calibrationCounts.map(xs => xs.map(x => x.n)), [[3, 4, 4], [3, 4, 4]]) &&
    same(calibrationCounts.map(xs => xs.map(x => x.rejections)), [[0, 3, 2], [1, 3, 4]]) &&
    same(vetoQuota, [0, 1, 2].map(cue =>
      Math.floor((calibrationCounts[0][cue].rejections +
        calibrationCounts[1][cue].rejections) / 2 + .5))),
  vetoPositions: seeds.every((seed, i) => same(genericPlan(seed).positions, expectedPlans[i])),
  sourceSchedule: sourceRows.slice(0, 12).every((r, i) =>
    r.cue === calibrationArms.students[0].training.correct.rows[i].cue)
};
if (!Object.values(prerequisite).every(Boolean)) {
  finish('stopped_prerequisite', { prerequisite });
  process.exit(0);
}
for (const seed of seeds) {
  const net = netFor(seed), agent = new RpsCueHeadBandit(net, seed);
  const initialParams = snapshot(agent), initial = frozen(seed, initialParams);
  data.students.push({ seed, initialParams, initial, training: {}, middle: {}, final: {} });
}
const initialGates = {
  headroom: data.students.every(s => s.initial.summary.wins <= 36),
  route: data.students.every(s => s.initial.summary.routed === 69 && s.initial.summary.cueFree),
  zeroHeads: data.students.every(s => s.initialParams.heads.every(h =>
    h.W.flat().every(v => v === 0) && h.b.every(v => v === 0))),
  frozenWeights: data.students.every(s => s.initial.weightsUnchanged)
};
if (!Object.values(initialGates).every(Boolean)) {
  finish('stopped_initial', { prerequisite, initialGates });
  process.exit(0);
}

for (const student of data.students) {
  const seed = student.seed;
  const correct = episode(seed, student.initialParams, 41000, 0, true,
    'correct', sourceRows, null, true);
  student.training.correct = correct;
  student.middle.correct = correct.middle;
  const generic = episode(seed, student.initialParams, 41000, 0, true,
    'generic', null, null, true);
  student.training.generic = generic;
  student.middle.generic = generic.middle;
  const untrained = episode(seed, student.initialParams, 41000, 0, true,
    'untrained', null, null, true);
  student.training.untrained = untrained;
  student.middle.untrained = untrained.middle;
  const bits = correct.rows.map(r => r.emittedBit), shuffled = shuffledBits(bits, seed);
  const shuffledArm = episode(seed, student.initialParams, 41000, 0, true,
    'shuffled', null, shuffled, true);
  student.training.shuffled = shuffledArm;
  student.middle.shuffled = shuffledArm.middle;
  const disconnected = episode(seed, student.initialParams, 41000, 0, true,
    'disconnected', null, null, true);
  student.training.disconnected = disconnected;
  student.middle.disconnected = disconnected.middle;
  for (const mode of modes) {
    const arm = student.training[mode];
    arm.targetUpdates6 = targetUpdates(arm.rows.slice(0, 6));
    arm.targetUpdates12 = targetUpdates(arm.rows);
  }
  const moved = bits.filter((x, i) => x !== shuffled[i]).length;
  student.shuffle = { correctBits: bits, shuffledBits: shuffled, moved,
    quality: shuffleQuality(shuffledArm.rows), preserved: same(counts(bits.slice(1)),
      counts(shuffled.slice(1))) };
  const rejectionCounts = (arm, rounds) => [0, 1, 2].map(cue =>
    arm.rows.slice(0, rounds).filter(r => r.selectedHead === cue && r.emittedBit === 0).length);
  const c6 = rejectionCounts(correct, 6), g6 = rejectionCounts(generic, 6),
    c12 = rejectionCounts(correct, 12), g12 = rejectionCounts(generic, 12);
  const actionCounts = arm => counts(arm.rows.filter(r => r.cue !== null).map(r => r.action));
  student.veto = { calibrationQuota: vetoQuota, positions: generic.genericPlan.positions,
    realizedSeen: generic.genericPlan.seen, correctReject6: c6, genericReject6: g6,
    correctReject12: c12, genericReject12: g12,
    correctActions: actionCounts(correct), genericActions: actionCounts(generic),
    quotaExact: same(g12, vetoQuota) && same(generic.genericPlan.seen, vetoCueN),
    rejectionMatch: [0, 1, 2].every(cue => Math.abs(c6[cue] - g6[cue]) <= 1 &&
      Math.abs(c12[cue] - g12[cue]) <= 1) &&
      Math.abs(c12.reduce((a, b) => a + b, 0) - g12.reduce((a, b) => a + b, 0)) <= 2,
    actionDistributionMatch: [0, 1, 2].every(a =>
      Math.abs(actionCounts(correct)[a] - actionCounts(generic)[a]) <= 3) };
}

const technical = {
  complete: data.students.every(s => modes.every(mode => s.training[mode].rows.length === 12 &&
    s.middle[mode].rows.length === 72)),
  checkpointNoninvasive: data.students.every(s => modes.every(mode =>
    s.training[mode].checkpointNoninvasive && s.middle[mode].weightsUnchanged)),
  trainingRoute: data.students.every(s => modes.every(mode =>
    s.training[mode].summary.routed === 11 && s.training[mode].summary.cueFree &&
    s.training[mode].resets === 11)),
  trainingMatchedInputs: data.students.every(s => modes.every(mode => same(
    s.training[mode].rows.map(r => [r.cue, r.opponent, r.rates, r.features, r.laneCounts]),
    s.training.correct.rows.map(r => [r.cue, r.opponent, r.rates, r.features, r.laneCounts])))),
  receiver: data.students.every(s => modes.every(mode => s.training[mode].rows.every(r =>
    r.emittedBit === null || r.decodedBit === r.emittedBit &&
      r.dnCounts[r.emittedBit] > r.dnCounts[1 - r.emittedBit]))),
  synapsesAndHeads: data.students.every(s => modes.every(mode =>
    s.training[mode].synapsesUnchanged && s.training[mode].rows.every(r =>
      r.update.selectedHead === r.selectedHead && r.update.otherHeadScoreMaxChange === 0))),
  genericNoTeacherQ: data.students.every(s => s.training.generic.rows.every(r =>
    r.teacherAdvice === null)),
  genericQuota: data.students.every(s => s.veto.quotaExact),
  genericRejectionMatch: data.students.every(s => s.veto.rejectionMatch),
  genericActionDistributionMatch: data.students.every(s => s.veto.actionDistributionMatch),
  shuffle: data.students.every(s => s.shuffle.moved >= 2 && s.shuffle.preserved &&
    s.shuffle.quality !== null && s.shuffle.quality <= .6)
};
for (const student of data.students) {
  for (const mode of modes)
    student.final[mode] = frozen(student.seed, student.training[mode].after);
}
const frozenGates = {
  complete: data.students.every(s => modes.every(mode =>
    s.final[mode].rows.length === 72 && s.final[mode].weightsUnchanged)),
  route: data.students.every(s => modes.every(mode =>
    s.middle[mode].summary.routed === 69 && s.final[mode].summary.routed === 69 &&
    s.middle[mode].summary.cueFree && s.final[mode].summary.cueFree)),
  matchedInputs: data.students.every(s => modes.every(mode =>
    ['middle', 'final'].every(checkpoint => same(
      s[checkpoint][mode].rows.map(r => [r.sequenceSeed, r.round, r.cue, r.opponent,
        r.rates, r.features, r.laneCounts]),
      s[checkpoint].correct.rows.map(r => [r.sequenceSeed, r.round, r.cue, r.opponent,
        r.rates, r.features, r.laneCounts])))))
};
const primary = data.students.map(s => {
  const wins6 = Object.fromEntries(modes.map(mode => [mode, s.middle[mode].summary.wins]));
  const wins12 = Object.fromEntries(modes.map(mode => [mode, s.final[mode].summary.wins]));
  const diff6 = Object.fromEntries(['generic', 'untrained', 'shuffled', 'disconnected'].map(mode =>
    [mode, wins6.correct - wins6[mode]]));
  const diff12 = Object.fromEntries(['generic', 'untrained', 'shuffled', 'disconnected'].map(mode =>
    [mode, wins12.correct - wins12[mode]]));
  const extraTargets6 = s.training.correct.targetUpdates6.total -
    s.training.disconnected.targetUpdates6.total;
  return { seed: s.seed, wins6, wins12, diff6, diff12, extraTargets6,
    passed: diff6.generic >= 4 && diff6.untrained >= 4 && diff6.shuffled >= 4 &&
      diff6.disconnected >= 8 && extraTargets6 >= 1 };
});
data.primary = primary;
const allValid = Object.values(prerequisite).every(Boolean) &&
  Object.values(initialGates).every(Boolean) && Object.values(technical).every(Boolean) &&
  Object.values(frozenGates).every(Boolean);
finish(!allValid ? 'invalid_control_or_technical' :
  primary.every(x => x.passed) ? 'passed_primary' : 'failed_primary',
  { prerequisite, initialGates, technical, frozenGates, primary });
