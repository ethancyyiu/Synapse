// Single locked exploratory hidden-strategy teaching pilot; refuses to overwrite its result.
// node tools/check_rps_hidden_strategy_teaching_pilot.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { Opponent, RpsContextualBandit, result } from '../js/rps.js';

const output = new URL('../data/rps_hidden_strategy_teaching_pilot.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite teaching pilot');
const read = path => JSON.parse(readFileSync(new URL(path, import.meta.url)));
const sub = read('../data/subgraph.json');
const original = read('../data/rps_independent_teachers.json');
const replication = read('../data/rps_strategy_b_replication.json');
const source = read('../data/rps_hidden_strategy_continuous_source.json');
const receiver = read('../data/rps_hidden_strategy_continuous_receiver.json');
const cueCalibration = read('../data/rps_readout_information.json');
const targets = [3141, 3165], dnScale = 7.388888888888889;
const { featureMean, featureStd } = original.design;
const teachers = { A: 783, B: 793 }, offsets = { A: 1, B: 2 };
const students = [{ seed: 821, strategy: 'A' }, { seed: 822, strategy: 'A' },
  { seed: 823, strategy: 'B' }, { seed: 824, strategy: 'B' }];
const arms = ['correct', 'shuffled', 'untrained', 'disconnected'];
const trainSeeds = Array.from({ length: 12 }, (_, i) => 41000 + i);
const evalSeeds = [52000, 52003, 52004], checkpoints = [0, 2, 4, 6, 8, 12];
const command = 'node tools/check_rps_hidden_strategy_teaching_pilot.mjs';
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const mean = xs => xs.reduce((s, x) => s + x, 0) / xs.length;
const snapshot = agent => ({ W: agent.W.map(w => [...w]), b: [...agent.b], baseline: agent.baseline });
const norm = (a, b) => Math.hypot(...a.W.flat().map((x, i) => b.W.flat()[i] - x),
  ...a.b.map((x, i) => b.b[i] - x));
const netFor = seed => new Network(sub, wiring(sub, 'real', seed), { seed });
const agentFor = (net, seed, withDn) => new RpsContextualBandit(net, { seed,
  featureMean, featureStd, ...(withDn ? { dnCountScale: dnScale } : {}), readoutLearn: true });
const opponent = (strategy, seed) => new Opponent({ strategy: 'previous_opponent',
  offset: offsets[strategy], noise: 0, seed });
const cuePrediction = rates => {
  const x = rates.map(Math.log1p);
  const distances = cueCalibration.centroids.map(c =>
    c.reduce((s, v, j) => s + (x[j] - v) ** 2, 0));
  return distances.indexOf(Math.min(...distances));
};
const bitOf = (scores, cue) => Number(scores[cue] >
  Math.max(...scores.filter((_, action) => action !== cue)));
const design = { teachers, students, arms, offsets, trainSeeds, evalSeeds, checkpoints,
  trainingRounds: 24, trainingEpisodes: 12, targets, dnHz: 300, dnWindowMs: [240, 280],
  cueHz: 150, cueWindowMs: [280, 520], countResetMs: 330, decisionMs: 520,
  updateMs: 620, roundEndMs: 680, dnScale, featureMean, featureStd,
  teacherBit: 'Q[cue] > max(other Q), ties zero',
  teacherStream: 'rng(teacherSeed*10000000 + episodeKey*1000 + round + 1)',
  studentStream: 'key=studentSeed*10000000 + episodeKey*1000 + round; network rng(key+2); cue-period network rng(key+3); action rng(key+501)',
  shuffle: 'Fisher-Yates pooled A/B within each cue, rng(2026110500+cue)',
  evaluation: '72 rounds, seeds 52000/52003/52004, episodeKeys 100/101/102; disconnected primary then common correct messages secondary',
  gates: { sourcePerCue: 0.8, shuffledBalancedMaximum: 0.6, dnCorrectTargetMinimum: 0.95,
    earlyMeanWinsAdvantagePer72: 4, earlyCheckpoints: [2, 4, 6, 8],
    reachEarlierByCheckpoints: 1, finalRetainedAdvantagePer72: 4,
    reliability: { winsPer72: 58, eachCueWinRate: 0.7, eachObservedActionSelections: 12 },
    aggregation: 'every student against every control; ties fail for speed and threshold timing' } };

if (!original.allValid || !replication.allValid || !source.passed || !receiver.passed ||
    !cueCalibration.passed || !same(original.design.featureMean, replication.design.featureMean) ||
    !same(original.design.featureStd, replication.design.featureStd) ||
    targets.some(i => sub.neurons.role[i] !== 'dn' || sub.neurons.type[i] !== 'DNge175') ||
    !original.teachers.some(t => t.seed === 783 && t.mapping === 'A') ||
    !replication.teachers.some(t => t.seed === 793 && t.mapping === 'B'))
  throw Error('Archived source, receiver, teacher, or interface check failed');

function countDn(net, counts) {
  for (let i = 0; i < net.nSpikes; i++) {
    if (net.spikes[i] === targets[0]) counts[0]++;
    else if (net.spikes[i] === targets[1]) counts[1]++;
  }
}

function teacherEpisode(net, agent, strategy, seed, sequenceSeed, episodeKey) {
  net.resetState(); agent.explore = false;
  const game = opponent(strategy, sequenceSeed), rows = [];
  for (let round = 0; round < 24; round++) {
    const cue = game.observe().previousOpponent;
    const vpn = cue === null ? [] : Array.from(net.laneVpn[cue].slice(2));
    net.rand = rng(seed * 10000000 + episodeKey * 1000 + round + 1);
    if (vpn.length) net.setDrive(vpn, 150);
    let row;
    for (let t = 0; t < 680; t++) {
      if (t === 50) net.resetCounts();
      if (t === 240) {
        if (vpn.length) net.setDrive(vpn, 0);
        if (cue !== null) {
          agent.decide(net);
          const scores = [...agent.scores], rates = [...agent.rates];
          row = { round, cue, bit: bitOf(scores, cue), rates, scores,
            action: agent.action, untrainedBit: 0 };
        } else row = { round, cue, bit: null, untrainedBit: null };
        game.play(cue === null ? 0 : agent.action); // Advance the fixed opponent, not the teacher weights.
        row.opponent = game.previousOpponent;
        net.resetCounts();
      }
      net.step();
    }
    if (!row) throw Error('Missing teacher decision');
    rows.push(row);
  }
  return rows;
}

function studentRound(net, agent, cue, bit, key, game, learn) {
  if (cue === null && bit !== null || bit !== null && bit !== 0 && bit !== 1)
    throw Error('Invalid or causally unavailable message');
  const cueIds = cue === null ? [] : Array.from(net.laneVpn[cue].slice(2));
  const dnCounts = [0, 0], target = bit === null ? null : targets[bit];
  net.rand = rng(key + 2); agent.rand = rng(key + 501);
  let row;
  for (let t = 0; t < 680; t++) {
    if (t === 240 && target !== null) net.setDrive([target], 300);
    if (t === 280) {
      if (target !== null) net.setDrive([target], 0);
      net.rand = rng(key + 3);
      if (cueIds.length) net.setDrive(cueIds, 150);
    }
    if (t === 330) net.resetCounts();
    if (t === 520) {
      if (cueIds.length) net.setDrive(cueIds, 0);
      agent.setDnWindowCounts([...dnCounts]);
      const action = agent.decide(net), rates = [...agent.rates], features = [...agent.f],
        scores = [...agent.scores], observed = game.play(action);
      agent.reward(net, observed.reward);
      net.resetCounts();
      row = { cue, bit, key, dnCounts: [...dnCounts], rates, features, scores,
        action, opponent: observed.opponent, reward: observed.reward,
        cuePrediction: cue === null ? null : cuePrediction(rates) };
    }
    if (t === 620) agent.afterDopamine(net, learn);
    net.step();
    if (t >= 240 && t < 280) countDn(net, dnCounts);
  }
  if (!row || row.reward !== result(row.action, row.opponent) ||
      !row.dnCounts.every((x, j) => row.features[j + 3] === x / dnScale) ||
      !row.rates.every((x, j) => Math.abs(row.features[j] -
        (Math.log1p(x) - featureMean[j]) / featureStd[j]) < 1e-10))
    throw Error('Student message, feature, or reward mismatch');
  return row;
}

function studentEpisode(net, agent, seed, strategy, sequenceSeed, episodeKey, bits, learn) {
  if (bits.length !== 24 || bits[0] !== null) throw Error('Invalid message schedule');
  net.resetState(); agent.explore = learn;
  const game = opponent(strategy, sequenceSeed), rows = [];
  for (let round = 0; round < 24; round++) {
    const cue = game.observe().previousOpponent;
    rows.push({ round, ...studentRound(net, agent, cue, bits[round],
      seed * 10000000 + episodeKey * 1000 + round, game, learn) });
  }
  return rows;
}

const summarize = rows => {
  const observed = rows.filter(r => r.cue !== null);
  return { wins: rows.filter(r => r.reward === 1).length,
    byCue: [0, 1, 2].map(cue => { const xs = observed.filter(r => r.cue === cue);
      return { cue, trials: xs.length, wins: xs.filter(r => r.reward === 1).length,
        actionCounts: [0, 1, 2].map(a => xs.filter(r => r.action === a).length),
        cueRecognized: xs.filter(r => r.cuePrediction === cue).length }; }),
    actionCounts: [0, 1, 2].map(a => observed.filter(r => r.action === a).length),
    meanDnCounts: [0, 1].map(j => mean(rows.map(r => r.dnCounts[j]))) };
};

// Build all frozen teacher recordings before student training.
const sourceRows = {}, teacherIntegrity = {};
for (const strategy of ['A', 'B']) {
  const seed = teachers[strategy], saved = [...original.teachers, ...replication.teachers]
    .find(t => t.seed === seed && t.mapping === strategy);
  const net = netFor(seed), agent = agentFor(net, seed, false);
  agent.W.forEach((w, a) => w.set(saved.final.W[a]));
  agent.b.set(saved.final.b); agent.baseline = saved.final.baseline;
  agent.readoutLearn = false;
  const before = snapshot(agent);
  const training = trainSeeds.map((sequenceSeed, i) =>
    teacherEpisode(net, agent, strategy, seed, sequenceSeed, i));
  const evaluation = evalSeeds.map((sequenceSeed, i) =>
    teacherEpisode(net, agent, strategy, seed, sequenceSeed, 100 + i));
  teacherIntegrity[strategy] = same(before, saved.final) && same(before, snapshot(agent)) &&
    net.m.every(x => x === 1);
  sourceRows[strategy] = { teacherSeed: seed, training, evaluation };
}

// One pooled, cue-stratified permutation. Assignment does not depend on rewards or student outcomes.
const shuffled = { A: trainSeeds.map(() => Array(24).fill(null)),
  B: trainSeeds.map(() => Array(24).fill(null)) };
for (const cue of [0, 1, 2]) {
  const slots = [];
  for (const strategy of ['A', 'B']) for (let episode = 0; episode < 12; episode++)
    for (let round = 1; round < 24; round++) {
      const row = sourceRows[strategy].training[episode][round];
      if (row.cue === cue) slots.push({ strategy, episode, round, bit: row.bit });
    }
  const bits = slots.map(x => x.bit), random = rng(2026110500 + cue);
  for (let i = bits.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [bits[i], bits[j]] = [bits[j], bits[i]];
  }
  slots.forEach((x, i) => { shuffled[x.strategy][x.episode][x.round] = bits[i]; });
}

const technicalPreflight = { teacherIntegrity,
  complete: ['A', 'B'].every(s => sourceRows[s].training.every(xs => xs.length === 24 &&
    xs[0].bit === null && xs.slice(1).every(x => [0, 1, 2].includes(x.cue))) &&
    sourceRows[s].evaluation.every(xs => xs.length === 24 && xs[0].bit === null)),
  sourceByStrategyCue: {}, shuffledByCue: {} };
for (const strategy of ['A', 'B']) for (const cue of [0, 1, 2]) {
  const xs = sourceRows[strategy].training.flat().filter(r => r.cue === cue);
  technicalPreflight.sourceByStrategyCue[`${strategy}${cue}`] = { trials: xs.length,
    expectedBit: strategy === 'B' ? 1 : 0,
    correctBits: xs.filter(r => r.bit === (strategy === 'B' ? 1 : 0)).length };
}
for (const cue of [0, 1, 2]) {
  const byStrategy = ['A', 'B'].map(strategy => {
    const xs = sourceRows[strategy].training.flat().filter(r => r.cue === cue);
    let correct = 0;
    for (let episode = 0; episode < 12; episode++) for (let round = 1; round < 24; round++)
      if (sourceRows[strategy].training[episode][round].cue === cue &&
          shuffled[strategy][episode][round] === (strategy === 'B' ? 1 : 0)) correct++;
    return { strategy, correct, trials: xs.length };
  });
  technicalPreflight.shuffledByCue[cue] = { byStrategy,
    balancedAccuracy: mean(byStrategy.map(x => x.correct / x.trials)) };
}
technicalPreflight.passed = Object.values(teacherIntegrity).every(Boolean) &&
  technicalPreflight.complete &&
  Object.values(technicalPreflight.sourceByStrategyCue).every(x =>
    x.correctBits / x.trials >= design.gates.sourcePerCue) &&
  Object.values(technicalPreflight.shuffledByCue).every(x =>
    x.balancedAccuracy <= design.gates.shuffledBalancedMaximum);
if (!technicalPreflight.passed) {
  const stopped = { command, design, status: 'stopped_before_student_training',
    technicalPreflight, sourceRows, shuffled };
  writeFileSync(output, `${JSON.stringify(stopped)}\n`);
  console.log(JSON.stringify({ status: stopped.status, technicalPreflight }));
  process.exit(0);
}

function evaluate(net, agent, student, arm, afterEpisodes, mode) {
  const before = snapshot(agent), all = [];
  for (let i = 0; i < 3; i++) {
    const source = sourceRows[student.strategy].evaluation[i];
    const bits = mode === 'correct' ? source.map(r => r.bit) : Array(24).fill(null);
    const rows = studentEpisode(net, agent, student.seed, student.strategy,
      evalSeeds[i], 100 + i, bits, false);
    if (rows.some((r, k) => r.cue !== source[k].cue || r.opponent !== source[k].opponent))
      throw Error('Frozen evaluation opponent alignment failed');
    all.push(...rows.map(r => ({ sequenceSeed: evalSeeds[i], ...r })));
  }
  const validity = { frozenController: same(before, snapshot(agent)),
    frozenSynapses: net.m.every(x => x === 1),
    greedy: all.every(r => r.action === r.scores.indexOf(Math.max(...r.scores))),
    balancedCues: same([0, 1, 2].map(c => all.filter(r => r.cue === c).length), [23, 23, 23]) };
  if (!Object.values(validity).every(Boolean)) throw Error('Frozen checkpoint validity failed');
  return { afterEpisodes, mode, arm, ...summarize(all), validity, rows: all };
}

const studentResults = [];
for (const student of students) {
  const net = netFor(student.seed), runs = [];
  for (const arm of arms) {
    const agent = agentFor(net, student.seed, true), initial = snapshot(agent);
    const curve = [], trainingRows = [], evaluations = [];
    for (const mode of ['disconnected', 'correct'])
      evaluations.push(evaluate(net, agent, student, arm, 0, mode));
    for (let episode = 0; episode < 12; episode++) {
      const source = sourceRows[student.strategy].training[episode];
      const bits = arm === 'correct' ? source.map(r => r.bit) :
        arm === 'shuffled' ? shuffled[student.strategy][episode] :
        arm === 'untrained' ? source.map(r => r.untrainedBit) : Array(24).fill(null);
      const rows = studentEpisode(net, agent, student.seed, student.strategy,
        trainSeeds[episode], episode, bits, true);
      if (rows.some((r, i) => r.cue !== source[i].cue || r.opponent !== source[i].opponent ||
          r.bit !== bits[i])) throw Error('Training opponent or message alignment failed');
      curve.push(rows.filter(r => r.reward === 1).length);
      trainingRows.push(...rows.map(r => ({ episode, ...r })));
      if (checkpoints.includes(episode + 1)) for (const mode of ['disconnected', 'correct'])
        evaluations.push(evaluate(net, agent, student, arm, episode + 1, mode));
    }
    runs.push({ arm, initial, final: snapshot(agent), controllerChangeNorm: norm(initial, snapshot(agent)),
      synapsesUnchanged: net.m.every(x => x === 1), curve, trainingRows, evaluations });
    console.log(`completed student ${student.seed} ${arm}`);
  }
  const correct = runs[0], controls = runs.slice(1);
  const e = (run, cp, mode = 'disconnected') => run.evaluations.find(x =>
    x.afterEpisodes === cp && x.mode === mode);
  const reliable = x => x.wins >= 58 && x.byCue.every(c => c.wins / c.trials >= 0.7) &&
    x.actionCounts.every(n => n >= 12);
  const firstReliable = run => checkpoints.find(cp => reliable(e(run, cp))) ?? Infinity;
  const early = run => mean([2, 4, 6, 8].map(cp => e(run, cp).wins));
  const earlyDifferences = Object.fromEntries(controls.map(run => [run.arm, early(correct) - early(run)]));
  const retainedDifferences = Object.fromEntries(controls.map(run =>
    [run.arm, e(correct, 12).wins - e(run, 12).wins]));
  const thresholdEpisodes = Object.fromEntries(runs.map(run => [run.arm, firstReliable(run)]));
  const dnIntegrity = Object.fromEntries(runs.filter(r => r.arm !== 'disconnected').map(run => {
    const byCue = [0, 1, 2].map(cue => {
      const xs = run.trainingRows.filter(r => r.cue === cue);
      return { cue, trials: xs.length, correctlyReceived: xs.filter(r =>
        r.dnCounts[r.bit] > r.dnCounts[1 - r.bit]).length };
    });
    return [run.arm, byCue];
  }));
  const validity = { matchedInitial: runs.every(r => same(r.initial, correct.initial)),
    initialZero: correct.initial.W.flat().every(x => x === 0) && correct.initial.b.every(x => x === 0),
    matchedTrainingSchedule: runs.every(r => same(r.trainingRows.map(x =>
      [x.episode, x.round, x.cue, x.opponent, x.key]), correct.trainingRows.map(x =>
      [x.episode, x.round, x.cue, x.opponent, x.key]))),
    matchedFrozenSchedule: runs.every(r => same(r.evaluations.map(v =>
      [v.afterEpisodes, v.mode, ...v.rows.map(x => `${x.sequenceSeed}/${x.round}/${x.cue}/${x.opponent}/${x.key}`)]),
      correct.evaluations.map(v => [v.afterEpisodes, v.mode, ...v.rows.map(x =>
        `${x.sequenceSeed}/${x.round}/${x.cue}/${x.opponent}/${x.key}`)]))),
    initialDisconnectedIdentical: runs.every(r => same(e(r, 0).rows.map(x =>
      [x.action, x.reward, x.rates]), e(correct, 0).rows.map(x => [x.action, x.reward, x.rates]))),
    counts: runs.every(r => r.trainingRows.length === 288 && r.evaluations.length === 12 &&
      r.evaluations.every(v => v.rows.length === 72)),
    frozenCheckpoints: runs.every(r => r.evaluations.every(v =>
      Object.values(v.validity).every(Boolean))),
    frozenSynapses: runs.every(r => r.synapsesUnchanged),
    controllerLearned: runs.every(r => r.controllerChangeNorm > 0),
    ownRewards: runs.every(r => r.trainingRows.every(x => x.reward === result(x.action, x.opponent))),
    dnReceipt: Object.values(dnIntegrity).every(byCue => byCue.every(x =>
      x.correctlyReceived / x.trials >= design.gates.dnCorrectTargetMinimum)),
    disconnectedZeroDn: runs.find(r => r.arm === 'disconnected').trainingRows.every(r =>
      r.dnCounts[0] === 0 && r.dnCounts[1] === 0) };
  const gates = { earlySpeed: Object.values(earlyDifferences).every(x => x >= 4),
    thresholdTiming: Number.isFinite(firstReliable(correct)) && controls.every(r =>
      firstReliable(r) === Infinity ||
      checkpoints.indexOf(firstReliable(correct)) + 1 <= checkpoints.indexOf(firstReliable(r))),
    finalReliable: reliable(e(correct, 12)),
    retention: Object.values(retainedDifferences).every(x => x >= 4) };
  studentResults.push({ ...student, runs, earlyDifferences, retainedDifferences,
    thresholdEpisodes, dnIntegrity, validity, gates,
    passed: Object.values(validity).every(Boolean) && Object.values(gates).every(Boolean) });
}

const allValid = studentResults.every(s => Object.values(s.validity).every(Boolean));
const gates = { earlySpeed: studentResults.every(s => s.gates.earlySpeed),
  thresholdTiming: studentResults.every(s => s.gates.thresholdTiming),
  finalReliable: studentResults.every(s => s.gates.finalReliable),
  retention: studentResults.every(s => s.gates.retention) };
const data = { command, design, status: 'complete', technicalPreflight, sourceRows, shuffled,
  studentResults, allValid, gates, passed: allValid && Object.values(gates).every(Boolean) };
writeFileSync(output, `${JSON.stringify(data)}\n`);
console.log(JSON.stringify({ status: data.status, allValid, gates, passed: data.passed,
  students: studentResults.map(s => ({ seed: s.seed, strategy: s.strategy, validity: s.validity,
    gates: s.gates, earlyDifferences: s.earlyDifferences,
    retainedDifferences: s.retainedDifferences, thresholdEpisodes: s.thresholdEpisodes,
    arms: s.runs.map(r => ({ arm: r.arm, curve: r.curve,
      disconnected: r.evaluations.filter(x => x.mode === 'disconnected').map(x =>
        [x.afterEpisodes, x.wins]), connected: r.evaluations.filter(x =>
        x.mode === 'correct').map(x => [x.afterEpisodes, x.wins]),
      final: r.evaluations.find(x => x.afterEpisodes === 12 && x.mode === 'disconnected'),
      controllerChangeNorm: r.controllerChangeNorm })) })) },
  (k, v) => k === 'rows' || k === 'trainingRows' ? undefined : v, 2));
