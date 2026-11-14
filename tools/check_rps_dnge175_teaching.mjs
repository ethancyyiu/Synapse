// One locked exploratory hybrid RPS teaching-speed comparison.
// node tools/check_rps_dnge175_teaching.mjs [pilot]
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { Opponent, RpsContextualBandit, result } from '../js/rps.js';

const pilot = process.argv[2] === 'pilot';
if (process.argv[2] && !pilot) throw Error('Unknown experiment mode');
const command = `node tools/check_rps_dnge175_teaching.mjs${pilot ? ' pilot' : ''}`;
const output = new URL(pilot ? '../data/rps_dnge175_teaching_pilot.json' :
  '../data/rps_dnge175_teaching.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite DNge175 teaching result');
const read = path => JSON.parse(readFileSync(new URL(path, import.meta.url)));
const sub = read('../data/subgraph.json');
const context = read('../data/rps_contextual_bandit.json');
const receiver = read('../data/rps_dnge175_receiver.json');
const cueCalibration = read('../data/rps_readout_information.json');
const threshold = 1.098235868125541, targets = [3141, 3165];
if (!context.allValid || context.passed || !receiver.passed || !cueCalibration.passed ||
    receiver.design.threshold !== threshold ||
    JSON.stringify(receiver.design.targets) !== JSON.stringify(targets) ||
    targets.some(i => sub.neurons.role[i] !== 'dn' || sub.neurons.type[i] !== 'DNge175'))
  throw Error('Archived protocol or receiving calibration changed');
const dnScale = (receiver.bitCentroids[0][0] + receiver.bitCentroids[1][1]) / 2;
if (dnScale !== 7.388888888888889) throw Error('DN calibration scale changed');
const featureMean = context.design.featureMean, featureStd = context.design.featureStd;
const pairs = pilot ? [[751, 712], [752, 712]] : [[741, 712], [742, 713], [743, 714]];
const smokeBase = pilot ? 2026150400 : 2026130400;
const smokeStudentBase = pilot ? 751 : 741;
const arms = ['correct', 'shuffled', 'untrained', 'disconnected'];
const trainSeeds = Array.from({ length: 12 }, (_, i) => 41000 + i);
const finalSeeds = [51000, 51001, 51002], checkpointSeed = 52000;
const checkpoints = [2, 4, 6], rounds = 24;
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const mean = xs => xs.reduce((s, x) => s + x, 0) / xs.length;
const counts = xs => [0, 1, 2].map(a => xs.filter(x => x === a).length);
const controller = agent => ({ W: agent.W.map(w => [...w]), b: [...agent.b], baseline: agent.baseline });
const norm = (a, b) => Math.hypot(...a.W.flat().map((v, i) => b.W.flat()[i] - v),
  ...a.b.map((v, i) => b.b[i] - v));
const approx = (a, b) => Math.abs(a - b) < 1e-10;
const opponent = seed => new Opponent({ strategy: 'previous_opponent', offset: 1, noise: 0, seed });
const cuePrediction = rates => {
  const x = rates.map(Math.log1p);
  const distances = cueCalibration.centroids.map(c => c.reduce((s, v, j) => s + (x[j] - v) ** 2, 0));
  return distances.indexOf(Math.min(...distances));
};
const archived = seed => {
  const run = context.students.find(x => x.seed === seed)?.runs.find(x => x.arm === 'contextual_bandit');
  if (!run || !run.initialNeuralAllOne || run.kcMbonChangedWeights !== 0 ||
      !run.finalController || run.after.wins < 58) throw Error(`Unavailable trained teacher ${seed}`);
  return run;
};
const netFor = seed => new Network(sub, wiring(sub, 'real', seed), { seed });
const agentFor = (net, seed, withDn) => new RpsContextualBandit(net, { seed,
  featureMean, featureStd, ...(withDn ? { dnCountScale: dnScale } : {}), readoutLearn: true });
const studentKey = (seed, episodeKey, round) => seed * 10000000 + 1000 * episodeKey + round;

// One step of the receiving window: retain two measured spike counts, never the source bit.
function countDnSpikes(net, windowCounts) {
  for (let k = 0; k < net.nSpikes; k++) {
    if (net.spikes[k] === targets[0]) windowCounts[0]++;
    else if (net.spikes[k] === targets[1]) windowCounts[1]++;
  }
}

function studentRound(net, agent, cue, bit, key, game, learn) {
  if (cue === null && bit !== null) throw Error('First round must have no message');
  if (bit !== null && bit !== 0 && bit !== 1) throw Error('Invalid message');
  const cueIds = cue === null ? [] : Array.from(net.laneVpn[cue].slice(2));
  const dnCounts = [0, 0];
  const target = bit === null ? null : targets[bit];
  net.rand = rng(key + 2); agent.rand = rng(key + 501);
  let row, updateValid = true;
  for (let t = 0; t < 680; t++) {
    if (t === 240 && target !== null) net.setDrive([target], 300);
    if (t === 280) {
      if (target !== null) net.setDrive([target], 0);
      net.rand = rng(key + 3); // Match cue-period draws, preserving all neural state.
      if (cueIds.length) net.setDrive(cueIds, 150);
    }
    if (t === 330) net.resetCounts();
    if (t === 520) {
      const cueActive = cueIds.every(i => net.drive[i] === 150);
      if (cueIds.length) net.setDrive(cueIds, 0);
      const measuredCounts = [...dnCounts];
      agent.setDnWindowCounts(measuredCounts);
      const action = agent.decide(net);
      const rates = [...agent.rates], features = [...agent.f], scores = [...agent.scores];
      const observed = game.play(action);
      agent.reward(net, observed.reward);
      net.resetCounts();
      row = { cue, bit, key, dnCounts: measuredCounts, rates, features, scores,
        action, opponent: observed.opponent, reward: observed.reward,
        cuePrediction: cue === null ? null : cuePrediction(rates), cueActive };
    }
    if (t === 620) {
      const before = controller(agent), a = agent.action, error = agent.r - agent.scores[a];
      agent.afterDopamine(net, learn);
      const after = controller(agent), step = learn ? agent.alpha * error : 0;
      updateValid &&= approx(after.b[a], before.b[a] + step) &&
        [0, 1, 2].every(k => k === a || same(before.W[k], after.W[k]) && before.b[k] === after.b[k]) &&
        row.features.every((f, j) => approx(after.W[a][j], before.W[a][j] + step * f)) &&
        after.baseline === before.baseline;
      row.dopamine = agent.D;
    }
    net.step();
    if (t >= 240 && t < 280) countDnSpikes(net, dnCounts);
  }
  if (!row || !row.cueActive || row.features.length !== 5 ||
      row.features.some(x => !Number.isFinite(x)) ||
      !row.dnCounts.every((x, j) => row.features[j + 3] === x / dnScale) ||
      !row.rates.every((x, j) => approx(row.features[j],
        (Math.log1p(x) - featureMean[j]) / featureStd[j])) ||
      row.reward !== result(row.action, row.opponent) || !updateValid)
    throw Error('Student round input or reward-update validity failed');
  return row;
}

function studentEpisode(net, agent, seed, sequenceSeed, episodeKey, bits, learn) {
  if (bits.length !== rounds || bits[0] !== null) throw Error('Invalid first-round message');
  net.resetState(); agent.explore = learn;
  const game = opponent(sequenceSeed), rows = [];
  for (let round = 0; round < rounds; round++) {
    const cue = game.observe().previousOpponent;
    const row = studentRound(net, agent, cue, bits[round], studentKey(seed, episodeKey, round), game, learn);
    rows.push({ round, ...row });
  }
  return rows;
}

function teacherEpisode(net, trained, untrained, seed, sequenceSeed, episodeKey) {
  net.resetState(); trained.explore = false; untrained.explore = false;
  const game = opponent(sequenceSeed), rows = [];
  for (let round = 0; round < rounds; round++) {
    const cue = game.observe().previousOpponent;
    const cueIds = cue === null ? [] : Array.from(net.laneVpn[cue].slice(2));
    const key = seed * 10000000 + 1000 * episodeKey + round;
    net.rand = rng(key + 1);
    if (cueIds.length) net.setDrive(cueIds, 150);
    for (let t = 0; t < 680; t++) {
      if (t === 50) net.resetCounts();
      if (t === 240) {
        if (cueIds.length) net.setDrive(cueIds, 0);
        const action = trained.decide(net);
        untrained.decide(net);
        const scores = [...trained.scores], untrainedScores = [...untrained.scores];
        const spread = Math.max(...scores) - Math.min(...scores);
        const untrainedSpread = Math.max(...untrainedScores) - Math.min(...untrainedScores);
        const bit = cue === null ? null : Number(spread > threshold);
        const untrainedBit = cue === null ? null : Number(untrainedSpread > threshold);
        const played = game.play(action);
        rows.push({ round, cue, opponent: played.opponent, bit, untrainedBit,
          rates: [...trained.rates], scores, spread, untrainedScores });
      }
      net.step();
    }
  }
  if (rows.length !== rounds || rows[0].bit !== null || rows[0].untrainedBit !== null ||
      rows.slice(1).some(r => r.untrainedBit !== 0 || r.untrainedScores.some(q => q !== 0)))
    throw Error('Teacher recording or untrained controller invalid');
  return rows;
}

function shuffled(bits, teacherSeed, episodeIndex) {
  const xs = bits.slice(1), random = rng(2026140400 + 1000 * teacherSeed + episodeIndex);
  for (let i = xs.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [xs[i], xs[j]] = [xs[j], xs[i]];
  }
  return [null, ...xs];
}

const summarize = rows => {
  const observed = rows.filter(r => r.cue !== null);
  return { wins: rows.filter(r => r.reward === 1).length,
    byCue: [0, 1, 2].map(cue => {
      const xs = observed.filter(r => r.cue === cue);
      return { cue, trials: xs.length, wins: xs.filter(r => r.reward === 1).length };
    }), actionCounts: counts(observed.map(r => r.action)),
    meanDnCounts: [0, 1].map(j => mean(observed.map(r => r.dnCounts[j]))) };
};

function smokeProbe(net, agent, cue, condition, key) {
  net.resetState(); net.rand = rng(key + 2); agent.explore = false;
  const cueIds = cue === null ? [] : Array.from(net.laneVpn[cue].slice(2));
  const target = cue === null || condition === 'none' ? null : targets[condition === 'bit0' ? 0 : 1];
  const dnCounts = [0, 0];
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
      const cueActive = cueIds.every(i => net.drive[i] === 150);
      if (cueIds.length) net.setDrive(cueIds, 0);
      agent.setDnWindowCounts([...dnCounts]);
      const action = agent.decide(net);
      row = { cue, condition, key, dnCounts: [...dnCounts], rates: [...agent.rates],
        features: [...agent.f], scores: [...agent.scores], action, cueActive,
        cuePrediction: cue === null ? null : cuePrediction(agent.rates) };
      net.resetCounts();
    }
    if (t === 620) agent.afterDopamine(net, false);
    net.step();
    if (t >= 240 && t < 280) countDnSpikes(net, dnCounts);
  }
  return row;
}

const smokeRows = [], smokeBySeed = [];
for (const [seed] of pairs) {
  const net = netFor(seed), agent = agentFor(net, seed, true), initial = controller(agent);
  const first = smokeProbe(net, agent, null, 'none', smokeBase - 10 + 100000 * (seed - smokeStudentBase));
  smokeRows.push({ seed, ...first, firstRound: true });
  for (const cue of [0, 1, 2]) for (let repeat = 0; repeat < 12; repeat++) {
    const key = smokeBase + 100000 * (seed - smokeStudentBase) + 1000 * cue + 10 * repeat;
    for (const condition of ['none', 'bit0', 'bit1'])
      smokeRows.push({ seed, repeat, ...smokeProbe(net, agent, cue, condition, key) });
  }
  const rows = smokeRows.filter(r => r.seed === seed && !r.firstRound);
  const byCue = [0, 1, 2].map(cue => {
    const ys = rows.filter(r => r.cue === cue);
    const byCondition = Object.fromEntries(['none', 'bit0', 'bit1'].map(condition => {
      const xs = ys.filter(r => r.condition === condition);
      return [condition, { correct: xs.filter(r => r.cuePrediction === cue).length,
        meanDnCounts: [0, 1].map(j => mean(xs.map(r => r.dnCounts[j]))) }];
    }));
    const extra = [0, 1].map(bit => byCondition[`bit${bit}`].meanDnCounts[bit] -
      byCondition.none.meanDnCounts[bit]);
    return { cue, byCondition, extra };
  });
  const featureValid = rows.every(r => r.cueActive && r.features.length === 5 &&
    r.features.every(Number.isFinite) && r.dnCounts.every((n, j) =>
      r.features[j + 3] === n / dnScale) && r.rates.every((rate, j) =>
      approx(r.features[j], (Math.log1p(rate) - featureMean[j]) / featureStd[j])));
  const firstRoundValid = first.cue === null && first.condition === 'none' &&
    first.dnCounts.length === 2 && first.dnCounts.every(n => n === 0) &&
    first.features.length === 5;
  smokeBySeed.push({ seed, byCue, checks: {
    fullCount: rows.length === 108 && byCue.every(x => ['none', 'bit0', 'bit1'].every(c =>
      rows.filter(r => r.cue === x.cue && r.condition === c).length === 12)),
    measuredFiveFeatures: featureValid, firstRoundValid,
    physicalReach: byCue.every(x => x.extra.every(n => n >= 1)),
    ...(pilot ? { dnBitSeparation: byCue.every(x => [0, 1].every(bit =>
      rows.filter(r => r.cue === x.cue && r.condition === `bit${bit}` &&
        r.dnCounts[bit] > r.dnCounts[1 - bit]).length >= 10)) } : {}),
    cueRecognition: byCue.every(x => ['bit0', 'bit1'].every(c =>
      x.byCondition[c].correct >= 10)),
    cueNoLargeDrop: byCue.every(x => ['bit0', 'bit1'].every(c =>
      x.byCondition.none.correct - x.byCondition[c].correct <= 2)),
    weightsUnchanged: net.m.every(x => x === 1) && same(controller(agent), initial) }
  });
}
const smokePassed = smokeBySeed.every(x => Object.values(x.checks).every(Boolean));
const design = { pairs, arms, trainSeeds, finalSeeds, checkpointSeed, checkpoints, rounds,
  threshold, targets, dnScale, featureMean, featureStd,
  opponent: 'previous_opponent offset 1 noise 0', trainingBudget: '12 x 24 per arm and student',
  teacherStreams: 'rng(teacherSeed*10000000+1000*episodeKey+round+1)',
  studentStreams: 'key=studentSeed*10000000+1000*episodeKey+round; network rng(key+2); action rng(key+501); cue-period network rng(key+3)',
  smokeStreams: `${smokeBase}+100000*(studentSeed-${smokeStudentBase})+1000*cue+10*repeat`,
  shuffleStreams: '2026140400+1000*teacherSeed+episodeIndex; Fisher-Yates within observed rounds 1-23',
  checkpoint: 'after episodes 2,4,6; frozen disconnected 24 rounds, opponent seed 52000, episodeKey 200; mean wins/24',
  finalEvaluation: 'disconnected and common correct held-out messages, each 3 x 24 rounds seeds 51000-51002, episodeKeys 100-102',
  gates: 'correct arm every student: early checkpoint mean > each control; final disconnected >=58/72, >=70% each state, >=12/69 each action, and >=4/72 wins versus each control; all validity checks' };
if (!smokePassed) {
  const stopped = { command, design,
    status: 'stopped_before_training', smokeBySeed, smokeRows, smokePassed,
    conclusion: 'Implementation or receiving-validity gate failed; no student training was run.' };
  writeFileSync(output, `${JSON.stringify(stopped, null, 2)}\n`);
  console.log(JSON.stringify({ status: stopped.status, smokeBySeed, smokePassed }, null, 2));
  process.exit(0);
}

const students = [];
for (const [seed, teacherSeed] of pairs) {
  const teacherNet = netFor(teacherSeed);
  const teacherAgent = agentFor(teacherNet, teacherSeed, false);
  const untrainedAgent = agentFor(teacherNet, teacherSeed, false);
  const savedTeacher = archived(teacherSeed).finalController;
  teacherAgent.W.forEach((row, a) => row.set(savedTeacher.W[a]));
  teacherAgent.b.set(savedTeacher.b); teacherAgent.baseline = savedTeacher.baseline;
  teacherAgent.readoutLearn = false; untrainedAgent.readoutLearn = false;
  if (!untrainedAgent.W.every(w => w.every(x => x === 0)) ||
      !untrainedAgent.b.every(x => x === 0)) throw Error('Untrained teacher not zero initialized');
  const teacherInitial = controller(teacherAgent);
  const teacherTraining = trainSeeds.map((sequenceSeed, episodeIndex) =>
    teacherEpisode(teacherNet, teacherAgent, untrainedAgent, teacherSeed, sequenceSeed, episodeIndex));
  const teacherFinal = finalSeeds.map((sequenceSeed, i) =>
    teacherEpisode(teacherNet, teacherAgent, untrainedAgent, teacherSeed, sequenceSeed, 100 + i));
  const teacherFrozen = same(controller(teacherAgent), teacherInitial) &&
    teacherNet.m.every(x => x === 1);
  if (!teacherFrozen) throw Error('Teacher parameters changed');

  const net = netFor(seed), runs = [];
  for (const arm of arms) {
    const agent = agentFor(net, seed, true), initialController = controller(agent);
    const curve = [], trainingRows = [], checkpointRows = [];
    for (let episodeIndex = 0; episodeIndex < 12; episodeIndex++) {
      const sourceRows = teacherTraining[episodeIndex];
      const correctBits = sourceRows.map(r => r.bit);
      const bits = arm === 'correct' ? correctBits :
        arm === 'shuffled' ? shuffled(correctBits, teacherSeed, episodeIndex) :
        arm === 'untrained' ? sourceRows.map(r => r.untrainedBit) :
        Array(rounds).fill(null);
      if (bits[0] !== null || arm === 'shuffled' &&
          !same(counts(bits.slice(1)), counts(correctBits.slice(1))))
        throw Error('Message schedule or shuffle changed');
      const rows = studentEpisode(net, agent, seed, trainSeeds[episodeIndex], episodeIndex, bits, true);
      if (rows.some((r, i) => r.cue !== sourceRows[i].cue ||
          r.opponent !== sourceRows[i].opponent)) throw Error('Teacher/student opponent schedule mismatch');
      curve.push(rows.filter(r => r.reward === 1).length);
      trainingRows.push(...rows.map(r => ({ episode: episodeIndex, ...r })));
      if (checkpoints.includes(episodeIndex + 1)) {
        const before = controller(agent);
        const frozen = studentEpisode(net, agent, seed, checkpointSeed, 200,
          Array(rounds).fill(null), false);
        if (!same(before, controller(agent)) || net.m.some(x => x !== 1) ||
            frozen.some(r => r.action !== r.scores.indexOf(Math.max(...r.scores))))
          throw Error('Checkpoint evaluation changed weights or action readout');
        checkpointRows.push({ afterEpisodes: episodeIndex + 1,
          ...summarize(frozen), rows: frozen });
      }
    }
    const finalController = controller(agent);
    const evaluations = {};
    for (const mode of ['disconnected', 'correct']) {
      const before = controller(agent), all = [];
      for (let i = 0; i < finalSeeds.length; i++) {
        const bits = mode === 'disconnected' ? Array(rounds).fill(null) :
          teacherFinal[i].map(r => r.bit);
        const rows = studentEpisode(net, agent, seed, finalSeeds[i], 100 + i, bits, false);
        if (rows.some((r, round) => r.cue !== teacherFinal[i][round].cue ||
            r.opponent !== teacherFinal[i][round].opponent))
          throw Error('Held-out teacher/student schedule mismatch');
        all.push(...rows.map(r => ({ sequenceSeed: finalSeeds[i], ...r })));
      }
      evaluations[mode] = { ...summarize(all), rows: all,
        controllerUnchanged: same(before, controller(agent)),
        neuralWeightsUnchanged: net.m.every(x => x === 1),
        frozenArgmax: all.every(r => r.action === r.scores.indexOf(Math.max(...r.scores))) };
    }
    runs.push({ arm, initialController, finalController,
      controllerChangeNorm: norm(initialController, finalController),
      neuralWeightsUnchanged: net.m.every(x => x === 1),
      curve, trainingRows, checkpointRows, earlyMeanWinsPer24: mean(checkpointRows.map(x => x.wins)),
      evaluations });
  }
  const correct = runs.find(r => r.arm === 'correct');
  const controls = runs.filter(r => r.arm !== 'correct');
  const final = correct.evaluations.disconnected;
  const validity = {
    teacherFrozen,
    matchedInitialController: runs.every(r => same(r.initialController, correct.initialController)),
    initialFiveFeatureZero: correct.initialController.W.every(w => w.length === 5 &&
      w.every(x => x === 0)) && correct.initialController.b.every(x => x === 0),
    matchedTrainingSchedule: runs.every(r => same(r.trainingRows.map(x =>
      [x.episode, x.round, x.cue, x.opponent]),
      correct.trainingRows.map(x => [x.episode, x.round, x.cue, x.opponent]))),
    fullTrainingCounts: runs.every(r => r.trainingRows.length === 288 &&
      r.curve.length === 12 && r.checkpointRows.length === 3 &&
      r.checkpointRows.every(x => x.rows.length === 24)),
    matchedEvaluationSchedule: runs.every(r => ['disconnected', 'correct'].every(mode =>
      same(r.evaluations[mode].rows.map(x => [x.sequenceSeed, x.cue, x.opponent]),
        correct.evaluations.disconnected.rows.map(x => [x.sequenceSeed, x.cue, x.opponent])))),
    fullFinalCounts: runs.every(r => ['disconnected', 'correct'].every(mode =>
      r.evaluations[mode].rows.length === 72 &&
      same(r.evaluations[mode].byCue.map(x => x.trials), [23, 24, 22]))),
    frozenEvaluations: runs.every(r => ['disconnected', 'correct'].every(mode =>
      r.evaluations[mode].controllerUnchanged &&
      r.evaluations[mode].neuralWeightsUnchanged && r.evaluations[mode].frozenArgmax)),
    frozenSynapses: runs.every(r => r.neuralWeightsUnchanged),
    controllersLearned: runs.every(r => r.controllerChangeNorm > 0),
    ownRewardOnly: runs.every(r => r.trainingRows.every(x => x.reward === result(x.action, x.opponent))),
    messageSchedule: runs.every(r => r.trainingRows.every(x => {
      const source = teacherTraining[x.episode];
      const expected = r.arm === 'correct' ? source[x.round].bit :
        r.arm === 'shuffled' ? shuffled(source.map(y => y.bit), teacherSeed, x.episode)[x.round] :
        r.arm === 'untrained' ? source[x.round].untrainedBit : null;
      return x.bit === expected;
    }) && r.evaluations.correct.rows.every(x => x.bit ===
      teacherFinal[x.sequenceSeed - finalSeeds[0]][x.round].bit) &&
      r.evaluations.disconnected.rows.every(x => x.bit === null))
  };
  const reliability = final.wins >= 58 && final.byCue.every(x => x.wins / x.trials >= 0.7) &&
    final.actionCounts.every(x => x >= 12);
  const earlyDifferences = Object.fromEntries(controls.map(r =>
    [r.arm, correct.earlyMeanWinsPer24 - r.earlyMeanWinsPer24]));
  const retentionDifferences = Object.fromEntries(controls.map(r =>
    [r.arm, final.wins - r.evaluations.disconnected.wins]));
  const gates = { reliability,
    learningSpeed: Object.values(earlyDifferences).every(x => x > 0),
    retention: Object.values(retentionDifferences).every(x => x >= 4) };
  students.push({ seed, teacherSeed, teacherTraining, teacherFinal, runs,
    validity, gates, earlyDifferences, retentionDifferences,
    passed: Object.values(validity).every(Boolean) && Object.values(gates).every(Boolean) });
}

const allValid = smokePassed && students.every(s => Object.values(s.validity).every(Boolean));
const gates = { reliability: students.every(s => s.gates.reliability),
  learningSpeed: students.every(s => s.gates.learningSpeed),
  retention: students.every(s => s.gates.retention) };
const resultData = { command, design,
  status: 'complete', smokeBySeed, smokeRows, smokePassed,
  students, allValid, gates, passed: allValid && Object.values(gates).every(Boolean) };
writeFileSync(output, `${JSON.stringify(resultData, null, 2)}\n`);
console.log(JSON.stringify({ status: resultData.status, smokePassed, allValid, gates,
  passed: resultData.passed,
  students: students.map(s => ({ seed: s.seed, teacherSeed: s.teacherSeed,
    validity: s.validity, gates: s.gates, earlyDifferences: s.earlyDifferences,
    retentionDifferences: s.retentionDifferences,
    runs: s.runs.map(r => ({ arm: r.arm, curve: r.curve,
      checkpoints: r.checkpointRows.map(x => x.wins), earlyMean: r.earlyMeanWinsPer24,
      disconnected: { wins: r.evaluations.disconnected.wins,
        byCue: r.evaluations.disconnected.byCue.map(x => `${x.wins}/${x.trials}`),
        actionCounts: r.evaluations.disconnected.actionCounts },
      correctMessages: r.evaluations.correct.wins, controllerChangeNorm: r.controllerChangeNorm })) }))
}, null, 2));

