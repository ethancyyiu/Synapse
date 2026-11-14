// One locked, teacher-only blocked RPS reversal/source feasibility experiment.
// node tools/check_rps_blocked_teacher_source.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { Opponent, RpsContextualBandit, RpsEpisode, result } from '../js/rps.js';

const output = new URL('../data/rps_blocked_teacher_source.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite blocked RPS teacher-source result');
const read = path => JSON.parse(readFileSync(new URL(path, import.meta.url)));
const sub = read('../data/subgraph.json');
const archived = read('../data/rps_contextual_bandit.json');
if (!archived.allValid) throw Error('Invalid archived contextual-bandit calibration');
const { featureMean, featureStd } = archived.design;
const seeds = [761, 762, 763], calibrationSeed = 761, heldoutSeeds = [762, 763];
const threshold = 1.098235868125541, trainOpponentSeed = 62000, evalOpponentSeed = 63000;
const episodes = 12, rounds = 24, switchAfter = 144, shuffleSeed = 2026110401;
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const approx = (a, b) => Math.abs(a - b) < 1e-10;
const snapshot = a => ({ W: a.W.map(w => [...w]), b: [...a.b], baseline: a.baseline });
const netFor = seed => new Network(sub, wiring(sub, 'real', seed), { seed });
const agentFor = (net, seed) => new RpsContextualBandit(net, { seed, featureMean, featureStd });
const opponent = (mapping, seed) => new Opponent({
  strategy: mapping === 'switch' ? 'switch' : 'previous_opponent',
  offset: mapping === 'B' ? 2 : 1, secondOffset: 2, switchAfter, noise: 0, seed });
const bitFrom = scores => Number(Math.max(...scores) - Math.min(...scores) > threshold);
const summarize = rows => ({ wins: rows.filter(r => r.reward === 1).length,
  byCue: [0, 1, 2].map(cue => {
    const xs = rows.filter(r => r.cue === cue);
    return { cue, trials: xs.length, wins: xs.filter(r => r.reward === 1).length,
      bits: [0, 1].map(bit => xs.filter(r => r.bit === bit).length) };
  }) });

function runEpisode(net, agent, game, seed, episodeKey, mapping, learn) {
  agent.explore = learn;
  const ep = new RpsEpisode(net, agent, game, { rounds, learn });
  const rows = [];
  let updatesValid = true;
  while (!ep.done) {
    const phase = ep.t % 400, round = Math.floor(ep.t / 400);
    if (phase === 0) {
      const key = seed * 10000000 + episodeKey * 1000 + round;
      net.rand = rng(key + 1); agent.rand = rng(key + 501);
    }
    const before = phase === 340 ? snapshot(agent) : null;
    ep.step();
    if (phase === 240) {
      const e = ep.events.at(-1), scores = [...agent.scores];
      rows.push({ round, cue: e.observation.previousOpponent, opponent: e.opponent,
        action: e.action, reward: e.reward, rates: [...agent.rates],
        features: [...agent.f], scores, spread: Math.max(...scores) - Math.min(...scores),
        bit: e.observation.previousOpponent === null ? null : bitFrom(scores), mapping });
    }
    if (phase === 340) {
      const after = snapshot(agent), r = rows.at(-1), a = r.action;
      const step = learn ? agent.alpha * (r.reward - r.scores[a]) : 0;
      updatesValid &&= approx(after.b[a], before.b[a] + step) &&
        r.features.every((v, j) => approx(after.W[a][j], before.W[a][j] + step * v)) &&
        [0, 1, 2].every(k => k === a ||
          (same(after.W[k], before.W[k]) && after.b[k] === before.b[k])) &&
        after.baseline === before.baseline;
    }
  }
  if (rows.length !== rounds || !updatesValid || net.m.some(v => v !== 1) ||
      rows.some(r => r.reward !== result(r.action, r.opponent) ||
        r.features.length !== 3 || r.bit !== (r.cue === null ? null : bitFrom(r.scores))))
    throw Error('Teacher episode validity failed');
  return rows;
}

function frozen(seed, params, mapping, episodeKey) {
  const net = netFor(seed), agent = agentFor(net, seed);
  agent.W.forEach((w, a) => w.set(params.W[a]));
  agent.b.set(params.b); agent.baseline = params.baseline;
  const before = snapshot(agent);
  const rows = runEpisode(net, agent, opponent(mapping, evalOpponentSeed),
    seed, episodeKey, mapping, false);
  if (!same(snapshot(agent), before) || net.m.some(v => v !== 1) ||
      rows.some(r => r.action !== r.scores.indexOf(Math.max(...r.scores))))
    throw Error('Frozen teacher evaluation changed weights or readout');
  return { ...summarize(rows), rows, weightsUnchanged: true };
}

const teachers = seeds.map(seed => {
  const net = netFor(seed), agent = agentFor(net, seed), initial = snapshot(agent);
  const game = opponent('switch', trainOpponentSeed), trainingRows = [], curve = [];
  let after6A, beforeB;
  for (let episode = 0; episode < episodes; episode++) {
    if (episode === 6) {
      const params = snapshot(agent);
      after6A = frozen(seed, params, 'A', 100);
      beforeB = frozen(seed, params, 'B', 101);
    }
    const mapping = episode < 6 ? 'A' : 'B';
    const rows = runEpisode(net, agent, game, seed, episode, mapping, true);
    trainingRows.push(...rows.map(r => ({ episode, ...r })));
    curve.push({ episode, mapping, ...summarize(rows) });
  }
  const final = snapshot(agent), after12B = frozen(seed, final, 'B', 101);
  const A = trainingRows.filter(r => r.mapping === 'A'), B = trainingRows.filter(r => r.mapping === 'B');
  const coverage = [A, B].map(rows => [0, 1, 2].map(cue => rows.filter(r => r.cue === cue).length));
  const validity = { fixedCoverage: same(coverage, [[47, 48, 48], [48, 48, 48]]),
    fullBudget: trainingRows.length === 288 && curve.length === 12 &&
      after6A.rows.length === 24 && beforeB.rows.length === 24 && after12B.rows.length === 24,
    firstCueOnlyNull: trainingRows.filter(r => r.cue === null).length === 1 &&
      trainingRows[0].cue === null && B[0].cue !== null,
    sameOpponentRule: trainingRows.every(r => r.opponent ===
      (r.cue === null ? game.initial : (r.cue + (r.mapping === 'A' ? 1 : 2)) % 3)),
    initialZero: initial.W.flat().every(v => v === 0) && initial.b.every(v => v === 0),
    controllerLearned: !same(initial, final), frozenSynapses: net.m.every(v => v === 1),
    frozenEvaluation: [after6A, beforeB, after12B].every(e => e.weightsUnchanged) };
  const adaptation = { AAfter6: after6A.wins >= 18,
    BAfter12: after12B.wins >= 18,
    BRecovery: after12B.wins - beforeB.wins >= 8 };
  return { seed, initial, final, curve, trainingRows, coverage,
    after6A, beforeB, after12B, validity, adaptation,
    passed: Object.values(validity).every(Boolean) && Object.values(adaptation).every(Boolean) };
});

const design = { seeds, calibrationSeed, heldoutSeeds, episodes, rounds,
  mappings: { A: 'next=(previous+1)%3', B: 'next=(previous+2)%3' },
  switchAfter, trainOpponentSeed, evalOpponentSeed, threshold, featureMean, featureStd,
  teacher: 'existing three-MBON contextual bandit, alpha .03, epsilon .20, frozen KC->MBON',
  trainingStreams: 'network rng(seed*10000000+episode*1000+round+1); action rng(key+501)',
  frozenEvaluations: 'after episode 6: A key100 and B key101; after episode 12: B key101; 24 rounds each',
  adaptationGate: 'every seed A-after6>=18/24, B-after12>=18/24, B-after12 minus B-before>=8/24',
  sourceDecoder: 'per-cue bit-to-mapping lookup fitted on teacher761 A-after6/B-after12 only; ties A',
  sourceGate: 'each cue pooled heldout balanced accuracy>=.75 and >=.10 above cue-stratified shuffle; each heldout teacher overall balanced>=.70 and emits bit1',
  shuffleSeed };
const adaptationPassed = teachers.every(t => t.passed);
if (!adaptationPassed) {
  const stopped = { command: 'node tools/check_rps_blocked_teacher_source.mjs', design,
    status: 'stopped_after_teacher_adaptation', teachers, adaptationPassed,
    sourceStatus: 'not_evaluated', conclusion: 'Teacher adaptation gate failed; no mapping-bit decoder was fitted or scored.' };
  writeFileSync(output, `${JSON.stringify(stopped, null, 2)}\n`);
  console.log(JSON.stringify({ status: stopped.status, teachers: teachers.map(t => ({
    seed: t.seed, A: t.after6A.wins, beforeB: t.beforeB.wins, afterB: t.after12B.wins,
    validity: t.validity, adaptation: t.adaptation })), adaptationPassed }, null, 2));
  process.exit(0);
}

const rowsFor = t => [
  ...t.after6A.rows.filter(r => r.cue !== null).map(r => ({ seed: t.seed, mapping: 'A', ...r })),
  ...t.after12B.rows.filter(r => r.cue !== null).map(r => ({ seed: t.seed, mapping: 'B', ...r })) ];
const calibration = rowsFor(teachers[0]);
const lookup = [0, 1, 2].map(cue => [0, 1].map(bit => {
  const count = mapping => calibration.filter(r => r.cue === cue && r.mapping === mapping &&
    r.bit === bit).length / calibration.filter(r => r.cue === cue && r.mapping === mapping).length;
  return count('B') > count('A') ? 'B' : 'A';
}));
const heldout = teachers.slice(1).flatMap(rowsFor);
const balanced = rows => {
  const byMapping = ['A', 'B'].map(mapping => {
    const xs = rows.filter(r => r.mapping === mapping);
    return { mapping, n: xs.length, correct: xs.filter(r => lookup[r.cue][r.bit] === mapping).length };
  });
  return { byMapping, accuracy: byMapping.reduce((s, x) => s + x.correct / x.n, 0) / 2 };
};
const random = rng(shuffleSeed), shuffled = heldout.map(r => ({ ...r }));
for (const cue of [0, 1, 2]) {
  const indices = shuffled.map((r, i) => r.cue === cue ? i : -1).filter(i => i >= 0);
  const bits = indices.map(i => shuffled[i].bit);
  for (let i = bits.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [bits[i], bits[j]] = [bits[j], bits[i]];
  }
  indices.forEach((index, j) => { shuffled[index].bit = bits[j]; });
}
const byCue = [0, 1, 2].map(cue => {
  const rows = heldout.filter(r => r.cue === cue), shuffledRows = shuffled.filter(r => r.cue === cue);
  const actual = balanced(rows), control = balanced(shuffledRows);
  return { cue, actual, shuffled: control, advantage: actual.accuracy - control.accuracy,
    bitCounts: ['A', 'B'].map(mapping => ({ mapping,
      counts: [0, 1].map(bit => rows.filter(r => r.mapping === mapping && r.bit === bit).length) })) };
});
const byTeacher = heldoutSeeds.map(seed => ({ seed,
  ...balanced(heldout.filter(r => r.seed === seed)),
  bitOneCount: heldout.filter(r => r.seed === seed && r.bit === 1).length }));
const sourceGates = { perCue: byCue.every(x => x.actual.accuracy >= .75),
  aboveShuffle: byCue.every(x => x.advantage >= .10),
  eachTeacher: byTeacher.every(x => x.accuracy >= .70),
  trainedBitDiffersFromUntrained: byTeacher.every(x => x.bitOneCount > 0) };
const resultData = { command: 'node tools/check_rps_blocked_teacher_source.mjs', design,
  status: 'complete', teachers, adaptationPassed, calibration, lookup, heldout,
  shuffled: shuffled.map(r => ({ seed: r.seed, mapping: r.mapping, cue: r.cue, bit: r.bit })),
  byCue, byTeacher, sourceGates, sourcePassed: Object.values(sourceGates).every(Boolean),
  conclusion: Object.values(sourceGates).every(Boolean) ?
    'Fixed bit passed exploratory source-information gate; no student experiment was run.' :
    'Fixed bit failed exploratory source-information gate; stop without student receiving/training.' };
writeFileSync(output, `${JSON.stringify(resultData, null, 2)}\n`);
console.log(JSON.stringify({ status: resultData.status, adaptationPassed,
  teachers: teachers.map(t => ({ seed: t.seed, A: t.after6A.wins,
    beforeB: t.beforeB.wins, afterB: t.after12B.wins })), lookup,
  byCue, byTeacher, sourceGates, sourcePassed: resultData.sourcePassed }, null, 2));
