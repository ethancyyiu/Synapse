// Locked Stage-1 teacher-only test of two fixed RPS opponent mappings.
// node tools/check_rps_strategy_b_replication.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { Opponent, RpsContextualBandit, RpsEpisode, result } from '../js/rps.js';

const output = new URL('../data/rps_strategy_b_replication.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite independent-teacher result');
const read = path => JSON.parse(readFileSync(new URL(path, import.meta.url)));
const sub = read('../data/subgraph.json');
const prior = read('../data/rps_contextual_bandit.json');
if (!prior.allValid) throw Error('Invalid archived contextual-bandit feature calibration');
const { featureMean, featureStd } = prior.design;
const groups = [{ mapping: 'B', offset: 2, seeds: [791, 792, 793] }];
const trainSeeds = Array.from({ length: 12 }, (_, i) => 41000 + i);
const evalSeeds = [52000, 52003, 52004];
const rounds = 24;
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const approx = (a, b) => Math.abs(a - b) < 1e-10;
const counts = xs => [0, 1, 2].map(a => xs.filter(x => x === a).length);
const snapshot = a => ({ W: a.W.map(w => [...w]), b: [...a.b], baseline: a.baseline });
const norm = (a, b) => Math.hypot(...a.W.flat().map((v, i) => b.W.flat()[i] - v),
  ...a.b.map((v, i) => b.b[i] - v));
const gameFor = (offset, seed) => new Opponent({ strategy: 'previous_opponent', offset, noise: 0, seed });
const netFor = seed => new Network(sub, wiring(sub, 'real', seed), { seed });
const agentFor = (net, seed) => new RpsContextualBandit(net,
  { seed, featureMean, featureStd, freezeSynapses: true, readoutLearn: true });
const summarize = rows => {
  const observed = rows.filter(r => r.cue !== null);
  return { wins: rows.filter(r => r.reward === 1).length,
    byCue: [0, 1, 2].map(cue => {
      const x = observed.filter(r => r.cue === cue);
      return { cue, trials: x.length, wins: x.filter(r => r.reward === 1).length,
        actionCounts: counts(x.map(r => r.action)) };
    }), observedActionCounts: counts(observed.map(r => r.action)) };
};

function episode(net, agent, seed, sequenceSeed, episodeKey, offset, learn) {
  agent.explore = learn;
  const game = gameFor(offset, sequenceSeed);
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
      const e = ep.events.at(-1);
      rows.push({ round, cue: e.observation.previousOpponent,
        opponent: e.opponent, action: e.action, reward: e.reward,
        rates: [...agent.rates], features: [...agent.f], scores: [...agent.scores] });
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
        r.opponent !== (r.cue === null ? game.initial : (r.cue + offset) % 3) ||
        r.features.length !== 3 ||
        (!learn && r.action !== r.scores.indexOf(Math.max(...r.scores)))))
    throw Error(`Episode validity failed for teacher ${seed}, sequence ${sequenceSeed}`);
  return rows;
}

function teacher(seed, mapping, offset) {
  const net = netFor(seed), agent = agentFor(net, seed), initial = snapshot(agent);
  const trainingRows = [], curve = [];
  trainSeeds.forEach((sequenceSeed, episodeIndex) => {
    const rows = episode(net, agent, seed, sequenceSeed, episodeIndex, offset, true);
    trainingRows.push(...rows.map(r => ({ episode: episodeIndex, sequenceSeed, ...r })));
    curve.push({ episode: episodeIndex, sequenceSeed, ...summarize(rows) });
  });
  const final = snapshot(agent), controllerChangeNorm = norm(initial, final);
  const frozenNet = netFor(seed), frozenAgent = agentFor(frozenNet, seed);
  frozenAgent.W.forEach((w, a) => w.set(final.W[a]));
  frozenAgent.b.set(final.b); frozenAgent.baseline = final.baseline;
  const frozenBefore = snapshot(frozenAgent), frozenRows = [];
  evalSeeds.forEach((sequenceSeed, i) => {
    const rows = episode(frozenNet, frozenAgent, seed, sequenceSeed, 100 + i, offset, false);
    frozenRows.push(...rows.map(r => ({ sequenceSeed, ...r })));
  });
  const frozen = { ...summarize(frozenRows), rows: frozenRows,
    controllerUnchanged: same(snapshot(frozenAgent), frozenBefore),
    synapsesUnchanged: frozenNet.m.every(x => x === 1) };
  const validity = {
    initialZero: initial.W.flat().every(v => v === 0) && initial.b.every(v => v === 0),
    completeTraining: trainingRows.length === 288 && curve.length === 12 &&
      trainingRows.filter(r => r.cue === null).length === 12,
    cueCoverage: same(counts(trainingRows.filter(r => r.cue !== null).map(r => r.cue)),
      offset === 1 ? [94, 88, 94] : [88, 94, 94]),
    completeEvaluation: frozenRows.length === 72 && frozenRows.filter(r => r.cue === null).length === 3 &&
      frozen.byCue.every(x => x.trials === 23),
    controllerLearned: controllerChangeNorm > 0,
    frozenController: frozen.controllerUnchanged,
    frozenSynapses: net.m.every(v => v === 1) && frozen.synapsesUnchanged
  };
  const gates = { wins: frozen.wins >= 58,
    eachCue: frozen.byCue.every(x => x.wins / x.trials >= 0.7),
    eachAction: frozen.observedActionCounts.every(n => n >= 12) };
  return { seed, mapping, offset, initial, final, controllerChangeNorm,
    curve, trainingRows, frozen, validity, gates,
    passed: Object.values(validity).every(Boolean) && Object.values(gates).every(Boolean) };
}

const teachers = groups.flatMap(group => group.seeds.map(seed => teacher(seed, group.mapping, group.offset)));
const allValid = teachers.every(t => Object.values(t.validity).every(Boolean));
const passed = teachers.every(t => t.passed);
const design = { groups, trainingEpisodes: 12, roundsPerEpisode: rounds,
  trainSeeds, evalSeeds, evalInitialMoves: [0, 1, 2],
  trainCueCoverage: { A: [94, 88, 94], B: [88, 94, 94] },
  evalCueCoverage: [23, 23, 23], featureMean, featureStd,
  controller: 'original three-MBON contextual bandit, alpha .03, epsilon .20, zero initialization, selected-action reward-prediction update',
  neural: 'real wiring; frozen KC-to-MBON weights; original 400 ms RPS episode',
  streams: 'network rng(seed*10000000+episodeKey*1000+round+1); action rng(key+501)',
  gates: 'every teacher >=58/72 wins, >=70% each observed cue, >=12 of each action across 69 observed rounds; all validity checks' };
const outputData = { command: 'node tools/check_rps_strategy_b_replication.mjs',
  design, teachers, allValid, passed,
  sourceStatus: 'not_evaluated', studentStatus: 'not_run' };
writeFileSync(output, `${JSON.stringify(outputData, null, 2)}\n`);
console.log(JSON.stringify({ allValid, passed, teachers: teachers.map(t => ({
  seed: t.seed, mapping: t.mapping, trainingCurve: t.curve.map(x => x.wins),
  frozenWins: t.frozen.wins, byCue: t.frozen.byCue.map(x => `${x.wins}/${x.trials}`),
  actions: t.frozen.observedActionCounts, controllerChangeNorm: t.controllerChangeNorm,
  validity: t.validity, gates: t.gates, passed: t.passed })) }, null, 2));
