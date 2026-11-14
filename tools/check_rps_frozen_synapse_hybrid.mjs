// One locked comparison of simultaneous versus frozen KC->MBON learning.
// node tools/check_rps_frozen_synapse_hybrid.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { Opponent, RpsEpisode, RpsRewardReadout } from '../js/rps.js';

const output = new URL('../data/rps_frozen_synapse_hybrid.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite locked frozen-synapse result');
const sub = JSON.parse(readFileSync(new URL('../data/subgraph.json', import.meta.url)));
const prior = JSON.parse(readFileSync(new URL('../data/rps_reward_readout.json', import.meta.url)));
const information = JSON.parse(readFileSync(new URL('../data/rps_readout_information.json', import.meta.url)));
if (!prior.allValid || !information.passed) throw Error('Required saved baseline invalid');
const { featureMean, featureStd } = prior.design;
const centroids = information.centroids;
if (centroids.length !== 3 || centroids.some(c => c.length !== 3)) throw Error('Invalid fixed information centroids');
const seeds = [671, 672, 673], episodes = 12, rounds = 24;
const trainSeeds = Array.from({ length: episodes }, (_, i) => 41000 + i);
const heldoutSeeds = [51000, 51001, 51002];
const arms = [
  { name: 'simultaneous', freezeSynapses: false, readoutLearn: true },
  { name: 'frozen_synapse', freezeSynapses: true, readoutLearn: true },
  { name: 'frozen_synapse_no_readout', freezeSynapses: true, readoutLearn: false }
];
const opponent = seed => new Opponent({ strategy: 'previous_opponent', offset: 1, noise: 0, seed });
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const counts = xs => [0, 1, 2].map(a => xs.filter(x => x === a).length);
const mean = xs => xs.reduce((s, x) => s + x, 0) / xs.length;
const readout = agent => ({ W: agent.W.map(w => [...w]), b: [...agent.b], baseline: agent.baseline });
const emptyReadout = { W: [[0, 0, 0], [0, 0, 0], [0, 0, 0]], b: [0, 0, 0], baseline: 0 };
function fly(seed, arm) {
  const net = new Network(sub, wiring(sub, 'real', seed), { seed });
  const agent = new RpsRewardReadout(net, { seed, featureMean, featureStd,
    freezeSynapses: arm.freezeSynapses, readoutLearn: arm.readoutLearn });
  return { net, agent };
}
function runEpisode(ep, studentSeed, episodeKey, onDopamine = null) {
  while (!ep.done) {
    const phase = ep.t % 400;
    if (phase === 0) {
      const round = Math.floor(ep.t / 400);
      const key = studentSeed * 10000000 + episodeKey * 1000 + round;
      ep.net.rand = rng(key + 1);
      ep.agent.rand = rng(key + 501);
    }
    ep.step();
    if (phase === 340 && onDopamine) onDopamine(ep.agent.D);
  }
  return ep;
}
function classify(rawRates) {
  const x = rawRates.map(r => Math.log1p(r));
  const distances = centroids.map(c => c.reduce((s, v, j) => s + (x[j] - v) ** 2, 0));
  return distances.indexOf(Math.min(...distances));
}
function scoredEvent(event, params, sequenceSeed) {
  const rates = [...event.rates];
  const features = rates.map((r, j) => (Math.log1p(r) - featureMean[j]) / featureStd[j]);
  const scores = params.W.map((w, a) => w.reduce((s, v, j) => s + v * features[j], params.b[a]));
  if (event.action !== scores.indexOf(Math.max(...scores))) throw Error('Frozen action/score mismatch');
  return { sequenceSeed, round: event.round, previous: event.observation.previousOpponent,
    opponent: event.opponent, action: event.action, reward: event.reward,
    rates, features, scores, paperMinusScissors: scores[1] - scores[2],
    fixedCentroidState: event.observation.previousOpponent === null ? null : classify(rates) };
}
function evaluate(seed, arm, weights, params) {
  const { net, agent } = fly(seed, arm);
  net.m.set(weights);
  agent.W.forEach((w, a) => w.set(params.W[a]));
  agent.b.set(params.b); agent.baseline = params.baseline;
  agent.explore = false;
  const frozenWeights = net.m.slice(), frozenReadout = readout(agent), events = [];
  heldoutSeeds.forEach((sequenceSeed, i) => {
    const ep = new RpsEpisode(net, agent, opponent(sequenceSeed), { rounds, learn: false });
    runEpisode(ep, seed, 100 + i);
    events.push(...ep.events.map(e => scoredEvent(e, frozenReadout, sequenceSeed)));
  });
  const observed = events.filter(e => e.previous !== null);
  const byPrevious = [0, 1, 2].map(previous => {
    const xs = observed.filter(e => e.previous === previous);
    return { previous, trials: xs.length, wins: xs.filter(e => e.reward === 1).length,
      actionCounts: counts(xs.map(e => e.action)),
      meanPaperMinusScissors: mean(xs.map(e => e.paperMinusScissors)),
      meanFeatures: [0, 1, 2].map(j => mean(xs.map(e => e.features[j]))),
      fixedCentroidCorrect: xs.filter(e => e.fixedCentroidState === previous).length };
  });
  const fixedCentroidConfusion = [0, 1, 2].map(previous => [0, 1, 2].map(predicted =>
    observed.filter(e => e.previous === previous && e.fixedCentroidState === predicted).length));
  return { wins: events.filter(e => e.reward === 1).length,
    observedActionCounts: counts(observed.map(e => e.action)), byPrevious,
    fixedCentroidConfusion,
    fixedCentroidAccuracy: observed.filter(e => e.fixedCentroidState === e.previous).length,
    weightsUnchanged: net.m.every((x, i) => x === frozenWeights[i]),
    readoutUnchanged: same(readout(agent), frozenReadout), events };
}
function train(seed, arm) {
  const { net, agent } = fly(seed, arm), initial = net.m.slice();
  const initialReadoutMatched = same(readout(agent), emptyReadout);
  const curve = [], events = [];
  let dopamineNonzeroSteps = 0;
  trainSeeds.forEach((sequenceSeed, episodeIndex) => {
    const ep = new RpsEpisode(net, agent, opponent(sequenceSeed),
      { rounds, learn: true, balancedActions: false });
    runEpisode(ep, seed, episodeIndex, D => { dopamineNonzeroSteps += Number(D !== 0); });
    curve.push(ep.events.filter(e => e.reward === 1).length);
    events.push(...ep.events.map(e => ({ episode: episodeIndex, round: e.round,
      previous: e.observation.previousOpponent, opponent: e.opponent,
      action: e.action, nativeAction: e.nativeAction, reward: e.reward })));
  });
  const changedWeights = [];
  for (let i = 0; i < net.m.length; i++)
    if (net.m[i] !== initial[i]) changedWeights.push([i, net.m[i]]);
  const finalReadout = readout(agent);
  const readoutWeightNorm = Math.sqrt(finalReadout.W.flat().reduce((s, x) => s + x * x, 0) +
    finalReadout.b.reduce((s, x) => s + x * x, 0));
  return { arm: arm.name, curve, dopamineNonzeroSteps, changedWeights,
    finalReadout, readoutWeightNorm,
    initialWeightsMatched: initial.every(x => x === 1), initialReadoutMatched,
    events, after: evaluate(seed, arm, net.m, finalReadout) };
}
const students = seeds.map(seed => {
  const runs = arms.map(arm => train(seed, arm));
  const [A, B, C] = runs;
  const schedule = A.events.map(e => [e.previous, e.opponent]);
  const matchedSchedule = runs.every(r => same(r.events.map(e => [e.previous, e.opponent]), schedule));
  const noForcedActions = runs.every(r => r.events.every(e => e.action === e.nativeAction));
  const valid = matchedSchedule && noForcedActions && runs.every(r =>
    r.initialWeightsMatched && r.initialReadoutMatched && r.dopamineNonzeroSteps > 0 &&
    r.events.length === 288 && r.after.events.length === 72 &&
    r.after.byPrevious.reduce((s, x) => s + x.trials, 0) === 69 &&
    r.after.weightsUnchanged && r.after.readoutUnchanged) &&
    A.changedWeights.length > 0 && A.readoutWeightNorm > 0 &&
    B.changedWeights.length === 0 && B.readoutWeightNorm > 0 &&
    C.changedWeights.length === 0 && same(C.finalReadout, emptyReadout);
  const reliable = r => r.after.wins >= 58 &&
    r.after.byPrevious.every(x => x.wins / x.trials >= 0.7) &&
    r.after.observedActionCounts.every(n => n >= 12);
  const gates = { B_reliable: reliable(B), B_minus_A_at_least_10: B.after.wins - A.after.wins >= 10,
    C_fails_and_unchanged: !reliable(C) && C.changedWeights.length === 0 &&
      same(C.finalReadout, emptyReadout) };
  return { seed, runs, validity: { matchedSchedule, noForcedActions, valid }, gates,
    supportsInterference: valid && Object.values(gates).every(Boolean) };
});
const result = { command: 'node tools/check_rps_frozen_synapse_hybrid.mjs',
  design: { seeds, episodes, rounds, trainSeeds, heldoutSeeds,
    opponent: 'previous_opponent offset 1 noise 0',
    fixedFeatureMean: featureMean, fixedFeatureStd: featureStd,
    fixedCentroidsFrom: 'data/rps_readout_information.json',
    training: 'softmax sampled; no forced actions; dopamine activity in all arms',
    evaluation: 'deterministic argmax; no weight or baseline learning',
    trialStreams: 'network rng(seed*10000000 + episodeKey*1000 + round + 1), action rng(key + 501)',
    arms },
  gates: { B_wins: '>=58/72 in each seed', B_eachState: '>=70% in each state',
    B_eachAction: '>=12/69 for every action', B_over_A: '>=10/72 in each paired seed',
    C: 'fails learning criteria and has unchanged neural/readout weights' },
  students, allValid: students.every(s => s.validity.valid),
  supportsInterference: students.every(s => s.supportsInterference) };
writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({ allValid: result.allValid,
  supportsInterference: result.supportsInterference,
  students: students.map(s => ({ seed: s.seed, validity: s.validity, gates: s.gates,
    runs: s.runs.map(r => ({ arm: r.arm, wins: r.after.wins,
      byPrevious: r.after.byPrevious.map(x => x.wins),
      observedActionCounts: r.after.observedActionCounts,
      paperMinusScissors: [r.after.byPrevious[0].meanPaperMinusScissors,
        r.after.byPrevious[2].meanPaperMinusScissors],
      fixedCentroidAccuracy: r.after.fixedCentroidAccuracy,
      curve: r.curve, kcMbonChanges: r.changedWeights.length,
      readoutWeightNorm: r.readoutWeightNorm,
      dopamineNonzeroSteps: r.dopamineNonzeroSteps })) })) }, null, 2));
