// Single locked comparison: frozen-synapse linear versus 8-tanh-unit reward-trained RPS readout.
// node tools/check_rps_nonlinear_readout.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { Opponent, RpsEpisode, RpsRewardReadout, RpsNonlinearReadout } from '../js/rps.js';

const output = new URL('../data/rps_nonlinear_readout.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite locked nonlinear result');
const sub = JSON.parse(readFileSync(new URL('../data/subgraph.json', import.meta.url)));
const prior = JSON.parse(readFileSync(new URL('../data/rps_reward_readout.json', import.meta.url)));
const information = JSON.parse(readFileSync(new URL('../data/rps_readout_information.json', import.meta.url)));
if (!prior.allValid || !information.passed || information.correct !== 98) throw Error('Required archived information is invalid');
const { featureMean, featureStd } = prior.design;
const seeds = [681, 682, 683], episodes = 12, rounds = 24;
const trainSeeds = Array.from({ length: episodes }, (_, i) => 41000 + i);
const heldoutSeeds = [51000, 51001, 51002];
const arms = [
  { name: 'linear_frozen', nonlinear: false, readoutLearn: true },
  { name: 'nonlinear_frozen', nonlinear: true, readoutLearn: true },
  { name: 'nonlinear_no_learning', nonlinear: true, readoutLearn: false },
];
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const counts = xs => [0, 1, 2].map(a => xs.filter(x => x === a).length);
const mean = xs => xs.reduce((s, x) => s + x, 0) / xs.length;
const opponent = seed => new Opponent({ strategy: 'previous_opponent', offset: 1, noise: 0, seed });
const controller = agent => ({ W: agent.W.map(w => [...w]), b: [...agent.b], baseline: agent.baseline,
  ...(agent.W1 ? { W1: agent.W1.map(w => [...w]), b1: [...agent.b1] } : {}) });
const parameters = p => [...p.W.flat(), ...p.b, ...(p.W1 ? [...p.W1.flat(), ...p.b1] : [])];
const changeNorm = (before, after) => Math.hypot(...parameters(after).map((x, i) => x - parameters(before)[i]));
function fly(seed, arm) {
  const net = new Network(sub, wiring(sub, 'real', seed), { seed });
  const Agent = arm.nonlinear ? RpsNonlinearReadout : RpsRewardReadout;
  const agent = new Agent(net, { seed, featureMean, featureStd,
    freezeSynapses: true, readoutLearn: arm.readoutLearn });
  return { net, agent };
}
function runEpisode(ep, studentSeed, episodeKey, onDecision = null, onDopamine = null) {
  while (!ep.done) {
    const phase = ep.t % 400;
    if (phase === 0) {
      const round = Math.floor(ep.t / 400);
      const key = studentSeed * 10000000 + episodeKey * 1000 + round;
      ep.net.rand = rng(key + 1);
      ep.agent.rand = rng(key + 501);
    }
    ep.step();
    if (phase === 240 && onDecision) onDecision(ep.agent, ep.events.at(-1));
    if (phase === 340 && onDopamine) onDopamine(ep.agent.D);
  }
  return ep;
}
function scoresFrom(params, f) {
  const h = params.W1?.map((w, k) => Math.tanh(w.reduce((s, x, j) => s + x * f[j], params.b1[k])));
  return params.W.map((w, a) => w.reduce((s, x, j) => s + x * (h ?? f)[j], params.b[a]));
}
function evaluate(seed, arm, weights, params) {
  const { net, agent } = fly(seed, arm);
  net.m.set(weights);
  agent.W.forEach((w, a) => w.set(params.W[a]));
  agent.b.set(params.b); agent.baseline = params.baseline;
  if (arm.nonlinear) {
    agent.W1.forEach((w, k) => w.set(params.W1[k]));
    agent.b1.set(params.b1);
  }
  agent.explore = false;
  const frozenWeights = net.m.slice(), frozenController = controller(agent), events = [];
  let scorePathMatched = true;
  heldoutSeeds.forEach((sequenceSeed, i) => {
    const decisions = [];
    const ep = new RpsEpisode(net, agent, opponent(sequenceSeed), { rounds, learn: false });
    runEpisode(ep, seed, 100 + i, (reader, event) => decisions.push({
      rates: [...reader.rates], features: [...reader.f], scores: [...reader.scores], action: event.action }));
    ep.events.forEach((event, round) => {
      const decision = decisions[round];
      const features = event.rates.map((r, j) => (Math.log1p(r) - featureMean[j]) / featureStd[j]);
      const scores = scoresFrom(frozenController, features);
      scorePathMatched &&= decision.action === event.action &&
        event.action === scores.indexOf(Math.max(...scores)) &&
        event.rates.every((x, j) => x === decision.rates[j]) &&
        features.every((x, j) => Math.abs(x - decision.features[j]) < 1e-12) &&
        scores.every((x, j) => Math.abs(x - decision.scores[j]) < 1e-12);
      events.push({ sequenceSeed, round, previous: event.observation.previousOpponent,
        opponent: event.opponent, action: event.action, reward: event.reward,
        rates: [...event.rates], features, scores, paperMinusScissors: scores[1] - scores[2] });
    });
  });
  const observed = events.filter(e => e.previous !== null);
  const byPrevious = [0, 1, 2].map(previous => {
    const xs = observed.filter(e => e.previous === previous);
    return { previous, trials: xs.length, wins: xs.filter(e => e.reward === 1).length,
      actionCounts: counts(xs.map(e => e.action)),
      meanPaperMinusScissors: mean(xs.map(e => e.paperMinusScissors)) };
  });
  return { wins: events.filter(e => e.reward === 1).length,
    observedActionCounts: counts(observed.map(e => e.action)), byPrevious,
    scorePathMatched, weightsUnchanged: net.m.every((x, i) => x === frozenWeights[i]),
    readoutUnchanged: same(controller(agent), frozenController), events };
}
function train(seed, arm) {
  const { net, agent } = fly(seed, arm), initialWeights = net.m.slice(), initialController = controller(agent);
  const curve = [], events = [];
  let dopamineNonzeroSteps = 0;
  trainSeeds.forEach((sequenceSeed, episodeIndex) => {
    const ep = new RpsEpisode(net, agent, opponent(sequenceSeed),
      { rounds, learn: true, balancedActions: false });
    runEpisode(ep, seed, episodeIndex, null, D => { dopamineNonzeroSteps += Number(D !== 0); });
    curve.push(ep.events.filter(e => e.reward === 1).length);
    events.push(...ep.events.map(e => ({ episode: episodeIndex, round: e.round,
      previous: e.observation.previousOpponent, opponent: e.opponent,
      action: e.action, nativeAction: e.nativeAction, reward: e.reward })));
  });
  const changedWeights = [];
  for (let i = 0; i < net.m.length; i++)
    if (net.m[i] !== initialWeights[i]) changedWeights.push([i, net.m[i]]);
  const finalController = controller(agent);
  return { arm: arm.name, curve, dopamineNonzeroSteps, changedWeights,
    initialWeightsAllOne: initialWeights.every(x => x === 1), initialController,
    finalController, controllerChangeNorm: changeNorm(initialController, finalController),
    events, after: evaluate(seed, arm, net.m, finalController) };
}
const reliable = run => run.after.wins >= 58 &&
  run.after.byPrevious.every(x => x.wins / x.trials >= 0.7) &&
  run.after.observedActionCounts.every(n => n >= 12);
const students = seeds.map(seed => {
  const runs = arms.map(arm => train(seed, arm)), [A, B, C] = runs;
  const schedule = A.events.map(e => [e.previous, e.opponent]);
  const matchedSchedule = runs.every(r => same(r.events.map(e => [e.previous, e.opponent]), schedule));
  const heldoutSchedule = A.after.events.map(e => [e.sequenceSeed, e.previous, e.opponent]);
  const matchedHeldoutSchedule = runs.every(r => same(r.after.events.map(e =>
    [e.sequenceSeed, e.previous, e.opponent]), heldoutSchedule));
  const initialUniform = [A, B, C].every(r => r.initialController.W.flat().every(x => x === 0) &&
    r.initialController.b.every(x => x === 0));
  const validity = { matchedSchedule, matchedHeldoutSchedule, initialUniform,
    matchedNonlinearInitialization: same(B.initialController, C.initialController),
    noForcedActions: runs.every(r => r.events.every(e => e.action === e.nativeAction)),
    fullCounts: runs.every(r => r.events.length === 288 && r.after.events.length === 72 &&
      r.after.byPrevious.map(x => x.trials).join(',') === '23,24,22'),
    initialNeuralWeights: runs.every(r => r.initialWeightsAllOne),
    frozenEvaluation: runs.every(r => r.after.weightsUnchanged && r.after.readoutUnchanged && r.after.scorePathMatched),
    frozenSynapses: runs.every(r => r.changedWeights.length === 0),
    dopamineSimulated: runs.every(r => r.dopamineNonzeroSteps > 0),
    rewardLearningChangesReadouts: A.controllerChangeNorm > 0 && B.controllerChangeNorm > 0,
    noLearningControllerUnchanged: C.controllerChangeNorm === 0 && same(C.initialController, C.finalController) };
  const gates = { nonlinearReliable: reliable(B), improvesLinear: B.after.wins > A.after.wins,
    noLearningFailsAndUnchanged: !reliable(C) && C.changedWeights.length === 0 &&
      same(C.initialController, C.finalController) };
  return { seed, runs, validity, gates, passed: Object.values(validity).every(Boolean) &&
    Object.values(gates).every(Boolean) };
});
const result = { command: 'node tools/check_rps_nonlinear_readout.mjs',
  design: { seeds, episodes, rounds, trainSeeds, heldoutSeeds,
    opponent: 'previous_opponent offset 1 noise 0', featureMean, featureStd,
    controller: '3 standardized MBON inputs -> 8 tanh -> 3 logits',
    hiddenInit: 'rng(seed*65537+17) uniform[-sqrt(6/11),+sqrt(6/11)]; all biases/output weights zero',
    policyGradientRate: 0.03, baselineRate: 0.05,
    training: 'softmax sampled; all synapses frozen; no forced actions/messages',
    evaluation: 'deterministic argmax; no exploration or learning',
    trialStreams: 'network rng(seed*10000000 + episodeKey*1000 + round + 1), action rng(key + 501)', arms },
  gates: { wins: 'B >=58/72 for every seed', eachState: 'B >=70% for each state',
    eachAction: 'B >=12/69 choices for every action', paired: 'B > A wins for every seed',
    noLearning: 'C fails full learning criterion and retains unchanged controller and KC->MBON weights' },
  students, allValid: students.every(s => Object.values(s.validity).every(Boolean)),
  passed: students.every(s => s.passed) };
writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({ allValid: result.allValid, passed: result.passed,
  students: students.map(s => ({ seed: s.seed, validity: s.validity, gates: s.gates,
    runs: s.runs.map(r => ({ arm: r.arm, wins: r.after.wins,
      byPrevious: r.after.byPrevious.map(x => `${x.wins}/${x.trials}`),
      observedActions: r.after.observedActionCounts,
      paperMinusScissors: [r.after.byPrevious[0].meanPaperMinusScissors,
        r.after.byPrevious[2].meanPaperMinusScissors],
      curve: r.curve, controllerChangeNorm: r.controllerChangeNorm,
      kcMbonChanges: r.changedWeights.length })) })) }, null, 2));
