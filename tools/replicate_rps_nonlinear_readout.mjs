// Locked, unchanged nonlinear RPS learning replication: student seeds 691-694 only.
// node tools/replicate_rps_nonlinear_readout.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { Opponent, RpsEpisode, RpsNonlinearReadout, result } from '../js/rps.js';

const output = new URL('../data/rps_nonlinear_replication.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite locked nonlinear replication');
const read = path => JSON.parse(readFileSync(new URL(path, import.meta.url)));
const sub = read('../data/subgraph.json');
const prior = read('../data/rps_nonlinear_readout.json');
const normalization = read('../data/rps_reward_readout.json');
if (!prior.allValid || !normalization.allValid ||
  JSON.stringify(prior.design.featureMean) !== JSON.stringify(normalization.design.featureMean) ||
  JSON.stringify(prior.design.featureStd) !== JSON.stringify(normalization.design.featureStd))
  throw Error('Archived nonlinear feature calibration is invalid');
const { featureMean, featureStd } = prior.design;
const seeds = [691, 692, 693, 694];
const trainSeeds = Array.from({ length: 12 }, (_, i) => 41000 + i);
const heldoutSeeds = [51000, 51001, 51002];
const opponent = seed => new Opponent({ strategy: 'previous_opponent', offset: 1, noise: 0, seed });
const controller = agent => ({ W: agent.W.map(row => [...row]), b: [...agent.b], baseline: agent.baseline,
  W1: agent.W1.map(row => [...row]), b1: [...agent.b1] });
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const mean = xs => xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : null;
const counts = xs => [0, 1, 2].map(a => xs.filter(x => x === a).length);
const norm = (a, b) => Math.hypot(...a.map((x, i) => x - b[i]));
const flat = params => [...params.W.flat(), ...params.b, ...params.W1.flat(), ...params.b1];
const softmax = scores => {
  const mx = Math.max(...scores), exp = scores.map(x => Math.exp(x - mx));
  const sum = exp.reduce((s, x) => s + x, 0);
  return exp.map(x => x / sum);
};
function fly(seed) {
  const net = new Network(sub, wiring(sub, 'real', seed), { seed });
  const agent = new RpsNonlinearReadout(net, { seed, featureMean, featureStd,
    freezeSynapses: true, readoutLearn: true });
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
function evaluate(seed, params) {
  const { net, agent } = fly(seed);
  agent.W.forEach((row, a) => row.set(params.W[a]));
  agent.b.set(params.b);
  agent.W1.forEach((row, k) => row.set(params.W1[k]));
  agent.b1.set(params.b1);
  agent.baseline = params.baseline;
  agent.explore = false;
  const frozenController = controller(agent), events = [];
  heldoutSeeds.forEach((sequenceSeed, i) => {
    const decisions = [];
    const ep = new RpsEpisode(net, agent, opponent(sequenceSeed), { rounds: 24, learn: false });
    runEpisode(ep, seed, 100 + i, (reader, event) => decisions.push({
      features: [...reader.f], scores: [...reader.scores], action: event.action }));
    ep.events.forEach((event, round) => {
      const decision = decisions[round], probabilities = softmax(decision.scores);
      const correct = (event.opponent + 1) % 3;
      events.push({ sequenceSeed, round, previous: event.observation.previousOpponent,
        opponent: event.opponent, action: event.action, reward: event.reward,
        rates: event.rates, features: decision.features, scores: decision.scores,
        probabilities, correctMargin: decision.scores[correct] -
          Math.max(...decision.scores.filter((_, a) => a !== correct)) });
    });
  });
  const observed = events.filter(e => e.previous !== null);
  const byPrevious = [0, 1, 2].map(previous => {
    const xs = observed.filter(e => e.previous === previous);
    return { previous, trials: xs.length, wins: xs.filter(e => e.reward === 1).length,
      selected: counts(xs.map(e => e.action)),
      correctMarginMean: mean(xs.map(e => e.correctMargin)),
      correctProbabilityMean: mean(xs.map(e => e.probabilities[(e.opponent + 1) % 3])) };
  });
  const wrong = events.filter(e => e.reward !== 1);
  return { wins: events.filter(e => e.reward === 1).length, byPrevious,
    observedActionCounts: counts(observed.map(e => e.action)),
    state0CorrectMarginMean: byPrevious[0].correctMarginMean,
    incorrect: { count: wrong.length, selectedProbabilityMean: mean(wrong.map(e => e.probabilities[e.action])),
      correctProbabilityMean: mean(wrong.map(e => e.probabilities[(e.opponent + 1) % 3])),
      selectedProbabilityAtLeast07: wrong.filter(e => e.probabilities[e.action] >= 0.7).length },
    validity: { fullCounts: events.length === 72 && byPrevious.map(x => x.trials).join(',') === '23,24,22',
      actionIsArgmax: events.every(e => e.action === e.scores.indexOf(Math.max(...e.scores))),
      rewardCorrect: events.every(e => e.reward === result(e.action, e.opponent)),
      scoreInputsMatched: events.every(e => e.features.every((x, j) =>
        Math.abs(x - (Math.log1p(e.rates[j]) - featureMean[j]) / featureStd[j]) < 1e-12)),
      weightsFrozen: net.m.every(x => x === 1), controllerFrozen: same(controller(agent), frozenController) },
    events };
}
let firstTrainingSchedule, matchedTrainingSchedule = true;
const students = seeds.map(seed => {
  const { net, agent } = fly(seed), initial = controller(agent), curve = [], training = [];
  const initialNeuralAllOne = net.m.every(x => x === 1);
  let dopamineNonzeroSteps = 0;
  trainSeeds.forEach((sequenceSeed, episodeIndex) => {
    const ep = new RpsEpisode(net, agent, opponent(sequenceSeed),
      { rounds: 24, learn: true, balancedActions: false });
    runEpisode(ep, seed, episodeIndex, null, D => { dopamineNonzeroSteps += Number(D !== 0); });
    curve.push(ep.events.filter(e => e.reward === 1).length);
    training.push(...ep.events.map(e => ({ episode: episodeIndex, previous: e.observation.previousOpponent,
      opponent: e.opponent, action: e.action, nativeAction: e.nativeAction, reward: e.reward })));
  });
  const final = controller(agent), after = evaluate(seed, final);
  const observed = training.filter(e => e.previous !== null);
  const trainingSchedule = training.map(e => [e.previous, e.opponent]);
  if (firstTrainingSchedule) matchedTrainingSchedule &&= same(trainingSchedule, firstTrainingSchedule);
  else firstTrainingSchedule = trainingSchedule;
  const kcMbonChangedWeights = net.m.reduce((n, x) => n + Number(x !== 1), 0);
  const validity = { initialAllOne: initialNeuralAllOne, initialUniform: initial.W.flat().every(x => x === 0) &&
      initial.b.every(x => x === 0), fullTraining: training.length === 288 && curve.length === 12,
    noForcedActions: training.every(e => e.action === e.nativeAction),
    trainingRewardsCorrect: training.every(e => e.reward === result(e.action, e.opponent)),
    frozenSynapses: kcMbonChangedWeights === 0, dopamineSimulated: dopamineNonzeroSteps > 0,
    rewardLearnedController: norm(flat(initial), flat(final)) > 0,
    frozenEvaluation: Object.values(after.validity).every(Boolean) };
  const gates = { wins: after.wins >= 58,
    eachState: after.byPrevious.every(x => x.wins / x.trials >= 0.7),
    eachAction: after.observedActionCounts.every(n => n >= 12) };
  return { seed, curve, dopamineNonzeroSteps,
    trainObservedActionCounts: counts(observed.map(e => e.action)),
    trainState0RewardedScissors: observed.filter(e => e.previous === 0 && e.action === 2 && e.reward === 1).length,
    controllerChangeNorm: norm(flat(initial), flat(final)), initialController: initial, finalController: final,
    kcMbonChangedWeights, after, validity, gates,
    valid: Object.values(validity).every(Boolean), passed: Object.values(validity).every(Boolean) &&
      Object.values(gates).every(Boolean) };
});
const schedule = students[0].after.events.map(e => [e.sequenceSeed, e.previous, e.opponent]);
const matchedHeldoutSchedule = students.every(s => same(s.after.events.map(e =>
  [e.sequenceSeed, e.previous, e.opponent]), schedule));
const resultData = { command: 'node tools/replicate_rps_nonlinear_readout.mjs',
  design: { seeds, trainingEpisodes: 12, roundsPerEpisode: 24, trainSeeds, heldoutSeeds,
    opponent: 'previous_opponent offset 1 noise 0', featureMean, featureStd,
    controller: prior.design.controller, policyGradientRate: prior.design.policyGradientRate,
    baselineRate: prior.design.baselineRate, hiddenInit: prior.design.hiddenInit,
    trialStreams: prior.design.trialStreams,
    training: 'existing nonlinear sampled-softmax reward learning; frozen KC->MBON',
    evaluation: 'original deterministic argmax, learning and exploration off' },
  gates: { wins: '>=58/72', eachState: '>=70% in each observed previous-move state',
    eachAction: '>=12/69 for rock, paper, scissors' },
  matchedTrainingSchedule, matchedHeldoutSchedule, students,
  allValid: matchedTrainingSchedule && matchedHeldoutSchedule && students.every(s => s.valid),
  allPassed: matchedTrainingSchedule && matchedHeldoutSchedule && students.every(s => s.passed) };
writeFileSync(output, `${JSON.stringify(resultData, null, 2)}\n`);
console.log(JSON.stringify({ allValid: resultData.allValid, allPassed: resultData.allPassed,
  students: students.map(s => ({ seed: s.seed, wins: s.after.wins,
    byPrevious: s.after.byPrevious.map(x => `${x.wins}/${x.trials}`),
    actions: s.after.observedActionCounts, state0Margin: s.after.state0CorrectMarginMean,
    incorrect: s.after.incorrect, controllerChangeNorm: s.controllerChangeNorm,
    kcMbonChangedWeights: s.kcMbonChangedWeights, validity: s.validity,
    gates: s.gates })) }, null, 2));
