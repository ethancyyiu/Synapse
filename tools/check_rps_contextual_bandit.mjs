// One locked comparison: nonlinear REINFORCE versus action-specific contextual bandit.
// node tools/check_rps_contextual_bandit.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { Opponent, RpsEpisode, RpsNonlinearReadout, RpsContextualBandit, result } from '../js/rps.js';

const output = new URL('../data/rps_contextual_bandit.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite locked contextual-bandit result');
const read = path => JSON.parse(readFileSync(new URL(path, import.meta.url)));
const sub = read('../data/subgraph.json');
const previous = read('../data/rps_nonlinear_readout.json');
const normalization = read('../data/rps_reward_readout.json');
if (!previous.allValid || !normalization.allValid ||
  JSON.stringify(previous.design.featureMean) !== JSON.stringify(normalization.design.featureMean) ||
  JSON.stringify(previous.design.featureStd) !== JSON.stringify(normalization.design.featureStd))
  throw Error('Required archived MBON feature calibration is invalid');
const { featureMean, featureStd } = previous.design;
const seeds = [711, 712, 713, 714], episodes = 12, rounds = 24;
const trainSeeds = Array.from({ length: episodes }, (_, i) => 41000 + i);
const heldoutSeeds = [51000, 51001, 51002];
const arms = [
  { name: 'nonlinear_reinforce', bandit: false, readoutLearn: true },
  { name: 'contextual_bandit', bandit: true, readoutLearn: true },
  { name: 'contextual_bandit_no_learning', bandit: true, readoutLearn: false },
];
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const counts = xs => [0, 1, 2].map(a => xs.filter(x => x === a).length);
const mean = xs => xs.reduce((s, x) => s + x, 0) / xs.length;
const opponent = seed => new Opponent({ strategy: 'previous_opponent', offset: 1, noise: 0, seed });
const controller = agent => ({ W: agent.W.map(row => [...row]), b: [...agent.b], baseline: agent.baseline,
  ...(agent.W1 ? { W1: agent.W1.map(row => [...row]), b1: [...agent.b1] } : {}) });
const flat = p => [...p.W.flat(), ...p.b, ...(p.W1 ? [...p.W1.flat(), ...p.b1] : [])];
const changeNorm = (before, after) => Math.hypot(...flat(after).map((x, i) => x - flat(before)[i]));
const approx = (a, b) => Math.abs(a - b) < 1e-12;
function fly(seed, arm) {
  const net = new Network(sub, wiring(sub, 'real', seed), { seed });
  const Agent = arm.bandit ? RpsContextualBandit : RpsNonlinearReadout;
  const agent = new Agent(net, { seed, featureMean, featureStd,
    freezeSynapses: true, readoutLearn: arm.readoutLearn });
  return { net, agent };
}
function runEpisode(ep, seed, episodeKey, onDecision = null, onUpdate = null) {
  while (!ep.done) {
    const phase = ep.t % 400;
    if (phase === 0) {
      const round = Math.floor(ep.t / 400), key = seed * 10000000 + episodeKey * 1000 + round;
      ep.net.rand = rng(key + 1); ep.agent.rand = rng(key + 501);
    }
    const before = phase === 340 && onUpdate ? controller(ep.agent) : null;
    ep.step();
    if (phase === 240 && onDecision) onDecision(ep.agent, ep.events.at(-1));
    if (phase === 340 && onUpdate) onUpdate(before, ep.agent, ep.events.at(-1));
  }
  return ep;
}
function scoreFrom(params, f) {
  const h = params.W1?.map((row, k) => Math.tanh(row.reduce((s, x, j) => s + x * f[j], params.b1[k])));
  return params.W.map((row, a) => row.reduce((s, x, j) => s + x * (h ?? f)[j], params.b[a]));
}
function evaluate(seed, arm, params) {
  const { net, agent } = fly(seed, arm);
  agent.W.forEach((row, a) => row.set(params.W[a]));
  agent.b.set(params.b); agent.baseline = params.baseline;
  if (!arm.bandit) { agent.W1.forEach((row, k) => row.set(params.W1[k])); agent.b1.set(params.b1); }
  agent.explore = false;
  const frozenController = controller(agent), events = [];
  let scorePathMatched = true;
  heldoutSeeds.forEach((sequenceSeed, i) => {
    const decisions = [], ep = new RpsEpisode(net, agent, opponent(sequenceSeed), { rounds, learn: false });
    runEpisode(ep, seed, 100 + i, (reader, event) => decisions.push({
      rates: [...reader.rates], features: [...reader.f], scores: [...reader.scores], action: event.action }));
    ep.events.forEach((event, round) => {
      const decision = decisions[round], features = event.rates.map((r, j) =>
        (Math.log1p(r) - featureMean[j]) / featureStd[j]);
      const scores = scoreFrom(frozenController, features);
      scorePathMatched &&= decision.action === event.action &&
        event.action === scores.indexOf(Math.max(...scores)) &&
        event.rates.every((x, j) => x === decision.rates[j]) &&
        features.every((x, j) => approx(x, decision.features[j])) &&
        scores.every((x, j) => approx(x, decision.scores[j]));
      events.push({ sequenceSeed, round, previous: event.observation.previousOpponent,
        opponent: event.opponent, action: event.action, reward: event.reward,
        rates: [...event.rates], features, scores });
    });
  });
  const observed = events.filter(e => e.previous !== null);
  const byPrevious = [0, 1, 2].map(previous => {
    const xs = observed.filter(e => e.previous === previous);
    return { previous, trials: xs.length, wins: xs.filter(e => e.reward === 1).length,
      actionCounts: counts(xs.map(e => e.action)),
      meanCorrectMargin: mean(xs.map(e => {
        const correct = (e.opponent + 1) % 3;
        return e.scores[correct] - Math.max(...e.scores.filter((_, a) => a !== correct));
      })) };
  });
  return { wins: events.filter(e => e.reward === 1).length,
    observedActionCounts: counts(observed.map(e => e.action)), byPrevious,
    scorePathMatched, weightsUnchanged: net.m.every(x => x === 1),
    controllerUnchanged: same(controller(agent), frozenController),
    rewardsCorrect: events.every(e => e.reward === result(e.action, e.opponent)), events };
}
function train(seed, arm) {
  const { net, agent } = fly(seed, arm), initialController = controller(agent);
  const initialNeuralAllOne = net.m.every(x => x === 1), curve = [], events = [];
  let dopamineNonzeroSteps = 0, selectedHeadOnly = true, selectedHeadRule = true;
  trainSeeds.forEach((sequenceSeed, episodeIndex) => {
    const ep = new RpsEpisode(net, agent, opponent(sequenceSeed),
      { rounds, learn: true, balancedActions: false });
    runEpisode(ep, seed, episodeIndex, null, (before, reader, event) => {
      dopamineNonzeroSteps += Number(reader.D !== 0);
      if (!arm.bandit) return;
      const a = event.action, after = controller(reader);
      selectedHeadOnly &&= [0, 1, 2].every(k => k === a ||
        same(before.W[k], after.W[k]) && before.b[k] === after.b[k]);
      const step = arm.readoutLearn ? reader.alpha * (event.reward - reader.scores[a]) : 0;
      selectedHeadRule &&= approx(after.b[a], before.b[a] + step) &&
        [0, 1, 2].every(j => approx(after.W[a][j], before.W[a][j] + step * reader.f[j]));
    });
    curve.push(ep.events.filter(e => e.reward === 1).length);
    events.push(...ep.events.map(e => ({ episode: episodeIndex, round: e.round,
      previous: e.observation.previousOpponent, opponent: e.opponent,
      action: e.action, nativeAction: e.nativeAction, reward: e.reward })));
  });
  const finalController = controller(agent), after = evaluate(seed, arm, finalController);
  const kcMbonChangedWeights = net.m.reduce((n, x) => n + Number(x !== 1), 0);
  return { arm: arm.name, curve, dopamineNonzeroSteps, kcMbonChangedWeights,
    initialNeuralAllOne, initialController, finalController,
    controllerChangeNorm: changeNorm(initialController, finalController),
    selectedHeadOnly, selectedHeadRule, events, after };
}
const reliable = run => run.after.wins >= 58 &&
  run.after.byPrevious.every(x => x.wins / x.trials >= 0.7) &&
  run.after.observedActionCounts.every(n => n >= 12);
const students = seeds.map(seed => {
  const runs = arms.map(arm => train(seed, arm)), [A, B, C] = runs;
  const trainingSchedule = A.events.map(e => [e.previous, e.opponent]);
  const heldoutSchedule = A.after.events.map(e => [e.sequenceSeed, e.previous, e.opponent]);
  const validity = { matchedTrainingSchedule: runs.every(r =>
      same(r.events.map(e => [e.previous, e.opponent]), trainingSchedule)),
    matchedHeldoutSchedule: runs.every(r =>
      same(r.after.events.map(e => [e.sequenceSeed, e.previous, e.opponent]), heldoutSchedule)),
    matchedBanditInitialization: same(B.initialController, C.initialController),
    initialUniform: runs.every(r => r.initialController.W.flat().every(x => x === 0) &&
      r.initialController.b.every(x => x === 0)),
    initialNeuralAllOne: runs.every(r => r.initialNeuralAllOne),
    fullCounts: runs.every(r => r.events.length === 288 && r.after.events.length === 72 &&
      r.after.byPrevious.map(x => x.trials).join(',') === '23,24,22'),
    noForcedActions: runs.every(r => r.events.every(e => e.action === e.nativeAction)),
    rewardsCorrect: runs.every(r => r.events.every(e => e.reward === result(e.action, e.opponent)) &&
      r.after.rewardsCorrect),
    frozenSynapses: runs.every(r => r.kcMbonChangedWeights === 0),
    dopamineSimulated: runs.every(r => r.dopamineNonzeroSteps > 0),
    frozenEvaluation: runs.every(r => r.after.weightsUnchanged && r.after.controllerUnchanged &&
      r.after.scorePathMatched),
    actionSpecificBanditUpdate: [B, C].every(r => r.selectedHeadOnly && r.selectedHeadRule),
    learningChangesControllers: A.controllerChangeNorm > 0 && B.controllerChangeNorm > 0,
    noLearningUnchanged: C.controllerChangeNorm === 0 && same(C.initialController, C.finalController) };
  const gates = { banditReliable: reliable(B), beatsReinforce: B.after.wins > A.after.wins,
    noLearningFailsAndUnchanged: !reliable(C) && C.kcMbonChangedWeights === 0 &&
      same(C.initialController, C.finalController) };
  return { seed, runs, validity, gates,
    passed: Object.values(validity).every(Boolean) && Object.values(gates).every(Boolean) };
});
const outputData = { command: 'node tools/check_rps_contextual_bandit.mjs',
  design: { seeds, episodes, rounds, trainSeeds, heldoutSeeds,
    opponent: 'previous_opponent offset 1 noise 0', featureMean, featureStd,
    bandit: 'three zero-initialized linear Q heads; epsilon-greedy training; selected-action squared-error update',
    banditAlpha: 0.03, banditEpsilon: 0.20,
    reinforce: previous.design.controller,
    trialStreams: 'network rng(seed*10000000+episodeKey*1000+round+1); action rng(key+501)',
    evaluation: 'deterministic argmax; no exploration or learning', arms },
  gates: { wins: 'B >=58/72 for every seed', eachState: 'B >=70% for each observed state',
    eachAction: 'B >=12/69 observed choices per action', paired: 'B > A wins for every seed',
    noLearning: 'C fails reliability and retains unchanged controller and KC->MBON weights' },
  students, allValid: students.every(s => Object.values(s.validity).every(Boolean)),
  passed: students.every(s => s.passed) };
writeFileSync(output, `${JSON.stringify(outputData, null, 2)}\n`);
console.log(JSON.stringify({ allValid: outputData.allValid, passed: outputData.passed,
  students: students.map(s => ({ seed: s.seed, validity: s.validity, gates: s.gates,
    runs: s.runs.map(r => ({ arm: r.arm, wins: r.after.wins,
      byPrevious: r.after.byPrevious.map(x => `${x.wins}/${x.trials}`),
      observedActions: r.after.observedActionCounts,
      state0Margin: r.after.byPrevious[0].meanCorrectMargin,
      curve: r.curve, controllerChangeNorm: r.controllerChangeNorm,
      kcMbonChanges: r.kcMbonChangedWeights })) })) }, null, 2));
