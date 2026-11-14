// One locked opt-in RPS MBON-baseline experiment; see docs/lab-notebook.md.
// node tools/check_rps_normalized_readout.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { Opponent, RpsBioRL, RpsEpisode } from '../js/rps.js';

const output = new URL('../data/rps_normalized_readout.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite normalized-readout result');
const calibration = JSON.parse(readFileSync(new URL('../data/rps_rate_baseline.json', import.meta.url)));
if (!calibration.valid || calibration.baseline?.length !== 3 ||
  !calibration.baseline.every(Number.isFinite)) throw Error('Fixed calibration did not pass');
const rateBaseline = Object.freeze([...calibration.baseline]);
const raw = JSON.parse(readFileSync(new URL('../data/rps_balanced_training_corrected.json', import.meta.url)));
const sub = JSON.parse(readFileSync(new URL('../data/subgraph.json', import.meta.url)));
const seeds = [651, 652, 653], episodes = 12, rounds = 24;
const trainSeeds = Array.from({ length: episodes }, (_, i) => 41000 + i);
const heldoutSeeds = [51000, 51001, 51002];
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const counts = xs => [0, 1, 2].map(a => xs.filter(x => x === a).length);
if (!raw.allValid || !same(raw.design.seeds, seeds) || !same(raw.design.trainSeeds, trainSeeds) ||
  !same(raw.design.heldoutSeeds, heldoutSeeds) || raw.design.episodes !== episodes ||
  raw.design.rounds !== rounds || !raw.design.trialKeyedNeuralAndTieStreams ||
  raw.students.length !== seeds.length) throw Error('Matched raw-readout control failed provenance check');
const opponent = seed => new Opponent({ strategy: 'previous_opponent', offset: 1, noise: 0, seed });
function fly(seed) {
  const net = new Network(sub, wiring(sub, 'real', seed), { seed });
  const agent = new RpsBioRL(net, { seed, rateBaseline });
  return { net, agent };
}
function runEpisode(ep, studentSeed, episodeKey, onLearningStep = null) {
  while (!ep.done) {
    const phase = ep.t % 400;
    if (phase === 0) {
      const round = Math.floor(ep.t / 400);
      const key = studentSeed * 10000000 + episodeKey * 1000 + round;
      ep.net.rand = rng(key + 1);
      ep.agent.rand = rng(key + 501);
    }
    if (phase !== 340 || !onLearningStep) { ep.step(); continue; }
    const event = ep.events.at(-1);
    const edges = event.reward === 1 ? ep.net.plastic[event.action] : null;
    const prior = edges ? Float64Array.from(edges, i => ep.net.m[i]) : null;
    ep.step();
    if (edges) onLearningStep(event, edges.some((i, n) => ep.net.m[i] !== prior[n]));
  }
}
function evaluate(seed, weights) {
  const { net, agent } = fly(seed);
  net.m.set(weights);
  const frozen = net.m.slice(), events = [];
  heldoutSeeds.forEach((opponentSeed, i) => {
    const ep = new RpsEpisode(net, agent, opponent(opponentSeed),
      { rounds, learn: false, balancedActions: false });
    runEpisode(ep, seed, 100 + i);
    events.push(...ep.events.map(e => ({ sequenceSeed: opponentSeed, round: e.round,
      previous: e.observation.previousOpponent, opponent: e.opponent, action: e.action,
      nativeAction: e.nativeAction, reward: e.reward, rates: e.rates })));
  });
  const observed = events.filter(e => e.previous !== null);
  return { wins: events.filter(e => e.reward === 1).length,
    observedActionCounts: counts(observed.map(e => e.action)),
    byPrevious: [0, 1, 2].map(previous => {
      const sample = observed.filter(e => e.previous === previous);
      return { previous, trials: sample.length,
        wins: sample.filter(e => e.reward === 1).length,
        actionCounts: counts(sample.map(e => e.action)) };
    }), weightsUnchanged: net.m.every((m, i) => m === frozen[i]),
    nativeReadoutOnly: events.every(e => e.action === e.nativeAction),
    baselineFrozen: Object.isFrozen(agent.o.rateBaseline) && same(agent.o.rateBaseline, rateBaseline),
    events };
}
function train(seed, learn) {
  const { net, agent } = fly(seed), initial = net.m.slice();
  const balancedCounts = [0, 0, 0], events = [], curve = [];
  const positiveRewardTrials = [0, 0, 0], positiveRewardWeightChangingSteps = [0, 0, 0];
  trainSeeds.forEach((opponentSeed, episodeIndex) => {
    const ep = new RpsEpisode(net, agent, opponent(opponentSeed),
      { rounds, learn, balancedActions: true, balancedCounts });
    runEpisode(ep, seed, episodeIndex, (event, changed) => {
      positiveRewardTrials[event.action]++;
      if (changed) positiveRewardWeightChangingSteps[event.action]++;
    });
    curve.push(ep.events.filter(e => e.reward === 1).length);
    events.push(...ep.events.map(e => ({ episode: episodeIndex, round: e.round,
      previous: e.observation.previousOpponent, opponent: e.opponent,
      action: e.action, nativeAction: e.nativeAction, reward: e.reward, rates: e.rates })));
  });
  const changedWeights = [];
  for (let i = 0; i < net.m.length; i++)
    if (net.m[i] !== initial[i]) changedWeights.push([i, net.m[i]]);
  return { name: learn ? 'normalized_learning' : 'normalized_no_learning',
    initialWeightsMatched: initial.every(m => m === 1),
    baselineFrozen: Object.isFrozen(agent.o.rateBaseline) && same(agent.o.rateBaseline, rateBaseline),
    curve, positiveRewardTrials, positiveRewardWeightChangingSteps,
    changedWeights, events, after: evaluate(seed, net.m) };
}

const students = seeds.map(seed => {
  const rawStudent = raw.students.find(s => s.seed === seed);
  const rawLearning = rawStudent?.runs.find(r => r.arm === 'balanced');
  const rawNoLearning = rawStudent?.runs.find(r => r.arm === 'balanced_no_learning');
  if (!rawStudent?.validity.valid || !rawLearning || !rawNoLearning)
    throw Error(`Missing valid raw controls: ${seed}`);
  const before = evaluate(seed, fly(seed).net.m);
  const learning = train(seed, true), noLearning = train(seed, false);
  const sourceSchedule = rawLearning.events.map(e => [e.previous, e.opponent]);
  const opponentsMatch = [learning, noLearning].every(r =>
    same(r.events.map(e => [e.previous, e.opponent]), sourceSchedule));
  const observedActionsMatch = [learning, noLearning].every(r => same(
    r.events.filter(e => e.previous !== null).map(e => e.action),
    rawLearning.events.filter(e => e.previous !== null).map(e => e.action)));
  const balancedWithinState = [learning, noLearning].every(r =>
    [0, 1, 2].every(previous => {
      const selected = counts(r.events.filter(e => e.previous === previous).map(e => e.action));
      return Math.max(...selected) - Math.min(...selected) <= 1;
    }));
  const noLearningUnchanged = noLearning.changedWeights.length === 0 &&
    same(noLearning.after.events, before.events);
  const valid = opponentsMatch && observedActionsMatch && balancedWithinState &&
    noLearningUnchanged && [learning, noLearning].every(r => r.initialWeightsMatched &&
      r.baselineFrozen && r.events.length === episodes * rounds &&
      r.after.weightsUnchanged && r.after.nativeReadoutOnly && r.after.baselineFrozen &&
      r.after.events.length === 72 &&
      r.after.byPrevious.reduce((n, x) => n + x.trials, 0) === 69);
  const gates = { wins: learning.after.wins >= 58,
    everyState: learning.after.byPrevious.every(r => r.wins / r.trials >= .7),
    everyAction: learning.after.observedActionCounts.every(n => n >= 12),
    gainOverRaw: learning.after.wins - rawLearning.after.wins >= 10,
    noLearningControl: noLearningUnchanged && noLearning.after.wins < 58 };
  return { seed, before, rawControls: { learningWins: rawLearning.after.wins,
    learningByPrevious: rawLearning.after.byPrevious,
    learningObservedActionCounts: rawLearning.after.observedActionCounts,
    noLearningWins: rawNoLearning.after.wins },
    learning, noLearning, validity: { opponentsMatch, observedActionsMatch,
      balancedWithinState, noLearningUnchanged, valid }, gates,
    passed: valid && Object.values(gates).every(Boolean) };
});
const result = { command: 'node tools/check_rps_normalized_readout.mjs',
  calibrationFile: 'data/rps_rate_baseline.json', rateBaseline,
  rawControlFile: 'data/rps_balanced_training_corrected.json',
  design: { seeds, episodes, rounds, trainSeeds, heldoutSeeds,
    balancedActions: true, learningArms: [true, false],
    trialKeyedNeuralAndTieStreams: true, opponent: 'previous_opponent, offset 1, noise 0',
    frozenEvaluation: 'same fixed normalized readout, no exploration, learn false' },
  gate: { wins: '>=58/72 per seed', everyState: '>=70% per observed state',
    everyAction: '>=12/69', gainOverRaw: '>=10/72 per seed',
    noLearningControl: 'weights unchanged, identical pre/post frozen behavior, <58/72' },
  students, allValid: students.every(s => s.validity.valid),
  passed: students.every(s => s.passed) };
writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({ rateBaseline, allValid: result.allValid, passed: result.passed,
  students: students.map(s => ({ seed: s.seed, normalizedBefore: s.before.wins,
    rawLearning: s.rawControls.learningWins,
    normalizedLearning: { wins: s.learning.after.wins,
      byPrevious: s.learning.after.byPrevious.map(r => r.wins),
      observedActionCounts: s.learning.after.observedActionCounts,
      positiveRewardWeightChangingSteps: s.learning.positiveRewardWeightChangingSteps },
    normalizedNoLearning: s.noLearning.after.wins, validity: s.validity, gates: s.gates })) }, null, 2));
