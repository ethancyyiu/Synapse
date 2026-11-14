// Approved, locked RPS training comparison; see docs/lab-notebook.md.
// node tools/check_rps_balanced_training.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { Opponent, RpsBioRL, RpsEpisode } from '../js/rps.js';

const output = new URL('../data/rps_balanced_training_corrected.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite balanced-training result');
const sub = JSON.parse(readFileSync(new URL('../data/subgraph.json', import.meta.url)));
const seeds = [651, 652, 653], episodes = 12, rounds = 24;
const trainSeeds = Array.from({ length: episodes }, (_, i) => 41000 + i);
const heldoutSeeds = [51000, 51001, 51002];
const arms = [
  { name: 'autonomous', balancedActions: false, learn: true },
  { name: 'balanced', balancedActions: true, learn: true },
  { name: 'balanced_no_learning', balancedActions: true, learn: false }
];
const opponent = seed => new Opponent({ strategy: 'previous_opponent', offset: 1, noise: 0, seed });
const counts = xs => [0, 1, 2].map(a => xs.filter(x => x === a).length);
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

function fly(seed) {
  const net = new Network(sub, wiring(sub, 'real', seed), { seed });
  return { net, agent: new RpsBioRL(net, { seed }) };
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
    ep.step(); // The original afterDopamine path performs (or skips) learning.
    if (edges) onLearningStep(event, edges.some((i, n) => ep.net.m[i] !== prior[n]));
  }
  return ep;
}
function evaluate(studentSeed, weights) {
  const { net, agent } = fly(studentSeed);
  net.m.set(weights);
  const frozen = net.m.slice(), events = [];
  heldoutSeeds.forEach((seed, i) => {
    const ep = new RpsEpisode(net, agent, opponent(seed),
      { rounds, learn: false, balancedActions: false });
    runEpisode(ep, studentSeed, 100 + i);
    events.push(...ep.events.map(e => ({ sequenceSeed: seed, round: e.round,
      previous: e.observation.previousOpponent, opponent: e.opponent,
      action: e.action, nativeAction: e.nativeAction, reward: e.reward, rates: e.rates })));
  });
  const weightsUnchanged = net.m.every((m, i) => m === frozen[i]);
  const nativeReadoutOnly = events.every(e => e.action === e.nativeAction);
  const observed = events.filter(e => e.previous !== null);
  return { wins: events.filter(e => e.reward === 1).length,
    observedActionCounts: counts(observed.map(e => e.action)),
    byPrevious: [0, 1, 2].map(previous => {
      const sample = observed.filter(e => e.previous === previous);
      return { previous, trials: sample.length,
        wins: sample.filter(e => e.reward === 1).length,
        actionCounts: counts(sample.map(e => e.action)) };
    }), weightsUnchanged, nativeReadoutOnly, events };
}
function train(studentSeed, arm) {
  const { net, agent } = fly(studentSeed), initial = net.m.slice();
  const events = [], curve = [];
  const balancedCounts = [0, 0, 0];
  const positiveRewardTrials = [0, 0, 0], positiveRewardWeightChangingSteps = [0, 0, 0];
  trainSeeds.forEach((seed, episodeIndex) => {
    const ep = new RpsEpisode(net, agent, opponent(seed),
      { rounds, learn: arm.learn, balancedActions: arm.balancedActions, balancedCounts });
    runEpisode(ep, studentSeed, episodeIndex, (event, changed) => {
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
  return { arm: arm.name, curve, actionCounts: counts(events.map(e => e.action)),
    positiveRewardTrials, positiveRewardWeightChangingSteps, changedWeights,
    initialWeightsMatched: initial.every(m => m === 1), events,
    after: evaluate(studentSeed, net.m) };
}

const students = seeds.map(seed => {
  const before = evaluate(seed, fly(seed).net.m);
  const runs = arms.map(arm => train(seed, arm));
  const [autonomous, balanced, noLearning] = runs;
  const opponentsMatch = runs.every(run => same(run.events.map(e => [e.previous, e.opponent]),
    autonomous.events.map(e => [e.previous, e.opponent])));
  const balancedActionSchedulesMatch = same(
    balanced.events.filter(e => e.previous !== null).map(e => e.action),
    noLearning.events.filter(e => e.previous !== null).map(e => e.action));
  const balancedActionsEqualByState = [balanced, noLearning].every(run =>
    [0, 1, 2].every(previous => {
      const actionCounts = counts(run.events.filter(e => e.previous === previous).map(e => e.action));
      return Math.max(...actionCounts) - Math.min(...actionCounts) <= 1;
    }));
  const noLearningWeightsUnchanged = noLearning.changedWeights.length === 0;
  const noLearningFrozenMatchesBefore = same(noLearning.after.events, before.events);
  const valid = opponentsMatch && balancedActionSchedulesMatch && balancedActionsEqualByState &&
    noLearningWeightsUnchanged &&
    noLearningFrozenMatchesBefore && runs.every(r => r.initialWeightsMatched &&
      r.after.weightsUnchanged && r.after.nativeReadoutOnly &&
      r.events.length === episodes * rounds && r.after.events.length === 72 &&
      r.after.byPrevious.reduce((n, x) => n + x.trials, 0) === 69);
  const gates = {
    paperRewardedChanges: balanced.positiveRewardWeightChangingSteps[1] > 0,
    finalWins: balanced.after.wins >= 58,
    gainOverAutonomous: balanced.after.wins - autonomous.after.wins >= 10,
    eachState: balanced.after.byPrevious.every(r => r.wins / r.trials >= .7),
    eachAction: balanced.after.observedActionCounts.every(n => n >= 12),
    noLearningControl: noLearningWeightsUnchanged &&
      noLearning.after.wins - autonomous.after.wins < 10
  };
  return { seed, before, runs, validity: { opponentsMatch,
    balancedActionSchedulesMatch, balancedActionsEqualByState, noLearningWeightsUnchanged,
    noLearningFrozenMatchesBefore, valid }, gates,
    passed: valid && Object.values(gates).every(Boolean) };
});
const result = { command: 'node tools/check_rps_balanced_training.mjs',
  design: { seeds, episodes, rounds, trainSeeds, heldoutSeeds,
    opponent: 'previous_opponent, offset 1, noise 0',
    balancedPolicy: 'cycle actions 0,1,2 separately per observed previous move across all 12 episodes',
    initialRound: 'autonomous', trialKeyedNeuralAndTieStreams: true,
    frozenEvaluation: 'original RpsBioRL.decide, balancedActions false, learn false' },
  gate: { paperRewardedChanges: '>0 per student', finalWins: '>=58/72',
    gainOverAutonomous: '>=10/72', eachObservedState: '>=70%',
    eachAction: '>=12/69', noLearningControl: 'unchanged weights and <10/72 gain over autonomous' },
  students, allValid: students.every(s => s.validity.valid),
  passed: students.every(s => s.passed) };
writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({ allValid: result.allValid, passed: result.passed,
  students: students.map(s => ({ seed: s.seed, before: s.before.wins,
    arms: s.runs.map(r => ({ arm: r.arm, wins: r.after.wins,
      byPrevious: r.after.byPrevious.map(x => x.wins),
      observedActionCounts: r.after.observedActionCounts,
      positiveRewardWeightChangingSteps: r.positiveRewardWeightChangingSteps,
      changedWeights: r.changedWeights.length })), validity: s.validity, gates: s.gates })) }, null, 2));
