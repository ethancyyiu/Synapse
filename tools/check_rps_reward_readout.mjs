// Locked Path B RPS comparison. Reuses the valid saved MBON arms; trains only the opt-in hybrid arms.
// node tools/check_rps_reward_readout.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { Opponent, RpsRewardReadout, RpsEpisode } from '../js/rps.js';

const output = new URL('../data/rps_reward_readout.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite reward-readout result');
const info = JSON.parse(readFileSync(new URL('../data/rps_readout_information.json', import.meta.url)));
if (!info.passed) throw Error('Information gate failed');
const prior = JSON.parse(readFileSync(new URL('../data/rps_balanced_training_corrected.json', import.meta.url)));
if (!prior.allValid || prior.students.some(s => !s.validity.valid)) throw Error('Saved MBON control invalid');
const probe = JSON.parse(readFileSync(new URL('../data/rps_frozen_input.json', import.meta.url)));
const sub = JSON.parse(readFileSync(new URL('../data/subgraph.json', import.meta.url)));
const seeds = [651, 652, 653], episodes = 12, rounds = 24;
const trainSeeds = Array.from({ length: episodes }, (_, i) => 41000 + i);
const heldoutSeeds = [51000, 51001, 51002];
const xs = probe.trials.map(r => r.rates.map(x => Math.log1p(x)));
if (xs.length !== 108) throw Error('Invalid feature calibration');
const featureMean = [0, 1, 2].map(j => xs.reduce((s, x) => s + x[j], 0) / xs.length);
const featureStd = [0, 1, 2].map(j => Math.sqrt(xs.reduce((s, x) => s + (x[j] - featureMean[j]) ** 2, 0) / xs.length));
if (featureStd.some(x => !Number.isFinite(x) || x <= 0)) throw Error('Invalid feature scales');
const opponent = seed => new Opponent({ strategy: 'previous_opponent', offset: 1, noise: 0, seed });
const counts = xs => [0, 1, 2].map(a => xs.filter(x => x === a).length);
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
function fly(seed, readoutLearn) {
  const net = new Network(sub, wiring(sub, 'real', seed), { seed });
  return { net, agent: new RpsRewardReadout(net, { seed, featureMean, featureStd, readoutLearn }) };
}
function readout(agent) { return { W: agent.W.map(w => [...w]), b: [...agent.b], baseline: agent.baseline }; }
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
    const priorWeights = edges ? Float64Array.from(edges, i => ep.net.m[i]) : null;
    ep.step();
    if (edges) onLearningStep(event, edges.some((i, n) => ep.net.m[i] !== priorWeights[n]));
  }
  return ep;
}
function evaluate(seed, weights, learnedReadout) {
  const { net, agent } = fly(seed, false);
  net.m.set(weights);
  if (learnedReadout) {
    agent.W.forEach((w, a) => w.set(learnedReadout.W[a]));
    agent.b.set(learnedReadout.b);
    agent.baseline = learnedReadout.baseline;
  }
  agent.explore = false;
  const frozenWeights = net.m.slice(), frozenReadout = readout(agent), events = [];
  heldoutSeeds.forEach((sequenceSeed, i) => {
    const ep = new RpsEpisode(net, agent, opponent(sequenceSeed), { rounds, learn: false });
    runEpisode(ep, seed, 100 + i);
    events.push(...ep.events.map(e => ({ sequenceSeed, round: e.round,
      previous: e.observation.previousOpponent, opponent: e.opponent,
      action: e.action, reward: e.reward, rates: e.rates })));
  });
  const observed = events.filter(e => e.previous !== null);
  return { wins: events.filter(e => e.reward === 1).length,
    observedActionCounts: counts(observed.map(e => e.action)),
    byPrevious: [0, 1, 2].map(previous => {
      const rows = observed.filter(e => e.previous === previous);
      return { previous, trials: rows.length, wins: rows.filter(e => e.reward === 1).length,
        actionCounts: counts(rows.map(e => e.action)) };
    }), weightsUnchanged: net.m.every((x, i) => x === frozenWeights[i]),
    readoutUnchanged: same(readout(agent), frozenReadout), events };
}
function train(seed, readoutLearn) {
  const { net, agent } = fly(seed, readoutLearn), initial = net.m.slice();
  const curve = [], events = [], positiveRewardWeightChangingSteps = [0, 0, 0];
  trainSeeds.forEach((sequenceSeed, episodeIndex) => {
    const ep = new RpsEpisode(net, agent, opponent(sequenceSeed), { rounds, learn: true });
    runEpisode(ep, seed, episodeIndex, (event, changed) => {
      if (changed) positiveRewardWeightChangingSteps[event.action]++;
    });
    curve.push(ep.events.filter(e => e.reward === 1).length);
    events.push(...ep.events.map(e => ({ episode: episodeIndex, round: e.round,
      previous: e.observation.previousOpponent, opponent: e.opponent,
      action: e.action, reward: e.reward })));
  });
  const changedWeights = [];
  for (let i = 0; i < net.m.length; i++)
    if (net.m[i] !== initial[i]) changedWeights.push([i, net.m[i]]);
  const finalReadout = readout(agent);
  return { arm: readoutLearn ? 'hybrid' : 'hybrid_readout_disabled', curve,
    actionCounts: counts(events.map(e => e.action)), positiveRewardWeightChangingSteps,
    changedWeights, finalReadout, initialWeightsMatched: initial.every(x => x === 1),
    events, after: evaluate(seed, net.m, finalReadout) };
}
const students = seeds.map(seed => {
  const before = evaluate(seed, fly(seed, true).net.m, null);
  const hybrid = train(seed, true), disabled = train(seed, false);
  const source = prior.students.find(s => s.seed === seed);
  const ordinary = source.runs.find(r => r.arm === 'autonomous');
  const balanced = source.runs.find(r => r.arm === 'balanced');
  const expectedSchedule = ordinary.events.map(e => [e.previous, e.opponent]);
  const matchedSchedule = [hybrid, disabled].every(r => same(r.events.map(e => [e.previous, e.opponent]), expectedSchedule));
  const noReadoutChange = same(disabled.finalReadout,
    { W: [[0, 0, 0], [0, 0, 0], [0, 0, 0]], b: [0, 0, 0], baseline: 0 });
  const validity = { matchedSchedule, noReadoutChange,
    valid: matchedSchedule && noReadoutChange && [hybrid, disabled].every(r =>
      r.initialWeightsMatched && r.events.length === 288 && r.after.events.length === 72 &&
      r.after.byPrevious.reduce((n, x) => n + x.trials, 0) === 69 &&
      r.after.weightsUnchanged && r.after.readoutUnchanged) };
  const gates = { wins: hybrid.after.wins >= 58,
    eachState: hybrid.after.byPrevious.every(x => x.wins / x.trials >= 0.7),
    eachAction: hybrid.after.observedActionCounts.every(n => n >= 12),
    gainOverOrdinary: hybrid.after.wins - ordinary.after.wins >= 10,
    disabledControl: noReadoutChange && disabled.after.wins < 58 &&
      (!disabled.after.byPrevious.every(x => x.wins / x.trials >= 0.7) ||
       !disabled.after.observedActionCounts.every(n => n >= 12)) };
  return { seed, before, savedControls: { ordinaryWins: ordinary.after.wins,
    balancedWins: balanced.after.wins,
    ordinaryByPrevious: ordinary.after.byPrevious.map(x => x.wins),
    balancedByPrevious: balanced.after.byPrevious.map(x => x.wins),
    ordinaryActions: ordinary.after.observedActionCounts,
    balancedActions: balanced.after.observedActionCounts },
    runs: [hybrid, disabled], validity, gates,
    passed: validity.valid && Object.values(gates).every(Boolean) };
});
const result = { command: 'node tools/check_rps_reward_readout.mjs',
  design: { seeds, episodes, rounds, trainSeeds, heldoutSeeds,
    opponent: 'previous_opponent offset 1 noise 0',
    featureMean, featureStd, readout: 'three-action linear softmax REINFORCE lr .03 baselineRate .05',
    training: 'softmax sampled; original dopamine enabled in both hybrid arms',
    evaluation: 'deterministic argmax; no exploration or learning',
    savedControls: 'data/rps_balanced_training_corrected.json' },
  gates: { wins: '>=58/72 each', eachState: '>=70% each', eachAction: '>=12/69 each',
    gainOverOrdinary: '>=10/72 each', disabledControl: 'unchanged readout; <58/72 and full gate failed' },
  students, allValid: students.every(s => s.validity.valid),
  passed: students.every(s => s.passed) };
writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({ allValid: result.allValid, passed: result.passed,
  students: students.map(s => ({ seed: s.seed, before: s.before.wins,
    controls: s.savedControls, runs: s.runs.map(r => ({ arm: r.arm, wins: r.after.wins,
      byPrevious: r.after.byPrevious.map(x => x.wins), actions: r.after.observedActionCounts,
      rewardedChanges: r.positiveRewardWeightChangingSteps, kcMbonChanges: r.changedWeights.length,
      curve: r.curve, readout: r.finalReadout })), validity: s.validity, gates: s.gates })) }, null, 2));
