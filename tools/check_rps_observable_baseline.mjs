// Locked no-teacher RPS baseline. Run predictor first, then each student seed.
// node tools/check_rps_observable_baseline.mjs predictor
// node tools/check_rps_observable_baseline.mjs student 651
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { Opponent, RpsBioRL, RpsEpisode, result } from '../js/rps.js';

const [mode, seedText] = process.argv.slice(2);
const seeds = [651, 652, 653], episodes = 12, rounds = 24;
if (mode !== 'predictor' && (mode !== 'student' || !seeds.includes(Number(seedText))))
  throw Error('Use predictor or student 651|652|653');
const output = new URL(`../data/rps_observable_${mode === 'predictor' ? 'lookup' : `student_${seedText}`}.json`, import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite saved baseline');
const opponent = seed => new Opponent({ strategy: 'previous_opponent', offset: 1, noise: 0, seed });
const trainSeeds = Array.from({ length: episodes }, (_, i) => 41000 + i);
const heldoutSeeds = [51000, 51001, 51002];
const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;
function environment(seed) {
  const rival = opponent(seed), rows = [];
  for (let round = 0; round < rounds; round++) {
    const previous = rival.observe().previousOpponent;
    const { opponent: next } = rival.play(0); // This opponent ignores the player's action.
    rows.push({ previous, next });
  }
  return rows;
}
function predictFromAction(action) { return (action + 2) % 3; }
function count(xs) { return [0, 1, 2].map(a => xs.filter(x => x === a).length); }

if (mode === 'predictor') {
  const training = environment(trainSeeds[0]);
  const mostCommon = xs => count(xs).indexOf(Math.max(...count(xs)));
  const fallback = mostCommon(training.map(r => r.next));
  const lookup = [0, 1, 2].map(previous => {
    const sample = training.filter(r => r.previous === previous);
    if (!sample.length) throw Error(`Missing observable state ${previous}`);
    return mostCommon(sample.map(r => r.next));
  });
  const heldout = heldoutSeeds.flatMap(seed => environment(seed).map((row, round) => ({ seed, round, ...row })));
  const random = rng(617), randomActions = heldout.map(() => Math.floor(random() * 3));
  const constantAction = (fallback + 1) % 3;
  const summary = {
    predictionAccuracy: mean(heldout.map(r => Number((r.previous === null ? fallback : lookup[r.previous]) === r.next))),
    observedStateAccuracy: mean(heldout.filter(r => r.previous !== null).map(r => Number(lookup[r.previous] === r.next))),
    winRate: mean(heldout.map(r => Number(result(((r.previous === null ? fallback : lookup[r.previous]) + 1) % 3, r.next) === 1))),
    randomWinRate: mean(heldout.map((r, i) => Number(result(randomActions[i], r.next) === 1))),
    constantWinRate: mean(heldout.map(r => Number(result(constantAction, r.next) === 1)))
  };
  const passed = summary.predictionAccuracy >= .9 && summary.observedStateAccuracy === 1;
  const resultFile = { command: 'node tools/check_rps_observable_baseline.mjs predictor',
    strategy: { strategy: 'previous_opponent', offset: 1, noise: 0 }, trainSeeds, heldoutSeeds,
    trainingRounds: rounds, lookup, fallback, constantAction, training, heldout, summary, passed };
  writeFileSync(output, `${JSON.stringify(resultFile, null, 2)}\n`);
  console.log(JSON.stringify({ lookup, summary, passed }, null, 2));
  process.exit(0);
}

const lookupFile = JSON.parse(readFileSync(new URL('../data/rps_observable_lookup.json', import.meta.url)));
if (!lookupFile.passed) throw Error('Non-neural predictor gate failed; no student training allowed');
const sub = JSON.parse(readFileSync(new URL('../data/subgraph.json', import.meta.url)));
const studentSeed = Number(seedText);
function fly() {
  const net = new Network(sub, wiring(sub, 'real', studentSeed), { seed: studentSeed });
  return { net, agent: new RpsBioRL(net, { seed: studentSeed }) };
}
function evaluate(weights) {
  const { net, agent } = fly();
  net.m.set(weights);
  const frozen = net.m.slice(), events = [];
  for (const seed of heldoutSeeds) {
    const ep = new RpsEpisode(net, agent, opponent(seed), { rounds, learn: false }).run();
    events.push(...ep.events.map(e => ({ seed, round: e.round,
      previous: e.observation.previousOpponent, opponent: e.opponent,
      action: e.action, reward: e.reward, rates: e.rates })));
  }
  if (net.m.some((weight, i) => weight !== frozen[i])) throw Error('Frozen evaluation changed weights');
  const observed = events.filter(e => e.previous !== null);
  return { wins: events.filter(e => e.reward === 1).length,
    winRate: mean(events.map(e => Number(e.reward === 1))),
    impliedPredictionCorrect: events.filter(e => predictFromAction(e.action) === e.opponent).length,
    actionCounts: count(events.map(e => e.action)),
    observedActionCounts: count(observed.map(e => e.action)),
    byPrevious: [0, 1, 2].map(previous => {
      const sample = observed.filter(e => e.previous === previous);
      return { previous, trials: sample.length, wins: sample.filter(e => e.reward === 1).length,
        actionCounts: count(sample.map(e => e.action)) };
    }), events };
}

const { net, agent } = fly(), initialWeights = net.m.slice();
const before = evaluate(initialWeights), curve = [];
const training = { actionCounts: [0, 0, 0], rewardedTrials: [0, 0, 0],
  rewardedStepsWithChanges: [0, 0, 0], rewardedEdgeChanges: [0, 0, 0],
  rewardedAbsoluteWeightChange: [0, 0, 0] };
for (const seed of trainSeeds) {
  const ep = new RpsEpisode(net, agent, opponent(seed), { rounds });
  while (!ep.done) {
    if (ep.t % 400 !== 340) { ep.step(); continue; }
    const event = ep.events.at(-1), action = event.action;
    training.actionCounts[action]++;
    const edges = event.reward === 1 ? net.plastic[action] : null;
    const old = edges ? Float64Array.from(edges, i => net.m[i]) : null;
    ep.step(); // Existing dopamine update; measurements are external to the model.
    if (!edges) continue;
    training.rewardedTrials[action]++;
    let changed = 0, amount = 0;
    edges.forEach((index, i) => {
      const d = Math.abs(net.m[index] - old[i]);
      if (d) changed++;
      amount += d;
    });
    if (changed) training.rewardedStepsWithChanges[action]++;
    training.rewardedEdgeChanges[action] += changed;
    training.rewardedAbsoluteWeightChange[action] += amount;
  }
  curve.push(ep.winRate);
}
const after = evaluate(net.m);
const gain = after.winRate - before.winRate;
const gates = { finalWinRate: after.winRate >= .8, gain: gain >= .2,
  eachPreviousState: after.byPrevious.every(r => r.wins / r.trials >= .7),
  eachAction: after.observedActionCounts.every(n => n >= 12) };
const resultFile = { command: `node tools/check_rps_observable_baseline.mjs student ${studentSeed}`,
  studentSeed, strategy: lookupFile.strategy, trainSeeds, heldoutSeeds,
  episodes, rounds, noTeacherMessages: true, before, after, curve,
  training, gain, lookupSummary: lookupFile.summary, gates,
  passed: Object.values(gates).every(Boolean) };
writeFileSync(output, `${JSON.stringify(resultFile, null, 2)}\n`);
console.log(JSON.stringify({ studentSeed, before: before.winRate, after: after.winRate,
  lookup: lookupFile.summary, gain, observedActionCounts: after.observedActionCounts,
  byPrevious: after.byPrevious, training, gates, passed: resultFile.passed }, null, 2));
