// Held-out analysis of the original RPS rule and unchanged connectome learner.
// Run: node tools/diagnose_rps_plateau.mjs
import { readFileSync, writeFileSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { Opponent, RpsBioRL, RpsEpisode, result } from '../js/rps.js';

const sub = JSON.parse(readFileSync(new URL('../data/subgraph.json', import.meta.url)));
const OUTPUT = new URL('../data/rps_plateau_diagnostics.json', import.meta.url);
const SEEDS = [1, 2, 3], EPISODES = 40, ROUNDS = 24;
const OPPONENT = { strategy: 'previous_opponent', offset: 1, noise: 0.1 };
const mean = a => a.reduce((s, x) => s + x, 0) / a.length;

function fly(seed, variant = 'real') {
  const net = new Network(sub, wiring(sub, variant, seed), { seed });
  return { net, agent: new RpsBioRL(net, { seed }) };
}

function labelAction(moves) {
  const counts = [0, 0, 0];
  for (const move of moves) counts[move]++;
  return (counts.indexOf(Math.max(...counts)) + 1) % 3;
}

function policies(trainingEvents) {
  const overall = trainingEvents.map(e => e.opponent);
  const majority = labelAction(overall);
  const lookup = [0, 1, 2].map(obs =>
    labelAction(trainingEvents.filter(e => e.observation.previousOpponent === obs).map(e => e.opponent)));
  return { majority, lookup, first: majority };
}

function policyEvaluation(seed, policy) {
  const opponents = [], observations = [], randomActions = [], majorityActions = [], lookupActions = [];
  const rand = rng(seed * 137 + 17);
  for (let test = 0; test < 3; test++) {
    const rival = new Opponent({ ...OPPONENT, seed: seed * 100000 + 500 + test });
    for (let round = 0; round < ROUNDS; round++) {
      const obs = rival.observe().previousOpponent;
      const randomAction = Math.floor(rand() * 3);
      const majorityAction = policy.majority;
      const lookupAction = obs === null ? policy.first : policy.lookup[obs];
      const { opponent } = rival.play(randomAction);
      opponents.push(opponent); observations.push(obs);
      randomActions.push(randomAction); majorityActions.push(majorityAction); lookupActions.push(lookupAction);
    }
  }
  const rate = actions => mean(actions.map((a, i) => Number(result(a, opponents[i]) === 1)));
  return { random: rate(randomActions), majority: rate(majorityActions), lookup: rate(lookupActions),
    policy, observations, opponents };
}

function evaluate(source, seed, { noNoise = false, earlyProbe = false } = {}) {
  const evaluator = fly(seed + 500);
  evaluator.net.m.set(source.net.m);
  if (noNoise) evaluator.net.noise.fill(0);
  const before = evaluator.net.m.slice();
  const events = [], early = [];
  for (let test = 0; test < 3; test++) {
    const ep = new RpsEpisode(evaluator.net, evaluator.agent,
      new Opponent({ ...OPPONENT, seed: seed * 100000 + 500 + test }),
      { rounds: ROUNDS, learn: false });
    while (!ep.done) {
      if (earlyProbe && ep.t % 400 === 180) {
        const rates = evaluator.net.groupMbon.map(ids =>
          ids.reduce((sum, i) => sum + evaluator.net.count[i], 0) / ids.length);
        const best = Math.max(...rates);
        early.push({ rates, winners: [0, 1, 2].filter(a => rates[a] === best) });
      }
      ep.step();
    }
    events.push(...ep.events);
  }
  if (evaluator.net.m.some((x, i) => x !== before[i])) throw Error('Frozen evaluation changed weights');
  const byObservation = [null, 0, 1, 2].map(obs => {
    const sample = events.filter(e => e.observation.previousOpponent === obs);
    return { observation: obs, n: sample.length,
      winRate: mean(sample.map(e => Number(e.reward === 1))),
      actionCounts: [0, 1, 2].map(a => sample.filter(e => e.action === a).length),
      opponentCounts: [0, 1, 2].map(a => sample.filter(e => e.opponent === a).length),
      meanRates: [0, 1, 2].map(a => mean(sample.map(e => e.rates[a]))),
      zeroRateShare: mean(sample.map(e => Number(e.rates.every(x => x === 0)))) };
  });
  return { winRate: mean(events.map(e => Number(e.reward === 1))), byObservation,
    actions: events.map(e => e.action), rewards: events.map(e => e.reward),
    early: earlyProbe ? early.map((e, i) => ({ ...e,
      finalAction: events[i].action, opponent: events[i].opponent,
      wouldWinFirstTie: result(e.winners[0], events[i].opponent) === 1 })) : null };
}

function weightState(net) {
  const values = net.plastic.flatMap(ids => [...ids].map(i => net.m[i]));
  return { mean: mean(values), changed: values.filter(x => x !== 1).length,
    atZero: values.filter(x => x === 0).length, atMax: values.filter(x => x === 4).length };
}

function runSeed(seed) {
  const student = fly(seed + 500);
  const noLearning = evaluate(student, seed);
  const trainEvents = [], curve = [], weights = [];
  for (let episode = 0; episode < EPISODES; episode++) {
    const ep = new RpsEpisode(student.net, student.agent,
      new Opponent({ ...OPPONENT, seed: seed * 1009 + episode }), { rounds: ROUNDS }).run();
    trainEvents.push(...ep.events);
    curve.push(ep.winRate);
    if ([4, 9, 19, 39].includes(episode)) weights.push({ episode: episode + 1, ...weightState(student.net) });
  }
  const trained = evaluate(student, seed, { earlyProbe: true });
  const noNoise = evaluate(student, seed, { noNoise: true });
  const baseline = policyEvaluation(seed, policies(trainEvents));
  console.log(`seed ${seed}: before ${noLearning.winRate.toFixed(3)}, after40 ${trained.winRate.toFixed(3)}, ` +
    `noNoise ${noNoise.winRate.toFixed(3)}, lookup ${baseline.lookup.toFixed(3)}`);
  return { seed, curve, weights, noLearning, trained, noNoise, baseline };
}

let output;
try { output = JSON.parse(readFileSync(OUTPUT)); }
catch { output = { seeds: SEEDS, trainingEpisodes: EPISODES, roundsPerEpisode: ROUNDS,
  heldoutEpisodes: 3, opponent: OPPONENT, runs: [] }; }
for (const seed of SEEDS) {
  if (output.runs.some(r => r.seed === seed)) continue;
  output.runs.push(runSeed(seed));
  writeFileSync(OUTPUT, JSON.stringify(output, null, 2));
}
