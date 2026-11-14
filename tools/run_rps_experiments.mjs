// Reproducible RPS study using the browser's connectome simulator.
// Run: node tools/run_rps_experiments.mjs
import { readFileSync, writeFileSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { ActivityChannel, Opponent, RpsBioRL, RpsEpisode } from '../js/rps.js';

const sub = JSON.parse(readFileSync(new URL('../data/subgraph.json', import.meta.url)));
const SEEDS = [1, 2, 3], EPISODES = 12, ROUNDS = 24, THRESHOLD = 0.6;
const OPPONENT = { strategy: 'previous_opponent', offset: 1, noise: 0.1 };
const CONDITIONS = ['alone', 'trained', 'untrained', 'shuffled_signal', 'shuffled_wiring_alone'];
const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;
const sd = xs => Math.sqrt(mean(xs.map(x => (x - mean(xs)) ** 2)));
const oppSeed = (seed, episode) => seed * 1009 + episode;
const opponent = seed => new Opponent({ ...OPPONENT, seed });

function fly(seed, variant = 'real') {
  const net = new Network(sub, wiring(sub, variant, seed), { seed });
  return { net, agent: new RpsBioRL(net, { seed }) };
}
function shuffleSignals(signals, seed) {
  const r = rng(seed), out = signals.map(x => [...x]);
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}
function thresholdRounds(curve) {
  for (let i = 2; i < curve.length; i++) if (mean(curve.slice(i - 2, i + 1)) >= THRESHOLD) return (i + 1) * ROUNDS;
  return null;
}
function saveTeacher(net) {
  const edges = net.plastic.flatMap(ids => [...ids]);
  const bytes = Buffer.alloc(edges.length * 8);
  edges.forEach((k, i) => bytes.writeDoubleLE(net.m[k], i * 8));
  writeFileSync(new URL('../data/rps_teacher.bin', import.meta.url), bytes);
}

function collectSignals(source, seed, channel) {
  const signals = [];
  for (let ep = 0; ep < EPISODES; ep++) {
    signals.push(new RpsEpisode(source.net, source.agent, opponent(oppSeed(seed, ep)),
      { rounds: ROUNDS, learn: false, channel, captureSignals: true }).run().signals);
  }
  return signals;
}

function evaluate(source, seed) {
  return [0, 1, 2].map(i => new RpsEpisode(source.net, source.agent,
    opponent(seed * 100000 + 500 + i), { rounds: ROUNDS, learn: false }).run().winRate);
}

function runCondition(seed, condition, trainedSignals, untrainedSignals) {
  const variant = condition === 'shuffled_wiring_alone' ? 'shuffled' : 'real';
  const student = fly(seed + 500, variant);
  const channel = new ActivityChannel(sub, student.net);
  const curve = [];
  for (let ep = 0; ep < EPISODES; ep++) {
    let signals = null;
    if (condition === 'trained') signals = trainedSignals[ep];
    else if (condition === 'untrained') signals = untrainedSignals[ep];
    else if (condition === 'shuffled_signal') signals = shuffleSignals(trainedSignals[ep], seed * 7919 + ep);
    curve.push(new RpsEpisode(student.net, student.agent, opponent(oppSeed(seed, ep)),
      { rounds: ROUNDS, channel, incomingSignals: signals }).run().winRate);
  }
  const finalWeights = student.net.m.slice();
  const heldout = evaluate(student, seed);
  if (student.net.m.some((x, i) => x !== finalWeights[i])) throw Error('evaluation changed weights');
  return { curve, heldout, finalDisconnected: mean(heldout),
    roundsToThreshold: thresholdRounds(curve), transmitted: channel.log };
}

function runSeed(seed) {
  const teacher = fly(seed);
  const teacherCurve = [];
  for (let ep = 0; ep < EPISODES; ep++) {
    teacherCurve.push(new RpsEpisode(teacher.net, teacher.agent,
      opponent(oppSeed(seed, ep)), { rounds: ROUNDS }).run().winRate);
  }
  if (seed === 1) saveTeacher(teacher.net);

  // Match the trained and untrained source streams; only trained synapses differ.
  teacher.net.rand = rng(seed);
  teacher.agent.rand = rng(seed * 31 + 7);
  const channel = new ActivityChannel(sub, teacher.net);
  const trainedSignals = collectSignals(teacher, seed, channel);
  const untrainedSignals = collectSignals(fly(seed), seed, channel);
  const teacherHeldout = evaluate(teacher, seed);
  const runs = Object.fromEntries(CONDITIONS.map(condition =>
    [condition, runCondition(seed, condition, trainedSignals, untrainedSignals)]));
  return { seed, teacherCurve, teacherHeldout, runs };
}

function summarizeCondition(seeds, condition) {
  const runs = seeds.map(x => x.runs[condition]);
  return { runs, summary: {
    curveMean: Array.from({ length: EPISODES }, (_, ep) => mean(runs.map(r => r.curve[ep]))),
    curveSd: Array.from({ length: EPISODES }, (_, ep) => sd(runs.map(r => r.curve[ep]))),
    last3Mean: mean(runs.map(r => mean(r.curve.slice(-3)))),
    last3Sd: sd(runs.map(r => mean(r.curve.slice(-3)))),
    heldoutMean: mean(runs.map(r => r.finalDisconnected)),
    heldoutSd: sd(runs.map(r => r.finalDisconnected)),
    roundsToThreshold: runs.map(r => r.roundsToThreshold),
    reachedThreshold: runs.filter(r => r.roundsToThreshold !== null).length,
  } };
}

const started = Date.now(), seeds = [];
for (const seed of SEEDS) {
  seeds.push(runSeed(seed));
  console.log(`seed ${seed}/${SEEDS.length} complete in ${((Date.now() - started) / 1000).toFixed(1)}s`);
}
const conditions = Object.fromEntries(CONDITIONS.map(condition =>
  [condition, summarizeCondition(seeds, condition)]));
const result = { meta: { seeds: SEEDS, episodes: EPISODES, roundsPerEpisode: ROUNDS,
  threshold: THRESHOLD, opponent: OPPONENT, channel: { bandwidth: 3, source: new ActivityChannel(sub, fly(1).net).source,
    targetWidth: 2, sampleAtMs: 180, driveUntilMs: 240, levelsHz: [0, 75, 150] },
  randomWinRate: 1 / 3, note: 'Held-out evaluation has no neural channel and no learning.' },
  teacher: { runs: seeds.map(x => ({ seed: x.seed, curve: x.teacherCurve, heldout: x.teacherHeldout })),
    last3Mean: mean(seeds.map(x => mean(x.teacherCurve.slice(-3)))),
    heldoutMean: mean(seeds.map(x => mean(x.teacherHeldout))) }, conditions };
writeFileSync(new URL('../data/rps_results.json', import.meta.url), JSON.stringify(result));
for (const [name, c] of Object.entries(conditions)) console.log(name, c.summary);
