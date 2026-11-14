// Record activity available to the RPS communication channel before each decision.
// Run: node tools/scan_rps_teacher_activity.mjs calibration|evaluation SEED
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { gzipSync } from 'node:zlib';
import { Network, rng, wiring } from '../js/sim.js';
import { Opponent, RpsBioRL, RpsEpisode } from '../js/rps.js';

const sub = JSON.parse(readFileSync(new URL('../data/subgraph.json', import.meta.url)));
const split = process.argv[2], seed = Number(process.argv[3]), variant = process.argv[4] ?? 'real';
if (!['calibration', 'evaluation'].includes(split) || !Number.isInteger(seed) ||
    !['real', 'shuffled'].includes(variant))
  throw Error('usage: node tools/scan_rps_teacher_activity.mjs calibration|evaluation SEED [real|shuffled]');
const suffix = variant === 'real' ? '' : `_${variant}`;
const output = new URL(`../data/rps_teacher_activity_${split}_${seed}${suffix}.json.gz`, import.meta.url);
if (existsSync(output)) throw Error(`Refusing to overwrite ${output}`);

const ROUNDS = 24, EPISODES = 20, PRETRAIN = 12, SWITCH = 10, SAMPLE_MS = 120;
const ids = sub.neurons.role.flatMap((role, id) => role === 'vnc' ? [] : [id]);
function fly() {
  const net = new Network(sub, wiring(sub, variant, seed), { seed });
  return { net, agent: new RpsBioRL(net, { seed }) };
}
function rival(ep, pretrain = false) {
  return new Opponent({ strategy: 'previous_opponent', offset: pretrain || ep < SWITCH ? 1 : 2,
    noise: .1, seed: pretrain ? seed * 50000 + ep : seed * 1009 + ep });
}
function sample(net) {
  const spikes = [], traces = [];
  for (let j = 0; j < ids.length; j++) {
    const i = ids[j], count = net.count[i], trace = net.trace[i];
    if (count) spikes.push([j, count]);
    if (trace >= .05) traces.push([j, Math.round(trace * 1000) / 1000]);
  }
  return { spikes, traces };
}
function stream(source, learn) {
  const samples = [], episodeRates = [];
  for (let ep = 0; ep < EPISODES; ep++) {
    const run = new RpsEpisode(source.net, source.agent, rival(ep), { rounds: ROUNDS, learn });
    const rows = [];
    while (!run.done) {
      if (run.t % 400 === SAMPLE_MS) {
        rows.push({ episode: ep, round: Math.floor(run.t / 400),
          previousOpponent: run.obs.previousOpponent, previousReward: run.obs.previousResult,
          ...sample(source.net) });
      }
      run.step();
    }
    for (let i = 0; i < rows.length; i++) {
      const event = run.events[i];
      Object.assign(rows[i], { action: event.action, nextOpponent: event.opponent,
        reward: event.reward, rates: event.rates });
    }
    samples.push(...rows);
    episodeRates.push(run.winRate);
    console.log(`${split} ${seed} ${learn ? 'trained' : 'untrained'} episode ${ep + 1}/${EPISODES}`);
  }
  return { episodeRates, samples };
}

const trained = fly(), untrained = fly();
const pretrainingRates = [];
for (let ep = 0; ep < PRETRAIN; ep++) {
  const run = new RpsEpisode(trained.net, trained.agent, rival(ep, true),
    { rounds: ROUNDS, learn: true }).run();
  pretrainingRates.push(run.winRate);
  console.log(`${split} ${seed} pretraining episode ${ep + 1}/${PRETRAIN}`);
}
for (const source of [trained, untrained]) {
  source.net.rand = rng(seed);
  source.agent.rand = rng(seed * 31 + 7);
}
const trainedRun = stream(trained, true), untrainedRun = stream(untrained, false);
for (let i = 0; i < trainedRun.samples.length; i++)
  if (trainedRun.samples[i].nextOpponent !== untrainedRun.samples[i].nextOpponent)
    throw Error(`Opponent histories differ at round ${i}`);
writeFileSync(output, gzipSync(JSON.stringify({ schema: 1, split, seed, variant,
  protocol: { roundsPerEpisode: ROUNDS, episodes: EPISODES, pretrainingEpisodes: PRETRAIN,
    switchEpisodeZeroBased: SWITCH, offsetBefore: 1, offsetAfter: 2, noise: .1,
    sampleMs: SAMPLE_MS, learningDuringRecording: true,
    countWindowStartsMs: 50, traceTimeConstantMs: 200 },
  neuronIds: ids, neuronRoles: ids.map(i => sub.neurons.role[i]),
  pretrainingRates, trained: trainedRun, untrained: untrainedRun })));
console.log(`saved ${output.pathname}`);
