// Check whether the locked Fly A source changes late even when the opponent never switches.
// Run: node tools/control_rps_teacher_noswitch.mjs SEED
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { Opponent, RpsBioRL, RpsEpisode } from '../js/rps.js';

const sub = JSON.parse(readFileSync(new URL('../data/subgraph.json', import.meta.url)));
const lock = JSON.parse(readFileSync(new URL('../data/rps_teacher_source_lock.json', import.meta.url)));
const seed = Number(process.argv[2]);
if (!Number.isInteger(seed) || lock.feature !== 'trace>=0.5')
  throw Error('usage: node tools/control_rps_teacher_noswitch.mjs SEED');
const output = new URL(`../data/rps_teacher_noswitch_${seed}.json`, import.meta.url);
if (existsSync(output)) throw Error(`Refusing to overwrite ${output}`);
const net = new Network(sub, wiring(sub, 'real', seed), { seed });
const agent = new RpsBioRL(net, { seed });
const episodeRates = [], sourceFrequency = [], sourceBits = [];
for (let ep = 0; ep < 12; ep++) {
  const opponent = new Opponent({ strategy: 'previous_opponent', offset: 1, noise: .1,
    seed: seed * 50000 + ep });
  new RpsEpisode(net, agent, opponent, { rounds: 24, learn: true }).run();
}
net.rand = rng(seed); agent.rand = rng(seed * 31 + 7);
for (let ep = 0; ep < 20; ep++) {
  const opponent = new Opponent({ strategy: 'previous_opponent', offset: 1, noise: .1,
    seed: seed * 1009 + ep });
  const run = new RpsEpisode(net, agent, opponent, { rounds: 24, learn: true });
  const bits = [];
  while (!run.done) {
    if (run.t % 400 === 120) bits.push(Number(net.trace[lock.neuron] >= .5));
    run.step();
  }
  sourceBits.push(...bits);
  sourceFrequency.push(bits.reduce((a, b) => a + b, 0) / bits.length);
  episodeRates.push(run.winRate);
  console.log(`no-switch seed ${seed} episode ${ep + 1}/20`);
}
writeFileSync(output, JSON.stringify({ seed, opponentOffset: 1, noise: .1,
  sourceNeuron: lock.neuron, sourceFeature: lock.feature, sourceFrequency,
  sourceBits, episodeRates }, null, 2));
console.log(`saved ${output.pathname}`);
