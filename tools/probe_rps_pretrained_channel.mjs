// Paired, frozen Fly B probe of both values observed in the real teacher stream.
// Run: node tools/probe_rps_pretrained_channel.mjs SEED [before|after]
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { Opponent, RpsBioRL, RpsEpisode } from '../js/rps.js';

const sub = JSON.parse(readFileSync(new URL('../data/subgraph.json', import.meta.url)));
const seed = Number(process.argv[2]);
const stage = process.argv[3] ?? 'before';
if (!Number.isInteger(seed) || !['before', 'after'].includes(stage))
  throw Error('usage: node tools/probe_rps_pretrained_channel.mjs SEED [before|after]');
const study = JSON.parse(readFileSync(new URL(`../data/rps_pretrained_teacher_${seed}.json`, import.meta.url)));
const suffix = stage === 'before' ? '' : '_after';
const output = new URL(`../data/rps_pretrained_channel_probe_${seed}${suffix}.json`, import.meta.url);
if (existsSync(output)) throw Error(`Refusing to overwrite ${output}`);
const lock = study.lock;
if (lock.neuron !== 2132 || lock.channel.strengthHz !== 300) throw Error('Source lock mismatch');

function fly() {
  const net = new Network(sub, wiring(sub, 'real', seed + 500), { seed: seed + 500 });
  return { net, agent: new RpsBioRL(net, { seed: seed + 500 }) };
}
class ReservedRpsEpisode extends RpsEpisode {
  startRound() {
    this.obs = this.opponent.observe();
    if (this.obs.previousOpponent !== null)
      this.net.setDrive(this.net.laneVpn[this.obs.previousOpponent].slice(8), 150);
  }
}
const student = fly();
for (let ep = 0; ep < 10; ep++) {
  const rival = new Opponent({ strategy: 'previous_opponent', offset: 1, noise: .1,
    seed: seed * 1009 + ep });
  new ReservedRpsEpisode(student.net, student.agent, rival,
    { rounds: 24, learn: true }).run();
}
const hash = createHash('sha256').update(Buffer.from(student.net.m.buffer)).digest('hex');
if (hash !== study.conditions.alone.preHash) throw Error('Probe student weights differ at switch');
if (stage === 'after') {
  const bits = study.teacherStreams.pretrained.rows.map(row => row.bit);
  const actual = [];
  for (let ep = 10; ep < 20; ep++) {
    const rival = new Opponent({ strategy: 'previous_opponent', offset: 2, noise: .1,
      seed: seed * 1009 + ep });
    const run = new ReservedRpsEpisode(student.net, student.agent, rival,
      { rounds: 24, learn: true });
    const targets = [0, 1].map(lane => Array.from(student.net.laneVpn[lane].slice(0, 8)));
    while (!run.done) {
      const phase = run.t % 400, round = Math.floor(run.t / 400);
      if (phase === 120) student.net.setDrive(targets[bits[(ep - 10) * 24 + round]], 300);
      if (phase === 240) targets.forEach(group => student.net.setDrive(group, 0));
      run.step();
    }
    actual.push(...run.events.map(event => [event.action, event.opponent, event.reward]));
  }
  const expected = study.conditions.pretrained.postEvents.map(event =>
    [event.action, event.nextOpponent, event.reward]);
  if (JSON.stringify(actual) !== JSON.stringify(expected))
    throw Error('Replayed pretrained student trajectory differs');
}

const records = [];
for (let rep = 0; rep < 24; rep++) {
  for (let previousOpponent = 0; previousOpponent < 3; previousOpponent++) {
    for (let bit = 0; bit < 2; bit++) {
      const { net, agent } = fly();
      net.m.set(student.net.m);
      net.rand = rng(seed * 100000 + rep);
      agent.rand = rng(seed * 200000 + rep);
      net.setDrive(net.laneVpn[previousOpponent].slice(8), 150);
      const targets = [0, 1].map(lane => Array.from(net.laneVpn[lane].slice(0, 8)));
      for (let t = 0; t < 240; t++) {
        if (t === 50) net.resetCounts();
        if (t === 120) net.setDrive(targets[bit], 300);
        net.step();
      }
      const kcActive = [];
      for (const i of net.idx.kc) if (net.count[i]) kcActive.push([i, net.count[i]]);
      const targetSpikes = targets.map(group => group.reduce((sum, i) => sum + net.count[i], 0));
      const action = agent.decide(net);
      records.push({ rep, previousOpponent, bit, targetSpikes,
        kcActive, mbonRates: agent.rates, action });
    }
  }
}
writeFileSync(output, JSON.stringify({ schema: 1, seed, stage, preSwitchStudentWeightsHash: hash,
  pairedRepetitionsPerPreviousMove: 24,
  note: `No reward or learning during probe; both actual channel bit values are tested on the same ${stage === 'before' ? 'pre-switch' : 'post-training'} Fly B weights and paired random seeds.`,
  records }));
console.log(`saved ${output.pathname}`);
