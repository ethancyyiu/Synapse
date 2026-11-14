// Focused locked-source teaching experiment. Run: node tools/test_rps_pretrained_teacher.mjs SEED
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { Opponent, RpsBioRL, RpsEpisode } from '../js/rps.js';

const sub = JSON.parse(readFileSync(new URL('../data/subgraph.json', import.meta.url)));
const lock = JSON.parse(readFileSync(new URL('../data/rps_teacher_source_lock.json', import.meta.url)));
const seed = Number(process.argv[2]);
if (!Number.isInteger(seed) || seed < 100 || lock.neuron !== 2132 ||
    lock.feature !== 'trace>=0.5' || lock.sampleMs !== 120 ||
    lock.channel.strengthHz !== 300 || lock.channel.endMs !== 240)
  throw Error('Expected SEED >= 100 and the existing locked neuron/channel');
const output = new URL(`../data/rps_pretrained_teacher_${seed}.json`, import.meta.url);
if (existsSync(output)) throw Error(`Refusing to overwrite ${output}`);
const ROUNDS = 24, SWITCH = 10, POST = 10, PRETRAIN = 20, WIDTH = 8;

function fly(offset = 0) {
  const net = new Network(sub, wiring(sub, 'real', seed + offset), { seed: seed + offset });
  return { net, agent: new RpsBioRL(net, { seed: seed + offset }) };
}
function opponent(ep, offset) {
  return new Opponent({ strategy: 'previous_opponent', offset, noise: .1,
    seed: seed * 1009 + ep });
}
function pretrainOpponent(ep, offset, series) {
  return new Opponent({ strategy: 'previous_opponent', offset, noise: .1,
    seed: seed * 50000 + series * 1000 + ep });
}
class ReservedRpsEpisode extends RpsEpisode {
  startRound() {
    this.obs = this.opponent.observe();
    if (this.obs.previousOpponent !== null)
      this.net.setDrive(this.net.laneVpn[this.obs.previousOpponent].slice(WIDTH), 150);
  }
}
function trainTeacher(source, offset, series, episodes = PRETRAIN) {
  const winRates = [];
  for (let ep = 0; ep < episodes; ep++) {
    const run = new RpsEpisode(source.net, source.agent,
      pretrainOpponent(ep, offset, series), { rounds: ROUNDS, learn: true }).run();
    winRates.push(run.winRate);
  }
  return winRates;
}
function teacherStream(source, learns) {
  source.net.rand = rng(seed);
  source.agent.rand = rng(seed * 31 + 7);
  const before = source.net.m.slice();
  const rows = [], episodeWinRates = [];
  for (let ep = SWITCH; ep < SWITCH + POST; ep++) {
    const run = new RpsEpisode(source.net, source.agent, opponent(ep, 2),
      { rounds: ROUNDS, learn: learns });
    const samples = [];
    while (!run.done) {
      if (run.t % 400 === lock.sampleMs) {
        samples.push({ episode: ep, round: Math.floor(run.t / 400),
          previousOpponent: run.obs.previousOpponent,
          previousReward: run.obs.previousResult,
          sourceTrace: source.net.trace[lock.neuron],
          bit: Number(source.net.trace[lock.neuron] < .5),
          sampledAtMs: lock.sampleMs });
      }
      run.step();
    }
    samples.forEach((row, i) => {
      const event = run.events[i];
      Object.assign(row, { teacherAction: event.action, nextOpponent: event.opponent,
        correctAction: (event.opponent + 1) % 3, teacherReward: event.reward });
    });
    rows.push(...samples);
    episodeWinRates.push(run.winRate);
  }
  const weightsChanged = source.net.m.some((value, i) => value !== before[i]);
  if (!learns && weightsChanged) throw Error('Frozen teacher weights changed');
  return { rows, episodeWinRates, weightsChanged };
}
function shuffle(bits) {
  const values = [...bits], random = rng(seed * 7919 + 271);
  for (let i = values.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [values[i], values[j]] = [values[j], values[i]];
  }
  return values;
}
function weightsHash(net) {
  return createHash('sha256').update(Buffer.from(net.m.buffer)).digest('hex');
}
function episodeRun(student, ep, bits, learn = true) {
  const { net, agent } = student;
  const targets = [0, 1].map(lane => Array.from(net.laneVpn[lane].slice(0, WIDTH)));
  const run = new ReservedRpsEpisode(net, agent, opponent(ep, ep < SWITCH ? 1 : 2),
    { rounds: ROUNDS, learn });
  const measurements = [];
  while (!run.done) {
    const phase = run.t % 400, round = Math.floor(run.t / 400);
    if (phase === 120 && bits) net.setDrive(targets[bits[round]], 300);
    if (phase === 240) {
      const kcActive = [];
      let kcSpikes = 0;
      for (const i of net.idx.kc) {
        const count = net.count[i];
        if (count) { kcActive.push([i, count]); kcSpikes += count; }
      }
      measurements.push({ targetSpikes: targets.map(group =>
        group.reduce((sum, i) => sum + net.count[i], 0)), kcSpikes, kcActive });
      targets.forEach(group => net.setDrive(group, 0));
    }
    run.step();
  }
  return { winRate: run.winRate,
    events: run.events.map((event, i) => ({ episode: ep, round: i,
      previousOpponent: event.observation.previousOpponent,
      previousReward: event.observation.previousResult,
      action: event.action, nextOpponent: event.opponent, reward: event.reward,
      correctAction: (event.opponent + 1) % 3,
      rates: event.rates, bit: bits ? bits[i] : null, ...measurements[i] })) };
}
function frozenDisconnected(student) {
  const evaluator = fly(500);
  evaluator.net.m.set(student.net.m);
  const before = evaluator.net.m.slice();
  const winRates = [];
  for (let test = 0; test < 5; test++) {
    const rival = new Opponent({ strategy: 'previous_opponent', offset: 2,
      noise: .1, seed: seed * 100000 + 500 + test });
    const run = new ReservedRpsEpisode(evaluator.net, evaluator.agent, rival,
      { rounds: ROUNDS, learn: false }).run();
    winRates.push(run.winRate);
  }
  if (evaluator.net.m.some((value, i) => value !== before[i]))
    throw Error('Frozen student evaluation changed weights');
  return winRates;
}
function studentCondition(kind, streams) {
  const student = fly(500);
  const preEvents = [];
  for (let ep = 0; ep < SWITCH; ep++)
    preEvents.push(...episodeRun(student, ep, null).events.map(e =>
      [e.action, e.nextOpponent, e.reward]));
  const preHash = weightsHash(student.net);
  const bits = streams[kind] ?? null;
  const postEvents = [], episodeWinRates = [];
  for (let ep = SWITCH; ep < SWITCH + POST; ep++) {
    const slice = bits?.slice((ep - SWITCH) * ROUNDS, (ep - SWITCH + 1) * ROUNDS) ?? null;
    const run = episodeRun(student, ep, slice);
    postEvents.push(...run.events);
    episodeWinRates.push(run.winRate);
  }
  return { preEvents, preHash, postEvents, episodeWinRates,
    frozenDisconnected: frozenDisconnected(student) };
}

const pretrainedFly = fly(), simultaneousFly = fly(), untrainedFly = fly();
const pretrainedRates = trainTeacher(pretrainedFly, 2, 1);
const simultaneousPretrainRates = trainTeacher(simultaneousFly, 1, 2);
const simultaneousOldRates = [];
for (let ep = 0; ep < SWITCH; ep++) {
  const run = new RpsEpisode(simultaneousFly.net, simultaneousFly.agent,
    opponent(ep, 1), { rounds: ROUNDS, learn: true }).run();
  simultaneousOldRates.push(run.winRate);
}
const streams = {
  pretrained: teacherStream(pretrainedFly, false),
  simultaneous: teacherStream(simultaneousFly, true),
  untrained: teacherStream(untrainedFly, false),
};
for (let i = 0; i < POST * ROUNDS; i++) {
  const next = streams.pretrained.rows[i].nextOpponent;
  if (streams.simultaneous.rows[i].nextOpponent !== next ||
      streams.untrained.rows[i].nextOpponent !== next)
    throw Error(`Teacher opponent sequences differ at post-switch round ${i}`);
}
const bits = { pretrained: streams.pretrained.rows.map(row => row.bit),
  simultaneous: streams.simultaneous.rows.map(row => row.bit),
  untrained: streams.untrained.rows.map(row => row.bit) };
bits.shuffled = shuffle(bits.pretrained);
const conditions = {};
for (const kind of ['alone', 'pretrained', 'simultaneous', 'untrained', 'shuffled']) {
  conditions[kind] = studentCondition(kind, bits);
  if (kind !== 'alone' && (conditions[kind].preHash !== conditions.alone.preHash ||
      JSON.stringify(conditions[kind].preEvents) !== JSON.stringify(conditions.alone.preEvents)))
    throw Error(`Student pre-switch history differs for ${kind}`);
  console.log(`seed ${seed} ${kind}: first72=${conditions[kind].episodeWinRates
    .slice(0, 3).reduce((a, b) => a + b, 0) / 3}, frozen=${conditions[kind]
    .frozenDisconnected.reduce((a, b) => a + b, 0) / 5}`);
}
const result = { schema: 1, seed, lock, protocol: { roundsPerEpisode: ROUNDS,
  preSwitchEpisodes: SWITCH, postSwitchEpisodes: POST, teacherPretrainEpisodes: PRETRAIN,
  offsetBefore: 1, offsetAfter: 2, noise: .1, sampleMs: 120, decisionMs: 240,
  studentMessagesStartAtSwitch: true, frozenEvaluationEpisodes: 5 },
  teacherTraining: { pretrainedRates, simultaneousPretrainRates,
    simultaneousOldRates }, teacherStreams: streams, conditions, shuffledBits: bits.shuffled };
writeFileSync(output, JSON.stringify(result));
console.log(`saved ${output.pathname}`);
