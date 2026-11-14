// Locked-source teaching test. Source selection is read from a calibration-only lock.
// Run after recording evaluation seeds: node tools/test_rps_selected_teacher_source.mjs full|postonly SEED
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { Network, rng, wiring } from '../js/sim.js';
import { Opponent, RpsBioRL, RpsEpisode } from '../js/rps.js';

const sub = JSON.parse(readFileSync(new URL('../data/subgraph.json', import.meta.url)));
const lock = JSON.parse(readFileSync(new URL('../data/rps_teacher_source_lock.json', import.meta.url)));
const mode = process.argv[2], seed = Number(process.argv[3]);
if (!['full', 'postonly'].includes(mode) || !Number.isInteger(seed))
  throw Error('usage: node tools/test_rps_selected_teacher_source.mjs full|postonly SEED');
if (lock.feature !== 'trace>=0.5' || lock.sampleMs !== 120)
  throw Error('This runner expects the locked raw trace threshold and 120 ms sample');
const output = new URL(`../data/rps_selected_teacher_${mode}_${seed}.json`, import.meta.url);
if (existsSync(output)) throw Error(`Refusing to overwrite ${output}`);
const teacher = JSON.parse(gunzipSync(readFileSync(new URL(
  `../data/rps_teacher_activity_evaluation_${seed}.json.gz`, import.meta.url))));
if (teacher.split !== 'evaluation' || lock.calibrationSeeds.includes(seed))
  throw Error('Teaching run must use independent evaluation seed');
const local = teacher.neuronIds.indexOf(lock.neuron);
if (local < 0 || teacher.neuronRoles[local] !== lock.role) throw Error('Source neuron missing');

const ROUNDS = 24, EPISODES = 20, SWITCH = 10, START = 120, DECIDE = 240;
const WIDTH = lock.channel.target === 'the first eight VPN neurons of label lane 0 or 1' ? 8 : NaN;
const HZ = lock.channel.strengthHz;
function fly(offset = 500) {
  const net = new Network(sub, wiring(sub, 'real', seed + offset), { seed: seed + offset });
  return { net, agent: new RpsBioRL(net, { seed: seed + offset }) };
}
function rival(ep, extra = 0) {
  return new Opponent({ strategy: 'previous_opponent', offset: ep < SWITCH ? 1 : 2,
    noise: .1, seed: extra ? seed * 100000 + extra : seed * 1009 + ep });
}
class ReservedRpsEpisode extends RpsEpisode {
  startRound() {
    this.obs = this.opponent.observe();
    if (this.obs.previousOpponent !== null)
      this.net.setDrive(this.net.laneVpn[this.obs.previousOpponent].slice(WIDTH), 150);
  }
}
function sourceLabels(condition) {
  return teacher[condition].samples.map(row => {
    const pair = row.traces.find(([j]) => j === local);
    return pair && pair[1] >= .5 ? 0 : 1;
  });
}
function shuffle(values) {
  const out = [...values], random = rng(seed * 7919 + 33);
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}
const trained = sourceLabels('trained');
const untrained = sourceLabels('untrained');
const shuffled = shuffle(trained);

function episodeRun(student, ep, labels) {
  const { net, agent } = student;
  const targets = [0, 1].map(lane => Array.from(net.laneVpn[lane].slice(0, WIDTH)));
  const run = new ReservedRpsEpisode(net, agent, rival(ep), { rounds: ROUNDS, learn: true });
  const measurements = [];
  while (!run.done) {
    const phase = run.t % 400, round = Math.floor(run.t / 400);
    if (phase === START && labels) net.setDrive(targets[labels[round]], HZ);
    if (phase === DECIDE) {
      measurements.push({ targetSpikes: targets.map(group => group.reduce((sum, i) => sum + net.count[i], 0)),
        kcSpikes: net.idx.kc.reduce((sum, i) => sum + net.count[i], 0) });
      targets.forEach(group => net.setDrive(group, 0));
    }
    run.step();
  }
  return { winRate: run.winRate, events: run.events.map((event, i) => ({
    episode: ep, round: i, observation: event.observation.previousOpponent,
    action: event.action, opponent: event.opponent, reward: event.reward,
    rates: event.rates, signal: labels ? labels[i] : null, ...measurements[i]
  })) };
}
function frozenDisconnected(student) {
  const evaluator = fly();
  evaluator.net.m.set(student.net.m);
  const before = evaluator.net.m.slice();
  const scores = [];
  for (let test = 0; test < 3; test++) {
    const run = new ReservedRpsEpisode(evaluator.net, evaluator.agent, rival(SWITCH, 500 + test),
      { rounds: ROUNDS, learn: false }).run();
    scores.push(run.winRate);
  }
  if (evaluator.net.m.some((x, i) => x !== before[i])) throw Error('Frozen evaluation changed weights');
  return scores;
}
function conditionRun(kind) {
  const student = fly();
  const stream = { trained, untrained, shuffled }[kind] ?? null;
  const rates = [], events = [];
  for (let ep = 0; ep < EPISODES; ep++) {
    const signal = stream && (mode === 'full' || ep >= SWITCH) ?
      stream.slice(ep * ROUNDS, (ep + 1) * ROUNDS) : null;
    const run = episodeRun(student, ep, signal);
    rates.push(run.winRate); events.push(...run.events);
  }
  const frozen = frozenDisconnected(student);
  const wins = events.map(event => Number(event.reward === 1));
  const first = SWITCH * ROUNDS;
  return { episodeWinRates: rates, events, frozenDisconnected: frozen,
    earlyPostSwitch24: wins.slice(first, first + 24).reduce((a, b) => a + b, 0) / 24,
    earlyPostSwitch72: wins.slice(first, first + 72).reduce((a, b) => a + b, 0) / 72 };
}
const conditions = {};
for (const kind of ['alone', 'trained', 'untrained', 'shuffled']) {
  conditions[kind] = conditionRun(kind);
  console.log(`${mode} seed ${seed} ${kind}: first24=${conditions[kind].earlyPostSwitch24.toFixed(3)} ` +
    `first72=${conditions[kind].earlyPostSwitch72.toFixed(3)} ` +
    `frozen=${(conditions[kind].frozenDisconnected.reduce((a, b) => a + b, 0) / 3).toFixed(3)}`);
}
if (mode === 'postonly') {
  const pre = JSON.stringify(conditions.alone.events.slice(0, SWITCH * ROUNDS)
    .map(e => [e.action, e.opponent, e.reward]));
  for (const kind of ['trained', 'untrained', 'shuffled'])
    if (pre !== JSON.stringify(conditions[kind].events.slice(0, SWITCH * ROUNDS)
      .map(e => [e.action, e.opponent, e.reward])))
      throw Error('Pre-switch student histories differed in postonly test');
}
writeFileSync(output, JSON.stringify({ schema: 1, mode, seed, lock, trainedLabels: trained,
  untrainedLabels: untrained, shuffledLabels: shuffled, conditions }, null, 2));
console.log(`saved ${output.pathname}`);
