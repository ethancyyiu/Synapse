// Stage 3: actual neural-activity teacher signals on the strategy-switch task.
// Run: node tools/test_rps_teaching_switch.mjs [pilot]
import { readFileSync, writeFileSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { MID_SOURCES, Opponent, RpsBioRL, RpsEpisode } from '../js/rps.js';

const sub = JSON.parse(readFileSync(new URL('../data/subgraph.json', import.meta.url)));
const PILOT = process.argv[2] === 'pilot';
const FINAL = ['final', 'postonly_final'].includes(process.argv[2]);
const OUTPUT = new URL(FINAL ? '../data/rps_teaching_switch_final.json'
  : '../data/rps_teaching_switch.json', import.meta.url);
const POST_ONLY_OUTPUT = new URL(FINAL ? '../data/rps_teaching_switch_postonly_final.json'
  : '../data/rps_teaching_switch_postonly.json', import.meta.url);
const SEEDS = PILOT ? [1] : FINAL ? [21, 22, 23] : [1, 2, 3], ROUNDS = PILOT ? 3 : 24;
const EPISODES = PILOT ? 2 : 20, PRETRAIN = PILOT ? 1 : 20;
const SWITCH_EPISODE = PILOT ? 1 : 10;
const SIGNAL_START = 120, DECIDE = 240, RESERVED_WIDTH = 8, SIGNAL_HZ = 300;
const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;

function fly(seed, variant = 'real') {
  const net = new Network(sub, wiring(sub, variant, seed), { seed });
  return { net, agent: new RpsBioRL(net, { seed }) };
}

function opponent(seed, ep, pretraining = false) {
  return new Opponent({ strategy: 'previous_opponent',
    offset: pretraining || ep >= SWITCH_EPISODE ? 2 : 1,
    noise: .1, seed: seed * 1009 + ep });
}

class ReservedRpsEpisode extends RpsEpisode {
  startRound() {
    this.obs = this.opponent.observe();
    if (this.obs.previousOpponent !== null)
      this.net.setDrive(this.net.laneVpn[this.obs.previousOpponent].slice(RESERVED_WIDTH), 150);
  }
}

function teacherStream(source, seed) {
  const episodes = [];
  for (let ep = 0; ep < EPISODES; ep++) {
    const run = new RpsEpisode(source.net, source.agent, opponent(seed, ep),
      { rounds: ROUNDS, learn: false });
    const samples = [];
    while (!run.done) {
      if (run.t % 400 === SIGNAL_START) {
        const raw = source.net.count[MID_SOURCES[0]];
        samples.push({ raw, label: raw > 0 ? 0 : 1,
          previousOpponent: run.obs.previousOpponent });
      }
      run.step();
    }
    samples.forEach((sample, i) => { sample.nextOpponent = run.events[i].opponent; });
    episodes.push(samples);
  }
  return episodes;
}

function shuffle(values, seed) {
  const out = [...values], rand = rng(seed);
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

function episodeRun(student, seed, ep, labels, learn = true) {
  const { net, agent } = student;
  const targets = [0, 1].map(lane => Array.from(net.laneVpn[lane].slice(0, RESERVED_WIDTH)));
  const run = new ReservedRpsEpisode(net, agent, opponent(seed, ep), { rounds: ROUNDS, learn });
  const measurements = [];
  while (!run.done) {
    const phase = run.t % 400, round = Math.floor(run.t / 400);
    if (phase === SIGNAL_START && labels)
      net.setDrive(targets[labels[round]], SIGNAL_HZ);
    if (phase === DECIDE) {
      measurements.push({ targetSpikes: targets.map(group =>
        group.reduce((sum, i) => sum + net.count[i], 0)),
        kcSpikes: net.idx.kc.reduce((sum, i) => sum + net.count[i], 0) });
      targets.forEach(group => net.setDrive(group, 0));
    }
    run.step();
  }
  return { events: run.events.map((event, i) => ({ observation: event.observation.previousOpponent,
    action: event.action, opponent: event.opponent, reward: event.reward, rates: event.rates,
    signal: labels ? labels[i] : null, ...measurements[i] })), winRate: run.winRate };
}

function frozenEvaluation(source, seed, variant) {
  const evaluator = fly(seed + 500, variant);
  evaluator.net.m.set(source.net.m);
  const before = evaluator.net.m.slice();
  const scores = [];
  for (let test = 0; test < 3; test++) {
    const rival = new Opponent({ strategy: 'previous_opponent', offset: 2, noise: .1,
      seed: seed * 100000 + 500 + test });
    const run = new ReservedRpsEpisode(evaluator.net, evaluator.agent, rival,
      { rounds: ROUNDS, learn: false }).run();
    scores.push(run.winRate);
  }
  if (evaluator.net.m.some((x, i) => x !== before[i])) throw Error('Evaluation changed student weights');
  return scores;
}

function recovery(rewards) {
  const start = SWITCH_EPISODE * ROUNDS;
  const wins = rewards.map(r => Number(r === 1));
  const window = PILOT ? 2 : 72;
  for (let end = start + window; end + window <= wins.length; end++) {
    let stable = true;
    for (let later = end; later <= end + window; later++) {
      if (mean(wins.slice(later - window, later)) < .5) { stable = false; break; }
    }
    if (stable) return end - start;
  }
  return null;
}

function runCondition(seed, kind, trained, untrained, postOnly = false) {
  const variant = kind === 'shuffled_wiring' ? 'shuffled' : 'real';
  const student = fly(seed + 500, variant);
  const episodeRates = [], events = [];
  for (let ep = 0; ep < EPISODES; ep++) {
    let labels = null;
    if (kind === 'trained' || kind === 'shuffled_wiring') labels = trained[ep].map(x => x.label);
    else if (kind === 'untrained') labels = untrained[ep].map(x => x.label);
    else if (kind === 'shuffled') labels = shuffle(trained[ep].map(x => x.label), seed * 7919 + ep);
    if (postOnly && ep < SWITCH_EPISODE) labels = null;
    const run = episodeRun(student, seed, ep, labels);
    episodeRates.push(run.winRate);
    events.push(...run.events);
  }
  const rewards = events.map(e => e.reward);
  return { episodeRates, events, recoveryRounds: recovery(rewards),
    postSwitchFirst3: mean(episodeRates.slice(SWITCH_EPISODE, SWITCH_EPISODE + 3)),
    postSwitchLast3: mean(episodeRates.slice(-3)),
    frozenDisconnected: frozenEvaluation(student, seed, variant) };
}

function runSeed(seed) {
  const trained = fly(seed), untrained = fly(seed);
  const pretraining = [];
  for (let ep = 0; ep < PRETRAIN; ep++) {
    const run = new RpsEpisode(trained.net, trained.agent,
      opponent(seed + 5000, ep, true), { rounds: ROUNDS }).run();
    pretraining.push(run.winRate);
  }
  // Match source randomness after pretraining; only learned synapses differ.
  for (const source of [trained, untrained]) {
    source.net.rand = rng(seed);
    source.agent.rand = rng(seed * 31 + 7);
  }
  const trainedStream = teacherStream(trained, seed);
  const untrainedStream = teacherStream(untrained, seed);
  const conditions = {};
  for (const condition of ['alone', 'trained', 'untrained', 'shuffled', 'shuffled_wiring']) {
    conditions[condition] = runCondition(seed, condition, trainedStream, untrainedStream);
    console.log(`seed ${seed} ${condition}: first3 ${conditions[condition].postSwitchFirst3.toFixed(3)}, ` +
      `recovery ${conditions[condition].recoveryRounds}, frozen ` +
      `${mean(conditions[condition].frozenDisconnected).toFixed(3)}`);
  }
  return { seed, teacherPretraining: pretraining, teacherSignals: trainedStream,
    untrainedSignals: untrainedStream, conditions };
}

if (['postonly', 'postonly_final'].includes(process.argv[2])) {
  const source = JSON.parse(readFileSync(OUTPUT));
  let output;
  try { output = JSON.parse(readFileSync(POST_ONLY_OUTPUT)); }
  catch { output = { seeds: SEEDS, episodes: EPISODES, roundsPerEpisode: ROUNDS,
    switchEpisode: SWITCH_EPISODE + 1, note: 'Same student history through switch; signals start after switch', runs: [] }; }
  for (const seed of SEEDS) {
    if (output.runs.some(r => r.seed === seed)) continue;
    const sourceRun = source.runs.find(r => r.seed === seed);
    const conditions = { alone: sourceRun.conditions.alone };
    for (const kind of ['trained', 'untrained', 'shuffled'])
      conditions[kind] = runCondition(seed, kind, sourceRun.teacherSignals,
        sourceRun.untrainedSignals, true);
    const first = conditions.trained.events.slice(0, SWITCH_EPISODE * ROUNDS)
      .map(e => [e.action, e.opponent, e.reward]);
    for (const kind of ['untrained', 'shuffled'])
      if (JSON.stringify(first) !== JSON.stringify(conditions[kind].events
        .slice(0, SWITCH_EPISODE * ROUNDS).map(e => [e.action, e.opponent, e.reward])))
        throw Error('Pre-switch student histories differed');
    output.runs.push({ seed, conditions });
    writeFileSync(POST_ONLY_OUTPUT, JSON.stringify(output, null, 2));
    console.log(`post-only seed ${seed}: ` + ['alone', 'trained', 'untrained', 'shuffled']
      .map(k => `${k} first3=${conditions[k].postSwitchFirst3.toFixed(3)}, ` +
        `recovery=${conditions[k].recoveryRounds}, ` +
        `frozen=${mean(conditions[k].frozenDisconnected).toFixed(3)}`).join('; '));
  }
} else if (PILOT) {
  const pilot = runSeed(1);
  console.log(`pilot complete: ${Object.keys(pilot.conditions).length} conditions`);
} else {
  let output;
  try { output = JSON.parse(readFileSync(OUTPUT)); }
  catch { output = { seeds: SEEDS, episodes: EPISODES, roundsPerEpisode: ROUNDS,
    switchEpisode: SWITCH_EPISODE + 1, pretrainingEpisodes: PRETRAIN,
    channel: { source: MID_SOURCES[0], encoding: 'source spike count at 120 ms > 0',
      targetLanes: [0, 1], targetWidth: RESERVED_WIDTH, hz: SIGNAL_HZ, startMs: SIGNAL_START,
      endMs: DECIDE, ordinaryObservationStartsAtLaneIndex: RESERVED_WIDTH }, runs: [] }; }
  for (const seed of SEEDS) {
    if (output.runs.some(r => r.seed === seed)) continue;
    output.runs.push(runSeed(seed));
    writeFileSync(OUTPUT, JSON.stringify(output, null, 2));
  }
}
