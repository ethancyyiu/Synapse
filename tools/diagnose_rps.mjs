// Read-only diagnostics for the existing connectome and RPS learning rule.
// Run: node tools/diagnose_rps.mjs current|hard|signal|positive
import { readFileSync, writeFileSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { ActivityChannel, Opponent, RpsBioRL, RpsEpisode, result, ROUND_MS, T_DECIDE } from '../js/rps.js';

const sub = JSON.parse(readFileSync(new URL('../data/subgraph.json', import.meta.url)));
const published = JSON.parse(readFileSync(new URL('../data/rps_results.json', import.meta.url)));
const SEEDS = [1, 2, 3];
const ROUNDS = 24;
const EPISODES = 20;
const CHECKPOINTS = [0, 1, 2, 3, 5, 10, 20];
const CURRENT = { strategy: 'previous_opponent', offset: 1, noise: 0.1 };
const LONG_SEQUENCE = [0, 1, 0, 2, 1, 2, 0, 2, 2, 1, 0, 1];
const mean = xs => xs.reduce((sum, x) => sum + x, 0) / xs.length;
const winRate = rewards => mean(rewards.map(x => Number(x === 1)));
const opponentSeed = (seed, episode) => seed * 1009 + episode;

function fly(seed) {
  const net = new Network(sub, wiring(sub, 'real', seed), { seed });
  return { net, agent: new RpsBioRL(net, { seed }) };
}

function opponent(kind, seed, episode, randomSeed = opponentSeed(seed, episode)) {
  if (kind === 'current') return new Opponent({ ...CURRENT, seed: randomSeed });
  if (kind === 'long_sequence') {
    const source = new Opponent({ strategy: 'sequence', sequence: LONG_SEQUENCE, noise: 0.1, seed: randomSeed });
    source.initial = Math.floor(rng(randomSeed + 17)() * LONG_SEQUENCE.length);
    return source;
  }
  if (kind === 'switch') {
    return new Opponent({ ...CURRENT, offset: episode < 10 ? 1 : 2, seed: randomSeed });
  }
  if (kind === 'probabilistic') {
    return new Opponent({ ...CURRENT, noise: 0.45, seed: randomSeed });
  }
  if (kind === 'history2' || kind === 'history3') {
    return new HistoryOpponent(kind === 'history2' ? 2 : 3, randomSeed);
  }
  throw Error(`unknown opponent: ${kind}`);
}

class HistoryOpponent {
  constructor(order, seed) {
    this.order = order;
    this.rand = rng(seed);
    this.history = Array.from({ length: order }, () => Math.floor(this.rand() * 3));
    this.previousOpponent = null;
    this.previousPlayer = null;
    this.previousResult = null;
  }
  observe() {
    return { previousOpponent: this.previousOpponent, previousPlayer: this.previousPlayer,
      previousResult: this.previousResult };
  }
  play(player) {
    const h = this.history;
    let next = this.order === 2
      ? (h.at(-1) + 2 * h.at(-2) + 1) % 3
      : (h.at(-1) + h.at(-2) + 2 * h.at(-3) + 1) % 3;
    if (this.rand() < 0.1) next = Math.floor(this.rand() * 3);
    const reward = result(player, next);
    h.push(next);
    this.previousOpponent = next;
    this.previousPlayer = player;
    this.previousResult = reward;
    return { opponent: next, reward };
  }
}

function runEpisode(source, rival, { signals = null, learn = true, details = false } = {}) {
  const channel = signals ? new ActivityChannel(sub, source.net) : null;
  const episode = new RpsEpisode(source.net, source.agent, rival,
    { rounds: ROUNDS, learn, channel, incomingSignals: signals }).run();
  const rewards = episode.events.map(e => e.reward);
  return details ? { rewards, moves: episode.events.map(e => e.opponent),
    previousMoves: episode.events.map(e => e.observation.previousOpponent) } : rewards;
}

function frozenEvaluation(source, seed, kind, checkpoint) {
  const evaluator = fly(seed + 500);
  evaluator.net.m.set(source.net.m);
  const rates = [];
  for (let test = 0; test < 3; test++) {
    const evalSeed = seed * 100000 + 500 + test;
    const rival = kind === 'switch'
      ? new Opponent({ ...CURRENT, offset: 2, seed: evalSeed })
      : opponent(kind, seed, checkpoint, evalSeed);
    rates.push(winRate(runEpisode(evaluator, rival, { learn: false })));
  }
  return rates;
}

function savedSignals(condition, seed, episode) {
  if (condition === 'alone') return null;
  return published.conditions[condition].runs[seed - 1].transmitted
    .slice(episode * ROUNDS, (episode + 1) * ROUNDS);
}

function currentTask() {
  const conditions = ['alone', 'trained', 'untrained', 'shuffled_signal'];
  const runs = [];
  for (const seed of SEEDS) {
    const byCondition = {};
    for (const condition of conditions) {
      const student = fly(seed + 500);
      const rewards = [];
      const episodeRates = [];
      const checkpoints = {};
      if (condition === 'alone') checkpoints[0] = frozenEvaluation(student, seed, 'current', 0);
      const count = condition === 'alone' ? EPISODES : 5;
      for (let episode = 0; episode < count; episode++) {
        const roundRewards = runEpisode(student, opponent('current', seed, episode),
          { signals: savedSignals(condition, seed, episode) });
        const rate = winRate(roundRewards);
        if (episode < 5 && Math.abs(rate - published.conditions[condition].runs[seed - 1].curve[episode]) > 1e-12)
          throw Error(`saved-result mismatch: ${condition}, seed ${seed}, episode ${episode + 1}`);
        rewards.push(...roundRewards);
        episodeRates.push(rate);
        if (condition === 'alone' && CHECKPOINTS.includes(episode + 1))
          checkpoints[episode + 1] = frozenEvaluation(student, seed, 'current', episode + 1);
      }
      byCondition[condition] = { rewards, episodeRates, checkpoints };
    }
    runs.push({ seed, conditions: byCondition });
    console.log(`current task: seed ${seed} complete`);
  }
  return { meta: { seeds: SEEDS, roundsPerEpisode: ROUNDS, episodesAlone: EPISODES,
    episodesWithSignals: 5, checkpoints: CHECKPOINTS,
    evaluation: '3 frozen, disconnected 24-round sequences per checkpoint' }, runs };
}

function stableRound(rewards) {
  // First 72-round trailing window at >=50% wins that stays there for 72 more rounds.
  const wins = rewards.map(x => Number(x === 1));
  for (let end = 72; end + 72 <= wins.length; end++) {
    let stable = true;
    for (let later = end; later <= end + 72; later++) {
      if (mean(wins.slice(later - 72, later)) < 0.5) { stable = false; break; }
    }
    if (stable) return end;
  }
  return null;
}

function harderTasks() {
  const kinds = ['long_sequence', 'history2', 'history3', 'switch', 'probabilistic'];
  const runs = {};
  for (const kind of kinds) {
    runs[kind] = [];
    for (const seed of SEEDS) {
      const student = fly(seed + 500);
      const rewards = [], episodeRates = [], moves = [], previousMoves = [];
      for (let episode = 0; episode < EPISODES; episode++) {
        const round = runEpisode(student, opponent(kind, seed, episode), { details: true });
        rewards.push(...round.rewards);
        moves.push(...round.moves);
        previousMoves.push(...round.previousMoves);
        episodeRates.push(winRate(round.rewards));
      }
      const heldout = frozenEvaluation(student, seed, kind, EPISODES);
      runs[kind].push({ seed, rewards, moves, previousMoves, episodeRates, heldout,
        stableRound: stableRound(rewards) });
      console.log(`${kind}: seed ${seed} complete`);
    }
  }
  return { meta: { seeds: SEEDS, episodes: EPISODES, roundsPerEpisode: ROUNDS,
    stability: 'first trailing 72-round window >=50% wins, sustained for 72 further windows',
    historyNoise: 0.1, sequence: LONG_SEQUENCE, probabilisticNoise: 0.45,
    switchEpisode: 11 }, runs };
}

function signalPath() {
  const runs = [];
  for (const seed of SEEDS) {
    const pair = [false, true].map(withSignal => {
      const student = fly(seed + 500);
      const signals = withSignal ? savedSignals('trained', seed, 0) : null;
      const channel = new ActivityChannel(sub, student.net);
      const episode = new RpsEpisode(student.net, student.agent, opponent('current', seed, 0),
        { rounds: ROUNDS, channel, incomingSignals: signals });
      const activity = [];
      while (!episode.done) {
        if (episode.t % ROUND_MS === T_DECIDE) {
          const target = channel.target.flat();
          activity.push({ targetSpikes: target.map(i => student.net.count[i]),
            mbonRates: student.net.groupMbon.map(ids =>
              ids.reduce((sum, i) => sum + student.net.count[i], 0) / ids.length),
            drive: target.map(i => student.net.drive[i]) });
        }
        episode.step();
      }
      return { activity, actions: episode.events.map(e => e.action),
        rewards: episode.events.map(e => e.reward),
        plasticWeights: student.net.plastic.flatMap(ids => [...ids].map(i => student.net.m[i])),
        transmitted: channel.log };
    });
    const [alone, signaled] = pair;
    const weightDifference = signaled.plasticWeights.reduce((sum, x, i) =>
      sum + Math.abs(x - alone.plasticWeights[i]), 0);
    runs.push({ seed, sourceSignals: savedSignals('trained', seed, 0),
      alone: { activity: alone.activity, actions: alone.actions, rewards: alone.rewards },
      signaled: { activity: signaled.activity, actions: signaled.actions, rewards: signaled.rewards,
        transmitted: signaled.transmitted },
      actionDifferences: signaled.actions.filter((x, i) => x !== alone.actions[i]).length,
      weightChangeAlone: alone.plasticWeights.reduce((sum, x) => sum + Math.abs(x - 1), 0),
      weightChangeSignaled: signaled.plasticWeights.reduce((sum, x) => sum + Math.abs(x - 1), 0),
      plasticWeightL1Difference: weightDifference });
    console.log(`signal path: seed ${seed} complete`);
  }
  return { meta: { seeds: SEEDS, episode: 1, rounds: ROUNDS,
    comparison: 'paired initial student seeds, same opponent sequences, signal versus no signal' }, runs };
}

function maskedOpponent(seed) {
  const underlying = new Opponent({ ...CURRENT, seed });
  return { observe: () => ({ ...underlying.observe(), previousOpponent: null }),
    play: action => underlying.play(action) };
}

function oracleSignals(seed) {
  const copy = new Opponent({ ...CURRENT, seed });
  return Array.from({ length: ROUNDS }, () => {
    const { opponent: nextMove } = copy.play(0);
    const signal = [0, 0, 0];
    signal[(nextMove + 1) % 3] = 2; // winning action, deliberately unavailable to a real teacher
    return signal;
  });
}

function positiveControl() {
  const runs = [];
  for (const seed of SEEDS) {
    const conditions = {};
    for (const condition of ['no_signal', 'oracle_signal']) {
      const student = fly(seed + 500);
      const rewards = [], episodeRates = [];
      for (let episode = 0; episode < EPISODES; episode++) {
        const randomSeed = opponentSeed(seed, episode);
        const signals = condition === 'oracle_signal' ? oracleSignals(randomSeed) : null;
        const roundRewards = runEpisode(student, maskedOpponent(randomSeed), { signals });
        rewards.push(...roundRewards);
        episodeRates.push(winRate(roundRewards));
      }
      const heldout = [];
      const evaluator = fly(seed + 500);
      evaluator.net.m.set(student.net.m);
      for (let test = 0; test < 3; test++) {
        const randomSeed = seed * 100000 + 500 + test;
        const signals = condition === 'oracle_signal' ? oracleSignals(randomSeed) : null;
        heldout.push(winRate(runEpisode(evaluator, maskedOpponent(randomSeed),
          { signals, learn: false })));
      }
      conditions[condition] = { rewards, episodeRates, heldout };
    }
    runs.push({ seed, conditions });
    console.log(`positive control: seed ${seed} complete`);
  }
  return { meta: { seeds: SEEDS, episodes: EPISODES, roundsPerEpisode: ROUNDS,
    warning: 'Oracle signal encodes the winning move before play; it is a separate channel-capacity control, not teacher communication. Normal previous-move observation is masked.' }, runs };
}

function channelProbe() {
  const runs = [];
  for (const seed of SEEDS) {
    const student = fly(seed + 500);
    const randomSeed = opponentSeed(seed, 0);
    const signals = oracleSignals(randomSeed);
    const channel = new ActivityChannel(sub, student.net);
    const episode = new RpsEpisode(student.net, student.agent, maskedOpponent(randomSeed),
      { rounds: ROUNDS, learn: false, channel, incomingSignals: signals });
    const targetSpikes = [];
    while (!episode.done) {
      if (episode.t % ROUND_MS === T_DECIDE)
        targetSpikes.push(channel.target.map(ids =>
          ids.reduce((sum, i) => sum + student.net.count[i], 0)));
      episode.step();
    }
    runs.push({ seed, rounds: episode.events.map((event, index) => ({
      oracleAction: signals[index].indexOf(2),
      chosenAction: event.action,
      reward: event.reward,
      mbonRates: event.rates,
      targetSpikes: targetSpikes[index],
    })) });
  }
  return { meta: { seeds: SEEDS, rounds: ROUNDS,
    description: 'Untrained frozen network; normal observation masked; oracle action is one-hot on channel.' }, runs };
}

function oracleWithObservation() {
  const runs = [];
  for (const seed of SEEDS) {
    const student = fly(seed + 500);
    const rewards = [], episodeRates = [];
    for (let episode = 0; episode < EPISODES; episode++) {
      const randomSeed = opponentSeed(seed, episode);
      const roundRewards = runEpisode(student, new Opponent({ ...CURRENT, seed: randomSeed }),
        { signals: oracleSignals(randomSeed) });
      rewards.push(...roundRewards);
      episodeRates.push(winRate(roundRewards));
    }
    const evaluator = fly(seed + 500);
    evaluator.net.m.set(student.net.m);
    const heldout = [];
    for (let test = 0; test < 3; test++) {
      const randomSeed = seed * 100000 + 500 + test;
      heldout.push(winRate(runEpisode(evaluator,
        new Opponent({ ...CURRENT, seed: randomSeed }),
        { signals: oracleSignals(randomSeed), learn: false })));
    }
    runs.push({ seed, rewards, episodeRates, heldout });
    console.log(`oracle with normal observation: seed ${seed} complete`);
  }
  return { meta: { seeds: SEEDS, episodes: EPISODES, roundsPerEpisode: ROUNDS,
    warning: 'The oracle signal exposes the actual winning action, including random deviations. It is an artificial positive control, not teacher communication.' }, runs };
}

function trainedTeacher(kind) {
  if (!['long_sequence', 'history2', 'history3', 'switch', 'probabilistic'].includes(kind))
    throw Error('Choose a harder opponent for the teacher run');
  const runs = [];
  for (const seed of SEEDS) {
    const teacher = fly(seed);
    const rewards = [], episodeRates = [];
    for (let episode = 0; episode < EPISODES; episode++) {
      const roundRewards = runEpisode(teacher, opponent(kind, seed, episode));
      rewards.push(...roundRewards);
      episodeRates.push(winRate(roundRewards));
    }
    const evaluator = fly(seed);
    evaluator.net.m.set(teacher.net.m);
    const heldout = [];
    for (let test = 0; test < 3; test++) {
      const evalSeed = seed * 100000 + 500 + test;
      const rival = kind === 'switch'
        ? new Opponent({ ...CURRENT, offset: 2, seed: evalSeed })
        : opponent(kind, seed, EPISODES, evalSeed);
      heldout.push(winRate(runEpisode(evaluator, rival, { learn: false })));
    }
    runs.push({ seed, rewards, episodeRates, heldout, stableRound: stableRound(rewards) });
    console.log(`teacher ${kind}: seed ${seed} complete`);
  }
  return { meta: { kind, seeds: SEEDS, episodes: EPISODES, roundsPerEpisode: ROUNDS,
    note: 'Teacher and student use identical architecture and task; only their random seeds differ.' }, runs };
}

const mode = process.argv[2];
const diagnostics = { current: currentTask, hard: harderTasks, signal: signalPath,
  positive: positiveControl, probe: channelProbe, oracle_observed: oracleWithObservation,
  teacher: () => trainedTeacher(process.argv[3]) };
if (!diagnostics[mode]) throw Error('Choose current, hard, signal, positive, probe, oracle_observed, or teacher');
const output = diagnostics[mode]();
const suffix = mode === 'teacher' ? `teacher_${process.argv[3]}` : mode;
const path = new URL(`../data/rps_diagnostic_${suffix}.json`, import.meta.url);
writeFileSync(path, JSON.stringify(output));
console.log(`saved ${path.pathname}`);
