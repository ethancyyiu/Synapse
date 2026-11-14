// Fly B training with no page access, so it can run on a background thread.
import { Network, wiring } from './sim.js';
import { RpsContextualBandit } from './rps.js';
import { PreferenceStudent, comparisonRound, preferenceRound, studentWeights } from './rps_challenge_students.js';
import { CHECKPOINT_ROUNDS, finishFrozenDecision, frozenStudentCheckpoint } from './rps_challenge_checkpoints.js';
import { challengeOpponent } from './rps_challenge_opponent.js';

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
// Yields to incoming messages without timers, which browsers can throttle to once a second.
const tick = () => new Promise(resolve => {
  const channel = new MessageChannel();
  channel.port1.onmessage = () => { channel.port1.close(); resolve(); };
  channel.port2.postMessage(0);
});
const teacherSnapshot = agent => JSON.stringify({ W: agent.W.map(row => [...row]), b: [...agent.b], baseline: agent.baseline });

// Students rebuilt from saved weights only, enough for an isolated frozen check.
export const weightStub = json => ({ agent: JSON.parse(json) });

export const inlineCheckpoint = ({ sub, studentSeed, featureMean, featureStd, config, weights }) =>
  frozenStudentCheckpoint({ sub, students: weights.map(weightStub), studentSeed, featureMean, featureStd,
    makeOpponent: opponentSeed => challengeOpponent(config, opponentSeed) });

// rounds: training budget (24–288, whole 24-round episodes). preview: skip the in-between frozen checks
// and run only the final test. pauseAtSix: full 288-round run that waits after round 6 until continueAfterSix.
export async function* runStudents({ sub, seed, config, teacherState, featureMean, featureStd, control, runCheckpoint,
  rounds = 288, preview = false, previewEpisodes, pauseAtSix = false }) {
  if (preview && previewEpisodes) rounds = previewEpisodes * 24;
  if (pauseAtSix) rounds = 288;
  if (!Number.isInteger(rounds) || rounds < 24 || rounds > 288 || rounds % 24) throw Error('Fly B rounds must be 24–288 in steps of 24');
  const episodes = rounds / 24;
  const checkpointAt = preview ? [] : [...CHECKPOINT_ROUNDS.filter(n => n > 0 && n < rounds), rounds];
  const studentSeed = seed + 10000, makeNet = netSeed => new Network(sub, wiring(sub, 'real', netSeed), { seed: netSeed });
  const student = () => {
    const net = makeNet(studentSeed);
    return { net, agent: new PreferenceStudent(net, studentSeed, featureMean, featureStd), game: null };
  };
  const alone = student(), guided = student(), random = student();
  const teacherNet = makeNet(seed);
  const teacherAgent = new RpsContextualBandit(teacherNet, { seed, featureMean, featureStd, readoutLearn: false });
  teacherState.W.forEach((row, action) => teacherAgent.W[action].set(row));
  teacherAgent.b.set(teacherState.b);
  teacherAgent.baseline = teacherState.baseline;
  teacherAgent.explore = false;
  const teacherWeights = teacherSnapshot(teacherAgent);
  if (studentWeights(alone.agent) !== studentWeights(guided.agent) ||
      ![alone, guided].every(s => s.agent.heads.every(h => h.W.every(w => w.every(v => v === 0)) && h.b.every(v => v === 0))))
    throw Error('Student initial conditions are not matched and independent');
  alone.agent.explore = guided.agent.explore = random.agent.explore = true;
  const teacher = { net: teacherNet, agent: teacherAgent, key: 0 };

  const pending = [], finished = [];
  const startCheckpoint = rounds => {
    const check = other => Promise.resolve(runCheckpoint({ sub, studentSeed, featureMean, featureStd, config,
      weights: [studentWeights(alone.agent), studentWeights(other.agent)] }));
    const job = Promise.all([check(guided), check(random)]).then(([taught, control]) => {
      if (taught.wins[0] !== control.wins[0]) throw Error('Matched frozen checks disagreed');
      finished.push({ rounds, wins: [...taught.wins, control.wins[1]], byCue: taught.byCue,
        disconnected: taught.disconnected && control.disconnected,
        weightsUnchanged: taught.weightsUnchanged && control.weightsUnchanged });
    });
    pending.push(job);
  };
  const checkpoints = [];
  const drain = () => {
    const ready = finished.splice(0);
    checkpoints.push(...ready);
    return ready.map(point => ({ type: 'checkpoint', point }));
  };

  if (!preview) startCheckpoint(0);
  let completed = 0, lastFrame = 0, shownPhase = -1;
  const trainingPattern = config.mode === 'custom' ?
    [challengeOpponent(config, 41000), challengeOpponent(config, 41000), challengeOpponent(config, 41000)] : null;
  for (let episode = 0; episode < episodes; episode++) {
    const games = trainingPattern ?? [0, 1, 2].map(() => challengeOpponent(config, 41000 + episode));
    alone.game = games[0]; guided.game = games[1]; random.game = games[2];
    const episodeWins = [0, 0, 0];
    for (let r = 0; r < 24; r++) {
      teacher.key = seed * 10000000 + episode * 1000 + r;
      const round = preferenceRound(teacher, alone, guided, studentSeed * 10000000 + episode * 1000 + r, random);
      let time = 0, steps = 0;
      while (!round.done) {
        while (control.paused && !control.step) await sleep(50);
        round.step(); time++; steps++;
        const phase = time < 240 ? 0 : time < 280 ? 1 : time < 620 ? 2 : 3;
        if (phase !== shownPhase) { shownPhase = phase; yield { type: 'phase', phase }; }
        const now = performance.now();
        if ((time % 40 === 0 && now - lastFrame >= 66) || time === 280) {
          lastFrame = now;
          yield { type: 'frame', nets: [teacherNet, alone.net, guided.net].map(net => ({ t: net.t, lastSpike: net.lastSpike.slice() })) };
        }
        // Display pacing only: speed below 1 shows fewer simulation steps per 16 ms frame.
        const perFrame = control.speed < 1 ? Math.max(1, Math.round(10 * control.speed)) : control.slow ? 5 : 0;
        if (perFrame && steps >= perFrame) { steps = 0; await sleep(16); }
      }
      if (control.step) control.step = 0; // a single-round step ends here; pause holds the next round
      completed++;
      round.rows.forEach((row, i) => { episodeWins[i] += Number(row.reward === 1); });
      yield { type: 'round', completed, total: pauseAtSix && !control.continueAfterSix ? 6 : rounds, episodes, episode,
        rows: round.rows, message: round.message };
      if (checkpointAt.includes(completed)) startCheckpoint(completed);
      yield* drain();
      if (pauseAtSix && completed === 6) {
        yield { type: 'six_evaluating' };
        await Promise.all(pending);
        yield* drain();
        yield { type: 'six_ready' };
        while (!control.continueAfterSix) await sleep(50);
      }
      await tick(); // lets pause and slow-motion messages arrive
    }
    yield { type: 'episode', point: { episode: episode + 1, wins: episodeWins } };
  }
  if (teacherSnapshot(teacherAgent) !== teacherWeights ||
      ![alone, guided, random].every(s => s.net.m.every(value => value === 1)))
    throw Error('A frozen teacher or KC→MBON synapse changed during comparison');

  yield { type: 'final' };
  const trained = [studentWeights(alone.agent), studentWeights(guided.agent)];
  alone.agent.explore = guided.agent.explore = false;
  let totals, byCue;
  if (!preview) {
    // The final checkpoint already ran this exact disconnected 72-game test on
    // independent frozen copies. Reuse its verified scores instead of repeating it.
    await Promise.all(pending);
    yield* drain();
    const last = checkpoints.find(point => point.rounds === rounds);
    if (!last?.disconnected || !last.weightsUnchanged) throw Error('Final frozen checkpoint missing or invalid');
    totals = last.wins.slice(0, 2);
    byCue = last.byCue;
  } else {
    totals = [0, 0];
    byCue = [0, 1].map(() => [0, 1, 2].map(() => ({ wins: 0, n: 0 })));
    const finalPattern = config.mode === 'custom' ?
      [challengeOpponent(config, 51000), challengeOpponent(config, 51000)] : null;
    for (let episode = 0; episode < 3; episode++) {
      const games = finalPattern ?? [0, 1].map(() => challengeOpponent(config, 51000 + episode));
      alone.game = games[0]; guided.game = games[1];
      for (let r = 0; r < 24; r++) {
        const round = comparisonRound(null, alone, guided, studentSeed * 10000000 + (100 + episode) * 1000 + r, false);
        finishFrozenDecision(round);
        round.rows.forEach((row, i) => {
          totals[i] += Number(row.reward === 1);
          if (row.cue !== null) { byCue[i][row.cue].n++; byCue[i][row.cue].wins += Number(row.reward === 1); }
          if (row.decodedBit !== null || row.dnCounts.some(Boolean) || row.updateStep !== 0)
            throw Error('Frozen evaluation received advice or learned');
        });
        await tick();
      }
    }
  }
  if (studentWeights(alone.agent) !== trained[0] || studentWeights(guided.agent) !== trained[1] ||
      teacherSnapshot(teacherAgent) !== teacherWeights ||
      ![alone, guided].every(s => s.net.m.every(value => value === 1)))
    throw Error('Frozen evaluation changed a controller or neural synapse');
  yield { type: 'done', totals, byCue };
}
