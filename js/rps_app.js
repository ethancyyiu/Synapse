import { Network, wiring } from './sim.js';
import { ActivityChannel, MOVES, Opponent, RpsBioRL, RpsRewardReadout, RpsNonlinearReadout, RpsEpisode,
  ROUND_MS, T_DECIDE, result } from './rps.js';
import { Brain } from './brain.js';
import { drawChart, showResults } from './rps_results.js';

const $ = id => document.getElementById(id);
const fetchData = (url, type) => fetch(url).then(r => {
  if (!r.ok) throw Error(`${url}: ${r.status}`);
  return r[type]();
});
const colors = Object.fromEntries(['paper', 'ink', 'graphite', 'gfp', 'magenta'].map(k =>
  [k, getComputedStyle(document.documentElement).getPropertyValue('--' + k).trim()]));
const pct = x => `${(x * 100).toFixed(1)}%`;
const score = x => x > 0 ? 'win' : x < 0 ? 'loss' : 'tie';
const OPPONENT = { strategy: 'previous_opponent', offset: 1, noise: 0.1 };
const EPISODE_ROUNDS = 24;
const DEFAULT_TEACHER_EPISODES = 12;
const FRAME_BUDGET_MS = 13;
const PULSE_MS = 500;
let sub, points, teacherWeights, results, teacher, student, teacherBrain, studentBrain;
let teacherEp, studentEp, signals, channel, episodeNo = 0, testMode = false, playing = false;
let teacherCurve = [], studentCurve = [], carried = 0, lastTime = performance.now(), lastSignalAt = -1e9;
let originalStates, hybridStates, nonlinearStates, gameAuto = false, gameBusy = false, gameTarget = 0;
export let game;
let gameNextAt = 0;

// The two game flies use independent, archived student snapshots. No live game round trains either model.
class HumanOpponent {
  constructor() { this.previousOpponent = this.previousPlayer = this.previousResult = null; this.pending = null; }
  observe() { return { previousOpponent: this.previousOpponent,
    previousPlayer: this.previousPlayer, previousResult: this.previousResult }; }
  play(flyMove) {
    if (!Number.isInteger(this.pending)) throw Error('Choose a move before the fly decides.');
    const opponent = this.pending, reward = result(flyMove, opponent);
    this.previousOpponent = opponent; this.previousPlayer = flyMove; this.previousResult = reward;
    this.pending = null;
    return { opponent, reward };
  }
}
function savedRun(data, seed, arm) {
  const run = data?.students?.find(x => x.seed === seed)?.runs?.find(x => x.arm === arm);
  if (!run || !Array.isArray(run.changedWeights)) throw Error(`Missing saved ${arm} state for seed ${seed}`);
  return run;
}
const matrix = (value, rows, cols) => Array.isArray(value) && value.length === rows &&
  value.every(row => Array.isArray(row) && row.length === cols && row.every(Number.isFinite));
const vector = (value, length) => Array.isArray(value) && value.length === length && value.every(Number.isFinite);
export function gameFly(seed, model) {
  const source = model === 'bio' ? originalStates : model === 'hybrid' ? hybridStates :
    model === 'nonlinear' ? nonlinearStates : null;
  const arm = { bio: 'autonomous', hybrid: 'hybrid', nonlinear: 'nonlinear_frozen' }[model];
  if (!source || !arm) throw Error(`Missing saved ${model} model`);
  if (model === 'nonlinear' && (!source.allValid || !source.design?.seeds?.includes(seed)))
    throw Error(`Incompatible saved nonlinear student ${seed}`);
  const net = new Network(sub, wiring(sub, 'real', seed), { seed });
  const run = savedRun(source, seed, arm);
  if (model === 'nonlinear' && (run.changedWeights.length || !run.after?.scorePathMatched ||
    !run.after?.weightsUnchanged || !run.after?.readoutUnchanged))
    throw Error(`Incompatible saved nonlinear neural state ${seed}`);
  const plastic = new Set(net.plastic.flatMap(ids => [...ids]));
  const seen = new Set();
  for (const [edge, multiplier] of run.changedWeights) {
    if (!plastic.has(edge) || seen.has(edge) || !Number.isFinite(multiplier) || multiplier < 0)
      throw Error(`Incompatible saved ${model} KC-to-MBON weights`);
    net.m[edge] = multiplier; seen.add(edge);
  }
  let agent;
  if (model === 'hybrid' || model === 'nonlinear') {
    const { featureMean, featureStd } = source.design;
    const { W, b, baseline, W1, b1 } = model === 'nonlinear' ? run.finalController ?? {} : run.finalReadout ?? {};
    const width = model === 'nonlinear' ? 8 : 3;
    if (!vector(featureMean, 3) || !vector(featureStd, 3) || featureStd.some(x => x <= 0) ||
      !matrix(W, 3, width) || !vector(b, 3) || !Number.isFinite(baseline) ||
      (model === 'nonlinear' && (!matrix(W1, 8, 3) || !vector(b1, 8) ||
        JSON.stringify(featureMean) !== JSON.stringify(hybridStates?.design?.featureMean) ||
        JSON.stringify(featureStd) !== JSON.stringify(hybridStates?.design?.featureStd))))
      throw Error(`Incompatible saved ${model} readout`);
    const Readout = model === 'nonlinear' ? RpsNonlinearReadout : RpsRewardReadout;
    agent = new Readout(net, { seed, featureMean, featureStd,
      freezeSynapses: model === 'nonlinear', readoutLearn: false });
    agent.W.forEach((row, a) => row.set(W[a]));
    agent.b.set(b); agent.baseline = baseline; agent.explore = false;
    if (model === 'nonlinear') {
      agent.W1.forEach((row, k) => row.set(W1[k]));
      agent.b1.set(b1);
    }
  } else agent = new RpsBioRL(net, { seed });
  return { net, agent };
}
function resetGame() {
  gameAuto = gameBusy = false;
  $('game-watch').textContent = 'Watch both flies play';
  const opponentMode = $('game-opponent').value;
  game = ['a', 'b'].map((id, i) => {
    const model = $(i ? 'game-model-b' : 'game-model').value;
    const seed = model === 'nonlinear' ? Number($(`game-seed-${id}`).value) : 651 + i;
    const fly = gameFly(seed, model);
    fly.model = model; fly.seed = seed;
    const opponent = opponentMode === 'human' ? new HumanOpponent() :
      new Opponent({ strategy: 'previous_opponent', offset: 1, noise: 0, seed: 62000 });
    fly.ep = new RpsEpisode(fly.net, fly.agent, opponent, { rounds: 1000, learn: false });
    fly.brain = new Brain($(`game-${id}-brain`), $(`game-${id}-header`), new Uint16Array(points), sub, colors);
    fly.kc = 0;
    return fly;
  });
  $('game-human').hidden = opponentMode !== 'human';
  $('game-pattern').hidden = opponentMode !== 'pattern';
  $('game-seed-a-wrap').hidden = $('game-model').value !== 'nonlinear';
  $('game-seed-b-wrap').hidden = $('game-model-b').value !== 'nonlinear';
  $('game-model-note').textContent = game.some(fly => fly.model === 'nonlinear')
    ? 'Experimental nonlinear RL controller: real simulated MBON spike rates feed a small additional neural network trained from RPS rewards. KC→MBON weights were frozen during its training. Two saved students passed individually, but the locked all-student gate failed; this is not evidence that the original dopamine-only fly learned reliable three-action RPS. Live play is frozen.'
    : game.some(fly => fly.model === 'hybrid')
    ? 'Experimental linear hybrid: saved dopamine-updated synapses plus a reward-trained MBON readout. Its three-action reliability gate failed. Live play is frozen.'
    : 'Original dopamine-only MBON policy, with saved KC-to-MBON weights from the autonomous RPS experiment. Live play is frozen.';
  $('game-status').textContent = opponentMode === 'human' ? 'Choose a move to play both flies.' : 'Watch or step through the patterned opponent.';
  $('game-history').replaceChildren();
  for (const id of ['a', 'b']) {
    const fly = game[id === 'a' ? 0 : 1];
    $(`game-${id}-model`).textContent = `${fly.model === 'bio' ? 'Original dopamine-only MBON' :
      fly.model === 'hybrid' ? 'Experimental linear hybrid reward-trained readout' :
      'Experimental nonlinear RL controller'} · saved student ${fly.seed} · frozen`;
    for (const key of ['move', 'opponent', 'kc', 'mbon-0', 'mbon-1', 'mbon-2']) $(`game-${id}-${key}`).textContent = '--';
    $(`game-${id}-score`).textContent = '0 wins / 0 rounds';
    $(`game-${id}-outcome`).textContent = 'waiting';
  }
}
function loadGame() {
  game = null;
  try { resetGame(); }
  catch (err) { $('game-status').textContent = `Selected saved model unavailable: ${err.message}`; }
}
function startGameRound(humanMove = null) {
  if (!game || gameBusy || game[0].ep.done) return;
  if ($('game-opponent').value === 'human') {
    if (!Number.isInteger(humanMove) || humanMove < 0 || humanMove > 2) return;
    for (const fly of game) fly.ep.opponent.pending = humanMove;
  }
  gameTarget = (game[0].ep.events.length + 1) * ROUND_MS;
  gameBusy = true;
  $('game-status').textContent = 'Simulating the neural decision...';
}
function showGameRound() {
  game.forEach((fly, i) => {
    const id = i ? 'b' : 'a', ev = fly.ep.events.at(-1);
    if (!ev) return;
    $(`game-${id}-move`).textContent = MOVES[ev.action];
    $(`game-${id}-opponent`).textContent = MOVES[ev.opponent];
    $(`game-${id}-outcome`).textContent = score(ev.reward);
    $(`game-${id}-score`).textContent = `${fly.ep.events.filter(x => x.reward === 1).length} wins / ${fly.ep.events.length} rounds`;
    $(`game-${id}-kc`).textContent = String(fly.kc);
    ev.rates.forEach((rate, a) => { $(`game-${id}-mbon-${a}`).textContent = rate.toFixed(2); });
    const row = document.createElement('tr');
    for (const value of [ev.round + 1, `Fly ${id.toUpperCase()}`, MOVES[ev.action], MOVES[ev.opponent], score(ev.reward)]) {
      const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
    }
    $('game-history').prepend(row);
  });
  $('game-status').textContent = `Round ${game[0].ep.events.length} complete. Both decisions came from the live simulator.`;
}
function advanceGameFrame(now) {
  if (!game) return;
  if (gameBusy) {
    const deadline = performance.now() + FRAME_BUDGET_MS;
    while (game[0].ep.t < gameTarget && performance.now() < deadline) {
      for (const fly of game) {
        if (fly.ep.t % ROUND_MS === T_DECIDE)
          fly.kc = fly.net.idx.kc.reduce((n, k) => n + (fly.net.count[k] > 0), 0);
        fly.ep.step();
      }
    }
    if (game[0].ep.t >= gameTarget) {
      gameBusy = false; showGameRound(); gameNextAt = now + 500;
    }
  } else if (gameAuto && now >= gameNextAt) startGameRound();
  for (const fly of game) fly.brain.frame(fly.net, fly.ep);
}

function makeFly(seed) {
  const net = new Network(sub, wiring(sub, 'real', seed), { seed });
  return { net, agent: new RpsBioRL(net, { seed }) };
}
function restoreTeacher() {
  teacher = makeFly(1);
  const edges = teacher.net.plastic.flatMap(ids => [...ids]);
  if (!teacherWeights || teacherWeights.byteLength !== edges.length * 8)
    throw Error('Missing or incompatible data/rps_teacher.bin. Run node tools/run_rps_experiments.mjs.');
  const values = new DataView(teacherWeights);
  edges.forEach((k, i) => { teacher.net.m[k] = values.getFloat64(i * 8, true); });
}
function newEpisode() {
  const seed = 1009 + episodeNo;
  signals = Array.from({ length: EPISODE_ROUNDS }, () => [0, 0, 0]);
  channel = new ActivityChannel(sub, student.net);
  studentEp = new RpsEpisode(student.net, student.agent, new Opponent({ ...OPPONENT, seed }),
    { rounds: EPISODE_ROUNDS, learn: !testMode, channel: testMode ? null : channel,
      incomingSignals: testMode ? null : signals });
  if (!testMode) teacherEp = new RpsEpisode(teacher.net, teacher.agent,
    new Opponent({ ...OPPONENT, seed }), { rounds: EPISODE_ROUNDS, learn: false,
      channel, captureSignals: true });
  $('status').textContent = testMode ? 'Teacher disconnected. Student playing alone with learning frozen.'
    : `Training episode ${episodeNo + 1}: both flies playing; Fly A is frozen after pretraining.`;
}
function resetStudent() {
  student = makeFly(501);
  studentCurve = [];
  teacherCurve = [];
  episodeNo = 0;
  testMode = false;
  playing = false;
  $('disconnect').textContent = 'Disconnect teacher & test student';
  $('play').textContent = 'Start experimental simulation';
  $('student-training').textContent = '0';
  $('student-rate').textContent = '—';
  $('signal-values').textContent = 'No signal yet';
  newEpisode();
  drawChart($('live-chart'), [], DEFAULT_TEACHER_EPISODES, colors);
}
function updateEvents(ep, prefix) {
  const ev = ep.events.at(-1);
  if (!ev) return;
  $(`${prefix}-move`).textContent = MOVES[ev.action];
  $(`${prefix}-opponent`).textContent = MOVES[ev.opponent];
  $(`${prefix}-score`).textContent = `${ep.events.filter(x => x.reward === 1).length}/${ep.events.length} wins · last: ${score(ev.reward)}`;
}
function advanceStep() {
  if (!testMode) {
    const count = teacherEp.signals.length;
    teacherEp.step();
    if (teacherEp.signals.length > count) {
      const round = teacherEp.signals.length - 1;
      signals[round] = teacherEp.signals[round];
    }
  }
  const count = channel.log.length;
  studentEp.step();
  if (!testMode && channel.log.length > count) {
    const signal = channel.log.at(-1);
    $('signal-values').textContent = `${signal.join(' · ')} spikes (quantized)`;
    lastSignalAt = performance.now();
  }
  if (studentEp.done) {
    if (testMode) {
      playing = false;
      $('play').textContent = 'Resume';
      $('status').textContent = `Independent test complete: ${pct(studentEp.winRate)} wins over ${EPISODE_ROUNDS} unseen rounds. Learning stayed frozen.`;
    } else {
      studentCurve.push(studentEp.winRate);
      teacherCurve.push(teacherEp.winRate);
      episodeNo++;
      $('student-training').textContent = String(episodeNo);
      drawChart($('live-chart'), [
        { values: teacherCurve, color: colors.gfp },
        { values: studentCurve, color: colors.ink },
      ], Math.max(DEFAULT_TEACHER_EPISODES, episodeNo), colors);
      newEpisode();
    }
  }
}
function frame(now) {
  const elapsed = Math.min(100, now - lastTime);
  lastTime = now;
  advanceGameFrame(now);
  if (playing && studentEp && !studentEp.done) {
    let steps;
    if ($('speed').value === 'max') steps = Infinity;
    else {
      carried += elapsed * Number($('speed').value);
      steps = Math.floor(carried);
      carried -= steps;
    }
    const deadline = performance.now() + FRAME_BUDGET_MS;
    while (steps-- > 0 && performance.now() < deadline && !studentEp.done) advanceStep();
  }
  if (teacherEp) {
    teacherBrain.frame(teacher.net, teacherEp);
    updateEvents(teacherEp, 'teacher');
  }
  if (studentEp) {
    studentBrain.frame(student.net, studentEp);
    updateEvents(studentEp, 'student');
    $('student-rate').textContent = pct(studentEp.winRate);
  }
  const age = performance.now() - lastSignalAt, pulse = $('pulse');
  pulse.classList.toggle('active', !testMode && age < PULSE_MS && channel?.log.at(-1)?.some(x => x > 0));
  pulse.style.left = `${Math.min(96, Math.max(0, age / PULSE_MS * 96))}%`;
  requestAnimationFrame(frame);
}

try {
  [sub, points, teacherWeights, results, originalStates, hybridStates, nonlinearStates] = await Promise.all([
    fetchData('data/subgraph.json', 'json'), fetchData('data/brain_points.bin', 'arrayBuffer'),
    fetchData('data/rps_teacher.bin', 'arrayBuffer'),
    fetchData('data/rps_results.json', 'json').catch(() => null),
    fetchData('data/rps_balanced_training_corrected.json', 'json').catch(() => null),
    fetchData('data/rps_reward_readout.json', 'json').catch(() => null),
    fetchData('data/rps_nonlinear_readout.json', 'json').catch(() => null),
  ]);
  restoreTeacher();
  const pointArray = new Uint16Array(points);
  teacherBrain = new Brain($('teacher-brain'), $('teacher-header'), pointArray, sub, colors);
  studentBrain = new Brain($('student-brain'), $('student-header'), pointArray, sub, colors);
  $('teacher-training').textContent = `${results?.meta.episodes ?? DEFAULT_TEACHER_EPISODES} completed`;
  showResults(results, colors); resetStudent();
  loadGame();
  $('game-reset').onclick = loadGame;
  for (const id of ['game-opponent', 'game-model', 'game-model-b', 'game-seed-a', 'game-seed-b'])
    $(id).onchange = loadGame;
  for (const button of $('game-human').querySelectorAll('[data-move]'))
    button.onclick = () => startGameRound(Number(button.dataset.move));
  $('game-next').onclick = () => startGameRound();
  $('game-watch').onclick = () => {
    gameAuto = !gameAuto;
    $('game-watch').textContent = gameAuto ? 'Pause watching' : 'Watch both flies play';
    if (gameAuto && !gameBusy) startGameRound();
  };
  $('play').onclick = () => { playing = !playing; $('play').textContent = playing ? 'Pause' : 'Resume experimental simulation'; };
  $('reset').onclick = resetStudent;
  $('disconnect').onclick = () => {
    testMode = !testMode;
    $('disconnect').textContent = testMode ? 'Reconnect & continue training' : 'Disconnect teacher & test student';
    $('signal-values').textContent = testMode ? 'Disconnected' : 'Waiting for next round';
    playing = true; $('play').textContent = 'Pause';
    newEpisode();
  };
  requestAnimationFrame(frame);
} catch (err) {
  $('status').textContent = `Could not start: ${err.message}. Serve this folder with python tools/serve.py.`;
  console.error(err);
}
