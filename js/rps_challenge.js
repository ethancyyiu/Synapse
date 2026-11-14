// Live Fly A and matched Fly B Challenge; the research replay below remains archived and separate.
import { Network, rng, wiring } from './sim.js';
import { MOVES, Opponent, RpsContextualBandit, RpsEpisode, T_DECIDE, result } from './rps.js';
import { inlineCheckpoint, runStudents } from './rps_student_engine.js';
import { analyzePattern, challengeOpponent } from './rps_challenge_opponent.js';
export { analyzePattern, challengeOpponent };
import { Brain } from './brain.js';
import { Brain3D } from './brain3d.js';

const $ = id => document.getElementById(id);
const arms = ['correct', 'shuffled', 'untrained', 'generic', 'disconnected'];
const labels = { correct: 'Trained Fly A', shuffled: 'Shuffled advice', untrained: 'Untrained teacher',
  generic: 'Generic veto', disconnected: 'No advice' };
const data = async (path, type = 'json') => {
  const response = await fetch(path);
  if (!response.ok) throw Error(`${path}: ${response.status}`);
  return response[type]();
};
async function brainCoordinates3D() {
  try {
    const [background, neurons] = await Promise.all([
      data('data/brain_points_3d.bin', 'arrayBuffer'),
      data('data/sim_neuron_positions_3d.bin', 'arrayBuffer'),
    ]);
    return { background: new Float32Array(background), neurons: new Float32Array(neurons) };
  } catch { return null; }
}
const brainViews = new Map();
const hiddenRoles = new Set(['vnc', 'mid']); // background activity starts hidden; the legend restores it
const brainNames = { 'challenge-teacher': 'Fly A', guided: 'Fly B With Teacher', alone: 'Fly B Alone' };
// One playback state for every run on the page (recorded replay, live Fly A, live Fly B).
// It only paces the display; it never changes what the simulation computes.
const playback = { active: false, paused: false, speed: 1, step: 0, where: '' };
let speedButtons = [];
// Display-only helpers for watching a run: follow the active brains, and show the final test on the scoreboard.
let followKey = '', followOff = false, missionFinal = false;
let replayNav = null, pendingSeek = null, timelineDragging = false, seeking = false; // recorded-run timeline
let evaluating = false; // a frozen 72-game test is running: header shows a moving bar
function setEvaluating(on) { evaluating = on; renderPlayback(); }
function followTo(key, el) {
  if (seeking || followOff || followKey === key || !el?.getBoundingClientRect || typeof scrollTo !== 'function' || $('tab-sim')?.hidden) return;
  followKey = key;
  const top = el.getBoundingClientRect().top + scrollY - ($('mission').offsetHeight || 0) - 12;
  scrollTo({ top: Math.max(0, top), behavior: 'smooth' });
}
// Solid hand icons (fist, open hand, V sign) in the page's accent colour; .cut lines use the box background.
const ICON_PATHS = [
  // Rock: closed fist with four knuckles and a thumb across the front.
  '<rect x="5" y="5.2" width="3.4" height="7" rx="1.7"/><rect x="8.3" y="4.4" width="3.4" height="7.6" rx="1.7"/>' +
  '<rect x="11.6" y="4.6" width="3.4" height="7.4" rx="1.7"/><rect x="14.9" y="5.6" width="3.2" height="6.6" rx="1.6"/>' +
  '<rect x="5" y="9" width="13.4" height="10.4" rx="3.6"/><rect x="8.2" y="19" width="7.6" height="3.6" rx="1"/>' +
  '<path class="cut" d="M8.4 5.6v5M11.7 5.2v5.2M15 5.6v4.8"/>' +
  '<path class="cut" d="M5 12.1H13.7a1.85 1.85 0 0 1 0 3.7H5"/>',
  // Paper: open hand, four fingers and a thumb.
  '<rect x="5.6" y="5" width="2.6" height="10" rx="1.3"/><rect x="8.7" y="2.6" width="2.6" height="11" rx="1.3"/>' +
  '<rect x="11.8" y="2.2" width="2.6" height="11.5" rx="1.3"/><rect x="14.9" y="3.6" width="2.6" height="10.4" rx="1.3"/>' +
  '<rect x="16.9" y="8.6" width="2.6" height="7.4" rx="1.3" transform="rotate(38 18.2 12.3)"/>' +
  '<rect x="5.6" y="10.8" width="12.4" height="9" rx="3.6"/><rect x="8" y="19.2" width="7.8" height="3.4" rx="1"/>',
  // Scissors: V sign, two fingers up, the rest curled under the thumb.
  '<rect x="7.4" y="1.8" width="2.8" height="11.5" rx="1.4" transform="rotate(-13 8.8 12)"/>' +
  '<rect x="12.9" y="1.8" width="2.8" height="11.5" rx="1.4" transform="rotate(13 14.3 12)"/>' +
  '<rect x="5.6" y="10.6" width="12.6" height="9.2" rx="3.6"/><rect x="8.2" y="19.2" width="7.6" height="3.4" rx="1"/>' +
  '<path class="cut" d="M8.6 11.4v2.4"/>' +
  '<path class="cut" d="M5.6 14H13.6a1.85 1.85 0 0 1 0 3.7H5.6"/>',
];
const rpsIcon = action => `<svg class="rps-icon" viewBox="0 0 24 24" aria-hidden="true">${ICON_PATHS[action]}</svg>`;
const outcomeCode = reward => reward > 0 ? 'w' : reward < 0 ? 'l' : 'd';
function renderStrip(id, codes) {
  $(id).replaceChildren(...codes.map(code => { const cell = document.createElement('span'); cell.className = code; return cell; }));
}
function showTeachMessage(message, guidedRow) {
  $('stage-card').hidden = true;
  $('teach-message').hidden = false;
  if (message.preference === null) {
    $('teach-move').textContent = 'No advice yet';
  } else {
    $('teach-move').innerHTML = `${rpsIcon(message.preference)}${move(message.preference)}`;
  }
  $('teach-dn-counts').textContent = `3141: ${message.dnCounts[0]} · 3165: ${message.dnCounts[1]}`;
  $('teach-detail').textContent = `Decoded: ${message.decoded === null ? 'none' : move(message.decoded)}`;
  $('teach-update').textContent = `${guidedRow.reward > 0 ? 'Win' : guidedRow.reward < 0 ? 'Loss' : 'Draw'} · played ${move(guidedRow.action)}`;
  const step = Number(guidedRow.updateStep);
  $('teach-update-detail').textContent = `${Number.isFinite(step) ? `Own-reward step ${step >= 0 ? '+' : ''}${step.toFixed(3)}` : 'Own-reward step unavailable'} · ${guidedRow.imitation ? `advice update toward decoded ${move(message.decoded)}` : 'no advice update'}`;
  $('teach-message').className = $('teach-message').className === 'teach-message pulse-a' ? 'teach-message pulse-b' : 'teach-message pulse-a';
}
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const blocked = () => playback.paused && !playback.step;
function renderPlayback() {
  const { active, paused, speed, where } = playback;
  $('play-toggle').disabled = $('play-step').disabled = !active;
  $('play-toggle').textContent = paused ? '▶ Resume' : '❚❚ Pause';
  $('play-toggle').className = paused && active ? 'paused' : '';
  $('mission').className = [paused && active && 'is-paused', active && 'is-running', missionFinal && 'is-final',
    replayNav && 'has-timeline', evaluating && 'is-evaluating']
    .filter(Boolean).join(' ');
  $('challenge-panel').className = active ? 'panel is-running' : 'panel';
  speedButtons.forEach(button => button.setAttribute('aria-checked', String(Number(button.dataset.speed) === speed)));
  $('play-status').textContent = !active ? (replayNav ? 'Finished · drag the timeline or click a step to rewatch' : 'Start a run to use these') :
    playback.step ? `Stepping one round…` :
    paused ? `Paused${where ? ` · ${where}` : ''} · → next round` : `Playing at ${({ 0.25: '¼', 0.5: '½' })[speed] ?? speed}×`;
}
function setPlayback(update) {
  if (update.active && !playback.active) { followKey = ''; followOff = false; missionFinal = false; }
  Object.assign(playback, update);
  const message = Object.fromEntries(['paused', 'speed', 'step'].filter(key => key in update).map(key => [key, update[key]]));
  if (Object.keys(message).length) activeRun?.control(message);
  renderPlayback();
}
// Called whenever one round has fully played; a pending single-round step ends here.
function roundDone(where) {
  playback.where = where;
  if (playback.step) playback.step = 0;
  renderPlayback();
}
// Waits ms of playing time at the current speed; holds while paused; exits if the run is reset.
async function hold(ms, token) {
  let left = ms;
  while (token === challengeToken && pendingSeek === null) {
    if (blocked()) { await sleep(50); continue; }
    if (left <= 0) return;
    const started = performance.now();
    await sleep(Math.min(left, 100) / playback.speed);
    left -= (performance.now() - started) * playback.speed;
  }
}
const PICK_HINT = 'Click a dot to identify a neuron.';

const STORY_IDLE = 'Watch the recorded rule change, or use Step 1 to train your own flies.';
function mission(step, title, progress = 0) {
  [1, 2, 3, 4].forEach(i => { $(`mission-step-${i}`).className = i === step ? 'step now' : i < step ? 'step done' : 'step'; });
  $('mission-title').textContent = title;
  $('mission-bar').style.width = `${Math.round(progress * 100)}%`;
  if (playback.active) {
    if (step === 2) followTo('teacher', $('teacher-card'));
    else if (step === 3) followTo('students', $('teach-head') ?? $('brain-stage'));
    else if (step === 5) followTo('end', $('step4-title'));
  }
}
function showPhase(active) {
  $('teach-arrow').className = active === 1 ? 'teach-arrow on' : 'teach-arrow';
  [0, 1, 2, 3].forEach(i => { $(`live-phase-${i}`).className = i === active ? 'phase on' : 'phase'; });
}
const brain = (prefix, coordinates, fallbackPoints, sub) => {
  if (brainViews.has(prefix)) return brainViews.get(prefix);
  const canvas = $(`${prefix}-brain`), header = $(`${prefix}-brain-header`);
  const colors = Object.fromEntries(['paper', 'ink', 'gfp', 'magenta'].map(name =>
    [name, getComputedStyle(document.documentElement).getPropertyValue(`--${name}`).trim()]));
  let view;
  if (canvas.ownerDocument && coordinates) {
    try { view = new Brain3D(canvas, header, coordinates.background, coordinates.neurons, sub, colors); }
    catch { view = new Brain(canvas, header, fallbackPoints, sub, colors); }
  } else view = new Brain(canvas, header, fallbackPoints, sub, colors);
  view.setHiddenRoles?.(hiddenRoles);
  view.onPick = pick => {
    if (!pick) { $('brain-pick').textContent = PICK_HINT; return; }
    const { index, type, role, activity } = pick;
    $('brain-pick').textContent = `Yellow ring: ${brainNames[prefix]} · neuron ${index} · type ${type} · class ${role.toUpperCase()} · ${activity}`;
  };
  brainViews.set(prefix, view);
  return view;
};
let anatomyBackground = null;
function clearBrain(prefix, label) {
  const view = brainViews.get(prefix);
  if (view instanceof Brain3D) {
    view.clear(label);
    return;
  }
  if (!view) { $(`${prefix}-brain-header`).textContent = label; return; } // a 2D context here would block WebGL later
  const canvas = $(`${prefix}-brain`);
  if (anatomyBackground && $('brain-stage')) {
    canvas.height = anatomyBackground.height;
    canvas.getContext('2d').drawImage(anatomyBackground, 0, 0);
  } else canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
  $(`${prefix}-brain-header`).textContent = label;
}
async function loadBrainAnatomy() {
  const loadToken = challengeToken;
  try {
    const [sub, bytes, coordinates] = await Promise.all([
      data('data/subgraph.json'), data('data/brain_points.bin', 'arrayBuffer'),
      brainCoordinates3D(),
    ]);
    const canvas = document.createElement('canvas'); canvas.width = 320;
    const colors = Object.fromEntries(['paper', 'ink', 'graphite', 'gfp', 'magenta'].map(name =>
      [name, getComputedStyle(document.documentElement).getPropertyValue(`--${name}`).trim()]));
    anatomyBackground = new Brain(canvas, document.createElement('span'),
      new Uint16Array(bytes), sub, colors).bg;
    if (loadToken === challengeToken) {
      const points = new Uint16Array(bytes);
      brain('challenge-teacher', coordinates, points, sub);
      brain('guided', coordinates, points, sub);
      brain('alone', coordinates, points, sub);
      clearBrain('challenge-teacher', 'Connectome anatomy only · live spikes begin with Fly A training.');
      clearBrain('guided', 'Connectome anatomy only · live spikes begin with student training.');
      clearBrain('alone', 'Connectome anatomy only · live spikes begin with student training.');
      $('brain-scene-status').textContent = brainViews.size === 3 &&
        [...brainViews.values()].every(view => view instanceof Brain3D) ?
        'True 3D positions' :
        'WebGL unavailable · showing the static 2D anatomy fallback.';
    }
  } catch (error) {
    for (const prefix of ['challenge-teacher', 'guided', 'alone'])
      $(`${prefix}-brain-header`).textContent = `Connectome anatomy unavailable: ${error.message}`;
  }
}
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const yes = (condition, message) => { if (!condition) throw Error(`Archived replay invalid: ${message}`); };

export function validateReplay(archive, source) {
  yes(archive.protocol?.teacher === 830 && source.protocol?.teacherSeed === 830,
    'teacher identity');
  yes(source.teacher?.source?.training?.length === 24, 'continuous teacher source');
  yes(archive.students?.length === 2 && same(archive.students.map(s => s.seed), [851, 852]),
    'independent student identities');
  yes(archive.passed === false && archive.status !== 'passed', 'failed gate must remain failed');
  for (const student of archive.students) {
    yes(student.initial?.rows?.length === 72 && student.initial.weightsUnchanged === true &&
      student.initial.rows.every(row => row.emittedBit === null && row.update?.step === 0),
    'initial frozen disconnected checkpoint');
    const trainingSchedule = student.training.correct.rows.map(row => [row.cue, row.opponent]);
    const frozenSchedule = student.middle.correct.rows.map(row =>
      [row.sequenceSeed, row.round, row.cue, row.opponent]);
    for (const arm of arms) {
      const training = student.training?.[arm]?.rows;
      yes(training?.length === 12, `${student.seed} ${arm} training rows`);
      yes(same(training.map(row => [row.cue, row.opponent]), trainingSchedule),
        'matched opponent training schedule');
      training.forEach((row, round) => {
        const teacher = source.teacher.source.training[round];
        yes(row.round === round && row.cue === teacher.cue, 'teacher and student cue schedule');
        yes(row.reward === result(row.action, row.opponent), 'actual RPS reward');
        yes(row.update?.selectedAction === row.action && row.update?.selectedHead === row.selectedHead &&
          Math.abs(row.update.step - 0.03 * (row.reward - row.scores[row.action])) < 1e-10,
        'own-reward selected-action update');
        if (row.cue === null) {
          yes(row.teacherAdvice === null && row.emittedBit === null, 'cue-free first round');
        } else {
          const expectedAdvice = Number(teacher.scores[row.candidate] >
            Math.max(...teacher.scores.filter((_, action) => action !== row.candidate)));
          if (arm === 'correct') yes(row.teacherAdvice === expectedAdvice &&
            row.emittedBit === expectedAdvice, 'teacher Q-derived approval');
          else yes(row.teacherAdvice === null, 'control has no recorded trained-teacher advice');
          if (arm === 'untrained') yes(row.emittedBit === 0, 'untrained bit');
          if (arm === 'disconnected') yes(row.emittedBit === null, 'disconnected bit');
        }
        if (row.emittedBit === null) yes(same(row.dnCounts, [0, 0]) && row.decodedBit === null,
          'disconnected DN counts');
        else yes(row.decodedBit === row.emittedBit &&
          row.dnCounts[row.emittedBit] > row.dnCounts[1 - row.emittedBit], 'measured DN bit');
        if (row.decodedBit === 1 || row.decodedBit === null)
          yes(row.action === row.candidate, 'approved or disconnected proposal');
        else yes(row.action !== row.candidate, 'rejected proposal');
      });
      for (const checkpoint of [student.middle?.[arm], student.final?.[arm]]) {
        yes(checkpoint?.rows?.length === 72 && checkpoint.weightsUnchanged === true,
          'frozen disconnected checkpoint');
        yes(same(checkpoint.rows.map(row =>
          [row.sequenceSeed, row.round, row.cue, row.opponent]), frozenSchedule),
        'matched held-out schedule');
        yes(checkpoint.rows.every(row => row.emittedBit === null && row.decodedBit === null &&
          same(row.dnCounts, [0, 0]) && row.update?.step === 0 &&
          row.reward === result(row.action, row.opponent)), 'checkpoint has no advice or learning');
        yes(checkpoint.summary.wins === checkpoint.rows.filter(row => row.reward === 1).length,
          'checkpoint win count');
      }
    }
  }
  return true;
}

function bars(id, rates) {
  const host = $(id);
  host.replaceChildren();
  rates.forEach((rate, action) => {
    const row = document.createElement('div'); row.className = 'neural-bar';
    const name = document.createElement('span'); name.textContent = MOVES[action];
    const track = document.createElement('span'); track.className = 'neural-track';
    const fill = document.createElement('span'); fill.className = 'neural-fill';
    fill.style.width = `${Math.min(100, Math.max(0, rate * 50))}%`;
    track.append(fill);
    const value = document.createElement('span'); value.textContent = Number(rate).toFixed(2);
    row.append(name, track, value); host.append(row);
  });
}
const move = action => action === null ? 'none' : MOVES[action];
const scores = values => values.map((value, action) => `${MOVES[action]} ${value.toFixed(2)}`).join(' · ');
const cue = value => value === null ? 'first round: no previous move' : `previous opponent move ${move(value)}`;
const rule = offset => `Strategy ${offset === 1 ? 'A' : 'B'}: the opponent starts on a random move, then always plays the move that comes ${offset === 1 ? 'after' : 'before'} its last one (after Rock it plays ${offset === 1 ? 'Paper' : 'Scissors'}). The flies only see the opponent's last move, so they have to learn "after this move comes that one".`;
const beats = action => (action + 1) % 3; // Paper beats Rock, Scissors beats Paper, Rock beats Scissors
const ROUND_OPTIONS = [24, 48, 96, 144, 288];
const roundsOf = id => { const n = Number($(id)?.value); return ROUND_OPTIONS.includes(n) ? n : 288; };
const teacherRounds = () => roundsOf('challenge-teacher-rounds');
const studentRounds = () => roundsOf('challenge-student-rounds');
const studentPlan = () => $('challenge-student-rounds')?.value === 'six' ?
  { rounds: 288, pauseAtSix: true, preview: false, first: 6 } :
  { rounds: studentRounds(), pauseAtSix: false, preview: $('challenge-fast')?.checked === true, first: studentRounds() };
const opponentName = config => config.mode === 'custom' ? `Own: ${config.sequence.map(a => MOVES[a][0]).join('')}` :
  config.noise ? 'Noisy circle' : config.offset === 1 ? 'Circle R→P→S' : 'Circle R→S→P';
function showStudentPlan() {
  const plan = studentPlan();
  $('step1-teacher-rounds').textContent = `${teacherRounds()} rounds`;
  $('step1-student-rounds').textContent = plan.pauseAtSix ? '6, then 288 rounds' : `${plan.rounds} rounds`;
  $('compare-plan').textContent = plan.pauseAtSix ?
    'Fly A sends move advice as spikes to one Fly B. Another trains alone; a third gets random messages. All learn from their own games. Pause after 6 rounds.' :
    'Fly A sends move advice as spikes to one Fly B. Another trains alone; a third gets random messages. All learn from their own games.';
  if (!comparisonRunning) $('compare-round').textContent = `0 / ${plan.first} training rounds`;
}
let runKind = 'live';
// Finished live runs on this page, so different round counts can be compared.
const runs = [];
const pearson = points => {
  if (points.length < 3) return null;
  const mean = key => points.reduce((sum, p) => sum + p[key], 0) / points.length;
  const mx = mean('x'), my = mean('y');
  let sxy = 0, sxx = 0, syy = 0;
  for (const { x, y } of points) { sxy += (x - mx) * (y - my); sxx += (x - mx) ** 2; syy += (y - my) ** 2; }
  return sxx && syy ? sxy / Math.sqrt(sxx * syy) : null;
};
function renderRuns() {
  const body = $('runs-table'); body.replaceChildren();
  runs.forEach((run, i) => {
    const tr = document.createElement('tr');
    [i + 1, run.opponent, run.seed, run.teacherRounds, `${run.teacherWins}/72`,
      run.studentRounds ?? '–', run.taught === null ? '–' : `${run.taught}/72`,
      run.alone === null ? '–' : `${run.alone}/72`].forEach((value, j) => {
      const cell = document.createElement(j ? 'td' : 'th');
      if (!j) cell.scope = 'row';
      cell.textContent = value; tr.append(cell);
    });
    body.append(tr);
  });
  const series = [
    ['Fly A', '#5cc17a', runs.map(run => ({ x: run.teacherRounds, y: run.teacherWins }))],
    ['Taught Fly B', '#ffffff', runs.filter(run => run.taught !== null).map(run => ({ x: run.studentRounds, y: run.taught }))],
    ['Fly B alone', '#8a929b', runs.filter(run => run.alone !== null).map(run => ({ x: run.studentRounds, y: run.alone }))],
  ];
  const x = rounds => 46 + 380 * ROUND_OPTIONS.indexOf(rounds) / (ROUND_OPTIONS.length - 1);
  const y = wins => 142 - 116 * wins / 72;
  const marks = series.map(([, color, points], s) => points.map(p => s === 0 ?
    `<rect x="${(x(p.x) - 9).toFixed(1)}" y="${(y(p.y) - 4).toFixed(1)}" width="8" height="8" fill="${color}"/>` :
    `<circle cx="${(x(p.x) + (s === 1 ? 3 : 10)).toFixed(1)}" cy="${y(p.y).toFixed(1)}" r="4" fill="${color}"/>`).join('')).join('');
  const ticks = ROUND_OPTIONS.map(n => `<text x="${x(n)}" y="161" text-anchor="middle" fill="currentColor" font-size="10">${n}</text>`).join('');
  $('runs-chart').innerHTML = runs.length ? `<svg viewBox="0 0 460 184" style="width:100%;max-width:680px" aria-hidden="true"><path d="M38 26V142H440" fill="none" stroke="currentColor" opacity=".5"/><text x="32" y="31" text-anchor="end" fill="currentColor" font-size="10">72</text><text x="32" y="143" text-anchor="end" fill="currentColor" font-size="10">0</text>${marks}${ticks}<text x="240" y="179" text-anchor="middle" fill="currentColor" font-size="10">training rounds · height: test wins out of 72</text></svg>` : '';
  const notes = series.map(([name, , points]) => {
    const value = new Set(points.map(p => p.x)).size > 1 ? pearson(points) : null;
    return value === null ? null : `${name} r = ${value.toFixed(2)} (${points.length} runs)`;
  }).filter(Boolean);
  $('runs-correlation').textContent = !runs.length ? 'No finished runs yet.' : notes.length ?
    `Correlation between rounds trained and test score: ${notes.join(' · ')}. Close to +1 means more rounds went with higher scores. With only a few live runs this is a rough guide, not a research result.` :
    'Finish at least three runs with different round counts to see a correlation.';
}
function lineChart(id, points, maximum, color = '#5cc17a') {
  if (!points.length) { $(id).innerHTML = ''; return; }
  const x = i => 38 + 390 * i / Math.max(1, points.length - 1);
  const y = value => 142 - 116 * value / maximum;
  const path = points.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(p.wins).toFixed(1)}`).join(' ');
  const dots = points.map((p, i) => `<circle cx="${x(i)}" cy="${y(p.wins)}" r="3" fill="${color}"/>`).join('');
  const ticks = points.map((p, i) => `<text x="${x(i)}" y="161" text-anchor="middle" fill="currentColor" font-size="10">${p.rounds}</text>`).join('');
  $(id).innerHTML = `<svg viewBox="0 0 460 174" style="width:100%;max-width:680px" aria-hidden="true"><path d="M38 26V142H430" fill="none" stroke="currentColor" opacity=".5"/><text x="32" y="31" text-anchor="end" fill="currentColor" font-size="10">${maximum}</text><text x="32" y="143" text-anchor="end" fill="currentColor" font-size="10">0</text><path d="${path}" fill="none" stroke="${color}" stroke-width="2.5"/>${dots}${ticks}</svg>`;
}

let replay, replaySource, replayRound = 0, challengeToken = 0, challengeRunning = false;
let comparisonReady = null, comparisonRunning = false;
let pattern = [0, 1, 2];
function showReplay() {
  const seed = Number($('replay-seed').value), arm = $('replay-arm').value;
  const student = replay.students.find(s => s.seed === seed);
  const row = student.training[arm].rows[replayRound];
  const teacher = replaySource.teacher.source.training[replayRound];
  $('replay-details').hidden = false;
  $('replay-round').textContent = `Training round ${replayRound + 1} of 12`;
  $('replay-teacher-source').textContent = `Pretrained Fly A, archived continuous neural trace · ${cue(teacher.cue)}.`;
  bars('replay-teacher-bars', teacher.rates);
  $('replay-teacher-q').textContent = `Reward estimates from Fly A’s own MBON spikes: ${scores(teacher.scores)}. These are not strategy labels or transferred weights.`;
  $('replay-cue').textContent = `Ordinary sensory cue: ${cue(row.cue)}. Fly B’s own VPN spike totals (Rock/Paper/Scissors): ${row.laneCounts.join('/')}; selected cue head ${row.selectedHead === 3 ? 'free' : row.selectedHead}.`;
  bars('replay-student-bars', row.rates);
  $('replay-proposal').textContent = `1. Provisional choice: ${move(row.candidate)}. Fly B made this choice before the DN message.`;
  const hypotheticalAdvice = row.cue === null ? null : Number(teacher.scores[row.candidate] >
    Math.max(...teacher.scores.filter((_, action) => action !== row.candidate)));
  const advice = hypotheticalAdvice === null ? 'none: no previous cue' :
    hypotheticalAdvice === 1 ? 'approve' : 'reject';
  const delivered = row.emittedBit === null ? 'no message' :
    row.emittedBit === 1 ? 'approve' : 'reject';
  $('replay-advice').textContent = row.cue === null ?
    '2. No advice on the first round: neither fly has observed a previous opponent move.' :
    `2. Trained Fly A would ${advice} from its archived Q scores. Selected archived condition (${labels[arm]}) delivered: ${delivered}. A rejection means “try a different move,” not “play the correct move.”`;
  $('replay-dn').textContent = `3. Measured DNge175 spikes, 240–280 ms: neuron 3141 (bit 0) = ${row.dnCounts[0]}, neuron 3165 (bit 1) = ${row.dnCounts[1]}. Decoded message: ${row.decodedBit === null ? 'none' : row.decodedBit}.`;
  $('replay-final').textContent = `4. Final choice: ${move(row.action)}${row.action === row.candidate ? ' (kept)' : ' (changed from proposal)'}. Opponent: ${move(row.opponent)}. Game reward: ${row.reward > 0 ? '+1 win' : row.reward < 0 ? '−1 loss' : '0 tie'}.`;
  $('replay-update').textContent = `5. Fly B’s own reward update: selected cue head ${row.selectedHead}, ${move(row.action)} value step ${row.update.step.toFixed(4)}. No teacher weights were copied.`;
  const body = $('replay-controls'); body.replaceChildren();
  for (const name of arms) {
    const tr = document.createElement('tr');
    const values = [labels[name], student.initial.summary.wins,
      student.middle[name].summary.wins, student.final[name].summary.wins];
    values.forEach((value, i) => {
      const cell = document.createElement(i === 0 ? 'th' : 'td');
      cell.textContent = i === 0 ? value : `${value}/72`;
      if (i === 0) cell.scope = 'row';
      tr.append(cell);
    }); body.append(tr);
  }
  const checkpoint = Number($('replay-checkpoint').value);
  const frozen = checkpoint === 0 ? student.initial : checkpoint === 6 ?
    student.middle[arm] : student.final[arm];
  const cueDetails = [0, 1, 2].map(c => {
    const rows = frozen.rows.filter(entry => entry.cue === c);
    const wins = rows.filter(entry => entry.reward === 1).length;
    const mean = [0, 1, 2].map(a => rows.reduce((sum, entry) => sum + entry.scores[a], 0) / rows.length);
    return `${MOVES[c]} cue ${wins}/${rows.length} wins; mean Q ${scores(mean)}`;
  });
  $('replay-frozen-detail').textContent = `${labels[arm]} · ${checkpoint} training rounds · ${frozen.summary.wins}/72 frozen disconnected wins. Action choices Rock/Paper/Scissors: ${frozen.summary.actions.join('/')}. ${cueDetails.join(' | ')}. No DN message or update in any evaluation row.`;
  $('replay-conclusion').textContent = `This older proposal/veto experiment failed its locked trained-teacher advantage gate. At six rounds, trained advice scored ${student.middle.correct.summary.wins}/72, untrained advice ${student.middle.untrained.summary.wins}/72, and generic veto ${student.middle.generic.summary.wins}/72. The generic-veto rejection counts also missed their locked match (7 versus 4). All evaluation rows are disconnected and frozen.`;
}

async function loadReplay() {
  $('replay-status').textContent = 'Loading and checking archived measurements…';
  try {
    [replay, replaySource] = await Promise.all([
      data('data/rps_teacher_guided_matched_veto.json'),
      data('data/rps_teacher_guided_reset_preflight.json'),
    ]);
    validateReplay(replay, replaySource);
    replayRound = 1; showReplay();
    $('replay-status').textContent = 'Archived replay loaded. No neural simulation or learning runs in this panel.';
  } catch (error) {
    replay = replaySource = null; $('replay-details').hidden = true;
    $('replay-status').textContent = error.message;
  }
}

const snapshot = agent => JSON.stringify({ W: agent.W.map(row => [...row]), b: [...agent.b],
  baseline: agent.baseline });
const challengeConfig = () => {
  const choice = $('challenge-opponent').value;
  return choice === 'custom' ? { mode: 'custom', sequence: [...pattern] } :
    choice === 'noisy-a' ? { mode: 'preset', offset: 1, noise: 0.3 } :
    { mode: 'preset', offset: choice === 'b' ? 2 : 1 };
};
const describeChallenge = config => config.mode === 'custom' ?
  `Your pattern: ${config.sequence.map(move).join(' → ')}, then it starts again. The flies only see the opponent's last move.` :
  config.noise ? 'Noisy circle (experimental): with 80% probability the opponent plays the move after its last one (after Rock, Paper); otherwise it plays one of the other two moves (10% each). Even a perfect fly wins only about 80% of games, and whether teaching helps here (a teacher-specific benefit) is unconfirmed.' :
  rule(config.offset);
function setChallengeDisabled(disabled) {
  for (const id of ['challenge-opponent', 'challenge-teacher-rounds', 'challenge-student-rounds', 'challenge-fast', 'challenge-seed',
    'challenge-add-rock', 'challenge-add-paper', 'challenge-add-scissors']) $(id).disabled = disabled;
  $('challenge-train').disabled = disabled;
  for (const item of $('challenge-pattern-list').children)
    for (const child of item.children) if (child.tagName === 'BUTTON') child.disabled = disabled;
}
function resetComparison() {
  activeRun?.stop();
  activeRun = null;
  setEvaluating(false);
  setPlayback({ active: false, paused: false, step: 0, where: '' });
  clearComparisonDisplay();
}
function clearComparisonDisplay() {
  showPhase(-1);
  $('live-story').textContent = STORY_IDLE;
  $('mission-taught').textContent = $('mission-alone').textContent = $('mission-random').textContent = '–';
  $('mission-mode').textContent = '';
  $('mission-mode').title = '';
  missionFinal = false;
  $('score-note').textContent = 'wins in last 24 rounds';
  $('teach-message').hidden = true;
  $('stage-card').hidden = true;
  renderStrip('guided-strip', []); renderStrip('alone-strip', []);
  $('run-next').hidden = true;
  $('run-about').textContent = '';
  mission(1, 'Press ▶ Watch the 90-second story');
  comparisonReady = null;
  comparisonRunning = false;
  $('compare-start').disabled = true;
  $('compare-round').textContent = `0 / ${studentPlan().first} training rounds`;
  $('compare-continue').hidden = true;
  $('compare-continue').disabled = true;
  $('compare-continue').onclick = null;
  $('compare-status').textContent = 'Both students idle. Train Fly A first.';
  $('compare-teacher-score').textContent = '';
  $('compare-result').textContent = 'Waiting for the final test.';
  $('compare-eval-loading').hidden = true;
  $('compare-learning').hidden = true;
  $('compare-frozen-chart').replaceChildren(); $('compare-training-chart').replaceChildren();
  $('compare-checkpoints').textContent = ''; $('compare-interpretation').textContent = '';
  $('compare-cards').hidden = true;
  $('compare-difference').hidden = true;
  for (const id of ['compare-alone-score', 'compare-guided-score']) $(id).textContent = '—';
  for (const id of ['compare-alone-cues', 'compare-guided-cues', 'compare-difference']) $(id).textContent = '';
  for (const prefix of ['alone', 'guided']) {
    $(`${prefix}-brain-view`).hidden = false;
    clearBrain(prefix, 'Connectome anatomy only · live spikes begin with student training.');
    $(`${prefix}-status`).textContent = 'Idle until Fly A finishes';
    $(`${prefix}-action`).textContent = prefix === 'guided' ? '4. Final action and reward: waiting' : 'No action yet';
    $(`${prefix}-recent`).textContent = 'Recent wins: waiting';
    $(`${prefix}-kc`).textContent = 'KC activity: waiting';
    $(`${prefix}-bars`).replaceChildren();
    $(`${prefix}-q`).textContent = prefix === 'guided' ? '5. Own-reward update: waiting' : 'Learned action values: waiting';
  }
  $('guided-proposal').textContent = '1. Own action: waiting';
  $('guided-advice').textContent = '2. Fly A preference: waiting';
  $('guided-dn').textContent = '3. Measured DN spikes: waiting';
}
function renderPattern() {
  const config = challengeConfig(), custom = config.mode === 'custom';
  $('challenge-pattern-wrap').hidden = !custom;
  const host = $('challenge-pattern-list'); host.replaceChildren();
  pattern.forEach((action, index) => {
    const item = document.createElement('span'); item.className = 'pattern-item';
    const name = document.createElement('strong'); name.textContent = `${index + 1}. ${move(action)}`;
    item.append(name);
    for (const [label, allowed, change] of [
      ['↑', index > 0, () => { [pattern[index - 1], pattern[index]] = [pattern[index], pattern[index - 1]]; }],
      ['↓', index < pattern.length - 1, () => { [pattern[index + 1], pattern[index]] = [pattern[index], pattern[index + 1]]; }],
      ['Remove', pattern.length > 1, () => { pattern.splice(index, 1); }],
    ]) {
      const button = document.createElement('button'); button.type = 'button';
      button.textContent = label; button.disabled = !allowed;
      button.onclick = () => { if (allowed && !challengeRunning) { change(); challengeReset(); } };
      item.append(button);
    }
    host.append(item);
  });
  // Example sequence: custom patterns repeat from their first move; preset rules are shown starting at Rock.
  const example = custom ? Array.from({ length: Math.max(4, pattern.length + 1) }, (_, i) => pattern[i % pattern.length]) :
    Array.from({ length: 4 }, (_, i) => (i * config.offset) % 3);
  $('challenge-pattern-preview').textContent = `${example.map(move).join(' → ')} → …`;
  $('challenge-answer-preview').textContent = `${example.map(action => move(beats(action))).join(' → ')} → …`;
  const warning = $('challenge-pattern-warning');
  if (config.noise) {
    warning.className = 'finding caution';
    warning.textContent = 'Mostly predictable: about 1 move in 5 breaks the pattern, so no fly can win every game.';
  } else if (!custom) {
    warning.className = 'legend';
    warning.textContent = 'Easy to predict: each move is always followed by the same next move.';
  } else {
    const info = analyzePattern(pattern);
    const conflicts = info.ambiguous.map(action => `after ${move(action)} it sometimes plays ${info.successors[action].map(move).join(' and sometimes ')}`);
    const coverage = info.coverage.length < 3 ?
      ` Only ${info.coverage.length} of the 3 moves ever appear, so the test only covers ${info.coverage.length === 1 ? 'that one' : 'those'}.` : '';
    warning.className = conflicts.length || info.coverage.length < 3 ? 'finding caution' : 'legend';
    warning.textContent = conflicts.length ?
      `Hard to predict: ${conflicts.join('; ')}. The flies only remember one move back, so they cannot always be right.${coverage}` :
      `Easy to predict: each move in your pattern is always followed by the same next move. Learning is still not guaranteed.${coverage}`;
  }
  for (const id of ['challenge-add-rock', 'challenge-add-paper', 'challenge-add-scissors'])
    $(id).disabled = pattern.length >= 5 || challengeRunning;
}
function challengeReset() {
  challengeToken++;
  dropTimeline();
  challengeRunning = false;
  resetComparison();
  setChallengeDisabled(false);
  renderPattern();
  $('challenge-rule').textContent = describeChallenge(challengeConfig());
  const aRounds = teacherRounds();
  $('challenge-status').textContent = `Ready to train Fly A. Budget: ${aRounds / 24} episodes × 24 rounds; frozen check: 72 rounds${challengeConfig().mode === 'custom' ? ' on the same repeating pattern with fresh neural streams' : ' on three new opponent sequences'}.`;
  $('challenge-teacher-progress').textContent = `0 / ${aRounds} training rounds`;
  $('challenge-teacher-plan').textContent = `Fly A trains for ${aRounds} rounds, using wins and losses to learn the pattern. Then it plays 72 test games without learning.`;
  showStudentPlan();
  lineChart('teacher-learning-chart', [], 24);
  $('teacher-learning-note').textContent = 'Train Fly A live to fill this in.';
  clearBrain('challenge-teacher', $('brain-stage') ?
    'Connectome anatomy only · live spikes begin with Fly A training.' :
    'Brain view loads when training starts');
  $('teacher-details').hidden = true; $('teacher-details-hint').hidden = false;
  $('challenge-teacher-last').textContent = 'Last game: waiting';
  $('teacher-badge').textContent = '';
  $('challenge-teacher-kc').textContent = "Neurons that noticed the opponent's last move: waiting";
  $('challenge-teacher-bars').replaceChildren();
  $('challenge-teacher-q').textContent = 'How good Fly A thinks each move is: waiting';
  $('challenge-teacher-eval').textContent = 'Test score: after training';
  $('challenge-eval-loading').hidden = true;
}

function dropTimeline() {
  replayNav = null; pendingSeek = null;
  showTimeline(0);
  renderPlayback();
}
async function runChallenge() {
  const seed = Number($('challenge-seed').value), config = challengeConfig();
  const totalRounds = teacherRounds(), episodes = totalRounds / 24;
  if (!Number.isInteger(seed) || seed < 1 || seed > 9999 ||
    config.mode === 'preset' && ![1, 2].includes(config.offset)) {
    $('challenge-status').textContent = 'Choose a supported opponent mode and an integer seed from 1 to 9999.'; return;
  }
  const token = ++challengeToken;
  challengeRunning = true;
  dropTimeline();
  resetComparison();
  $('teacher-details').hidden = false; $('teacher-details-hint').hidden = true;
  $('challenge-eval-loading').hidden = true;
  setPlayback({ active: true });
  clearBrain('challenge-teacher', 'Brain view loads when training starts');
  setChallengeDisabled(true);
  $('challenge-status').textContent = 'Loading the original connectome and fixed MBON feature calibration…';
  try {
    const [sub, archive, pointBytes, coordinates] = await Promise.all([
      data('data/subgraph.json'), data('data/rps_contextual_bandit.json'),
      data('data/brain_points.bin', 'arrayBuffer'),
      brainCoordinates3D(),
    ]);
    if (token !== challengeToken) return;
    if (!archive.allValid || !archive.design?.featureMean || !archive.design?.featureStd)
      throw Error('Saved MBON calibration is unavailable.');
    const net = new Network(sub, wiring(sub, 'real', seed), { seed });
    const points = new Uint16Array(pointBytes);
    const teacherBrain = brain('challenge-teacher', coordinates, points, sub);
    if ($('brain-stage')) anatomyBackground = teacherBrain.bg;
    teacherBrain.frame(net);
    const agent = new RpsContextualBandit(net, { seed,
      featureMean: archive.design.featureMean, featureStd: archive.design.featureStd,
      readoutLearn: true });
    const recent = [], byCue = [0, 1, 2].map(() => ({ wins: 0, total: 0 }));
    let completed = 0;
    const blocks = [];
    async function runEpisode(ep, episodeKey, training) {
      while (!ep.done && token === challengeToken) {
        const started = performance.now();
        // Display pacing: slow speeds cap simulation steps per animation frame; pause holds here.
        const stepCap = playback.speed < 1 ? (playback.speed <= 0.25 ? 4 : 12) : Infinity;
        let stepsThisFrame = 0;
        do {
          if (blocked() || stepsThisFrame++ >= stepCap) break;
          const phase = ep.t % 400;
          if (phase === 0) {
            const round = Math.floor(ep.t / 400), key = seed * 10000000 + episodeKey * 1000 + round;
            net.rand = rng(key + 1); agent.rand = rng(key + 501);
          }
          const activeKcs = phase === T_DECIDE ? net.idx.kc.reduce((n, i) => n + Number(net.count[i] > 0), 0) : 0;
          ep.step();
          if (phase === T_DECIDE && training) {
            teacherBrain.frame(net, ep);
            const event = ep.events.at(-1);
            $('live-story').textContent = 'Playing the opponent and learning from wins and losses.';
            mission(2, training ? 'Fly A is learning the game' : 'Fly A is taking its test', training ? completed / totalRounds : 1);
            $('challenge-teacher-last').textContent = `Last game${training ? '' : ' (test)'}: Fly A played ${move(event.action)}, the opponent played ${move(event.opponent)} · ${event.reward > 0 ? 'win' : event.reward < 0 ? 'loss' : 'tie'}`;
            $('challenge-teacher-kc').textContent = `Neurons that noticed the opponent's last move: ${activeKcs} Kenyon cells fired`;
            bars('challenge-teacher-bars', agent.rates);
            $('challenge-teacher-q').textContent = `How good Fly A thinks each move is: ${scores(agent.scores)} (higher is better; learned only from its own wins and losses, never told the answer)`;
          }
          if (phase === 399) roundDone(training ? `Fly A round ${completed + 1}/${totalRounds}` :
            `test round ${ep.events.length}/24`);
          if (phase === 399 && !training && ep.events.length % 12 === 0)
            $('challenge-eval-loading-text').textContent = `Checking 72 frozen games… ${(episodeKey - 100) * 24 + ep.events.length}/72`;
          if (phase === 399 && training) {
            const event = ep.events.at(-1); completed++;
            recent.push(event.reward === 1 ? 1 : 0); if (recent.length > 24) recent.shift();
            const recentWins = recent.reduce((a, b) => a + b, 0);
            $('teacher-badge').textContent = `Fly A is learning · round ${completed}/${totalRounds} · won ${recentWins} of its last ${recent.length} games`;
            $('challenge-teacher-progress').textContent = `${completed} / ${totalRounds} training rounds · recent win rate ${(100 * recentWins / recent.length).toFixed(1)}% (${recentWins}/${recent.length})`;
          }
        } while (!ep.done && token === challengeToken && performance.now() - started < 12 * Math.max(1, playback.speed));
        // Frozen checks have no neural display frames to paint; yield promptly
        // while leaving the page responsive and the loading bar animated.
        if (training || blocked()) await new Promise(resolve => requestAnimationFrame(resolve));
        else await sleep(0);
      }
    }
    const trainingPattern = config.mode === 'custom' ? challengeOpponent(config, 41000) : null;
    for (let episode = 0; episode < episodes && token === challengeToken; episode++) {
      $('challenge-status').textContent = `Training Fly A on ${describeChallenge(config)} Episode ${episode + 1}/${episodes}.`;
      const ep = new RpsEpisode(net, agent, trainingPattern ?? challengeOpponent(config, 41000 + episode),
        { rounds: 24, learn: true });
      await runEpisode(ep, episode, true);
      if (token !== challengeToken) break;
      blocks.push({ rounds: completed, wins: ep.events.filter(event => event.reward === 1).length });
      lineChart('teacher-learning-chart', blocks, 24);
      $('teacher-learning-note').textContent = `Across: rounds trained so far · height: wins in that block of 24 training games. Fly A has trained ${completed} of ${totalRounds} rounds.`;
    }
    if (token !== challengeToken) return;
    const trained = snapshot(agent);
    agent.explore = false;
    let wins = 0;
    $('challenge-eval-loading-text').textContent = 'Checking 72 frozen games… 0/72';
    $('challenge-eval-loading').hidden = false;
    setEvaluating(true);
    mission(2, 'Fly A is taking its frozen test', 1);
    $('live-story').textContent = 'Checking 72 new games without exploration or learning.';
    const evaluationPattern = config.mode === 'custom' ? challengeOpponent(config, 51000) : null;
    for (let evaluation = 0; evaluation < 3 && token === challengeToken; evaluation++) {
      $('challenge-status').textContent = `Frozen, teacher-free held-out check ${evaluation + 1}/3…`;
      const ep = new RpsEpisode(net, agent, evaluationPattern ?? challengeOpponent(config, 51000 + evaluation),
        { rounds: 24, learn: false });
      await runEpisode(ep, 100 + evaluation, false);
      for (const event of ep.events) {
        wins += Number(event.reward === 1);
        const c = event.observation.previousOpponent;
        if (c !== null) { byCue[c].total++; byCue[c].wins += Number(event.reward === 1); }
      }
    }
    if (token !== challengeToken) return;
    $('challenge-eval-loading').hidden = true;
    setEvaluating(false);
    if (snapshot(agent) !== trained || !net.m.every(value => value === 1))
      throw Error('Frozen evaluation changed a controller or KC→MBON weight.');
    const byCueText = byCue.map((entry, action) =>
      `${move(action)} ${entry.total ? `${entry.wins}/${entry.total}` : 'not observed'}`).join(', ');
    $('challenge-teacher-eval').textContent = `Test score: won ${wins} of 72 new games with learning switched off. After each opponent move: ${byCueText}.`;
    $('compare-teacher-score').textContent = `Fly A: ${wins}/72 frozen wins.`;
    $('challenge-status').textContent = `Live Fly A training and frozen evaluation complete for seed ${seed}. This result belongs only to this Challenge run; the Fly B replay is independent.`;
    const run = { opponent: opponentName(config), seed, teacherRounds: totalRounds, teacherWins: wins,
      studentRounds: null, taught: null, alone: null };
    runs.push(run); renderRuns();
    $('teacher-learning-note').textContent = `Across: rounds trained so far · height: wins in that block of 24 training games. After ${totalRounds} rounds, Fly A won ${wins}/72 test games.`;
    comparisonReady = { run, sub, net, agent, seed, config, points, coordinates, teacherBrain,
      featureMean: archive.design.featureMean, featureStd: archive.design.featureStd };
    $('compare-start').disabled = false;
    $('compare-status').textContent = 'Fly A finished. Both students are still idle.';
    $('teacher-badge').textContent = `Fly A passed its test: won ${wins} of 72 games`;
    mission(2, `Fly A scored ${wins}/72. Press Teach Fly B`, 1);
    $('live-story').textContent = 'Fly A is frozen and ready to teach.';
  } catch (error) {
    if (token === challengeToken) {
      $('challenge-status').textContent = `Challenge stopped: ${error.message}`;
      mission(2, 'Stopped'); $('live-story').textContent = error.message;
    }
  } finally {
    if (token === challengeToken) {
      $('challenge-eval-loading').hidden = true;
      setEvaluating(false);
      challengeRunning = false;
      setPlayback({ active: false, paused: false, step: 0 });
      setChallengeDisabled(false);
      renderPattern();
    }
  }
}

const recentText = values => `Recent wins: ${values.reduce((a, b) => a + b, 0)}/${values.length} (last ${values.length} rounds)`;
function learningChart(id, points, maximum, label) {
  if (!points.length) return;
  const x = i => 38 + 390 * i / Math.max(1, points.length - 1);
  const y = value => 142 - 116 * value / maximum;
  const colors = ['#8a929b', '#ffffff', '#5a616a']; // alone, taught, random messages
  const lines = points[0].wins.map((_, arm) => {
    const path = points.map((point, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(point.wins[arm]).toFixed(1)}`).join(' ');
    const dots = points.map((point, i) => `<circle cx="${x(i)}" cy="${y(point.wins[arm])}" r="3" fill="${colors[arm]}"/>`).join('');
    return `<path d="${path}" fill="none" stroke="${colors[arm]}" stroke-width="2.5"${arm === 2 ? ' stroke-dasharray="5 4"' : ''}/>${dots}`;
  }).join('');
  const ticks = points.map((point, i) => `<text x="${x(i)}" y="161" text-anchor="middle" fill="currentColor" font-size="10">${point[label]}</text>`).join('');
  $(id).innerHTML = `<svg viewBox="0 0 460 174" style="width:100%;max-width:680px" aria-hidden="true"><path d="M38 26V142H430" fill="none" stroke="currentColor" opacity=".5"/><text x="32" y="31" text-anchor="end" fill="currentColor" font-size="10">${maximum}</text><text x="32" y="143" text-anchor="end" fill="currentColor" font-size="10">0</text>${lines}${ticks}</svg>`;
}
export async function loadSpeedEvidence() {
  try {
    const archive = await data('data/rps_teacher_specific_speed_initial.json');
    const pair = archive.pairs?.[0];
    const rounds = [0, 6, 12, 24, 48, 96, 288];
    const arms = ['correct', 'alone', 'untrained', 'shuffled', 'generic'];
    if (archive.passed !== false || archive.pairs.length !== 1 ||
        pair.teacherSeed !== 830 || pair.studentSeed !== 10830 ||
        !same(pair.checkpoints.map(p => p.rounds), rounds) ||
        pair.checkpoints.some(p => arms.some(arm => !Number.isInteger(p.wins[arm]) ||
          p.wins[arm] < 0 || p.wins[arm] > 72)) ||
        !same(pair.first58, { correct: 12, alone: 96, untrained: 12,
          shuffled: 12, generic: 12 }))
      throw Error('archived five-arm checkpoints did not match the documented protocol');
    const colors = ['#e466b8', '#5cc17a', '#66bde0', '#f5b45c', '#b29cf2'];
    const x = i => 38 + 390 * i / 6;
    const y = wins => 142 - 116 * wins / 72;
    const lines = arms.map((arm, index) => {
      const points = pair.checkpoints.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(p.wins[arm]).toFixed(1)}`).join(' ');
      const dots = pair.checkpoints.map((p, i) => `<circle cx="${x(i)}" cy="${y(p.wins[arm])}" r="2.5" fill="${colors[index]}"/>`).join('');
      return `<path d="${points}" fill="none" stroke="${colors[index]}" stroke-width="2.5"/>${dots}`;
    }).join('');
    const ticks = rounds.map((round, i) => `<text x="${x(i)}" y="161" text-anchor="middle" fill="currentColor" font-size="10">${round}</text>`).join('');
    $('evidence-speed-chart').innerHTML = `<svg viewBox="0 0 460 174" style="width:100%;max-width:680px" aria-hidden="true"><path d="M38 26V142H430" fill="none" stroke="currentColor" opacity=".5"/><text x="32" y="31" text-anchor="end" fill="currentColor" font-size="10">72</text><text x="32" y="143" text-anchor="end" fill="currentColor" font-size="10">0</text>${lines}${ticks}</svg>`;
    const body = $('evidence-speed-table'); body.replaceChildren();
    for (const point of pair.checkpoints) {
      const tr = document.createElement('tr');
      [point.rounds, ...arms.map(arm => `${point.wins[arm]}/72`)].forEach((value, i) => {
        const cell = document.createElement(i ? 'td' : 'th');
        if (!i) cell.scope = 'row';
        cell.textContent = value; tr.append(cell);
      });
      body.append(tr);
    }
    const immediate = pair.checkpoints.find(p => p.rounds === 12);
    const retained = pair.checkpoints.find(p => p.rounds === 288);
    const diagnosticTable = $('diagnostic-table-body'); diagnosticTable.replaceChildren();
    for (const point of pair.checkpoints) {
      const tr = document.createElement('tr');
      [point.rounds, ...arms.map(arm => `${point.wins[arm]}/72`)].forEach((value, i) => {
        const cell = document.createElement(i ? 'td' : 'th');
        cell.textContent = value; if (!i) cell.scope = 'row'; tr.append(cell);
      });
      diagnosticTable.append(tr);
    }
    $('diagnostic-immediate').textContent = `After 12 training rounds, the trained-advice policy scored ${immediate.wins.correct}/72 frozen disconnected wins; untrained, shuffled, and generic veto scored ${immediate.wins.untrained}/72, ${immediate.wins.shuffled}/72, and ${immediate.wins.generic}/72. This older proposal/veto result did not show a teacher-specific retained advantage.`;
    $('diagnostic-retained').textContent = `By 288 rounds, all five older proposal/veto conditions scored ${retained.wins.correct}/72 frozen disconnected wins. This is separate from the newer action-preference experiment below.`;
    $('evidence-speed-status').textContent = 'Line colors: trained advice pink; no advice green; untrained blue; shuffled orange; matched generic purple. Archived five-arm gate: failed.';
  } catch (error) {
    $('evidence-speed-status').textContent = `Archived chart unavailable: ${error.message}`;
  }
}
const preferenceEvidence = {
  'a-11007': ['rps_action_preference_teaching_v3.json', 11007],
  'a-11008': ['rps_action_preference_teaching_v3.json', 11008],
  'a-11011': ['rps_action_preference_confirmation_first.json', 11011],
  'a-11012': ['rps_action_preference_confirmation_second.json', 11012],
  'b-11015': ['rps_action_preference_strategy_b_initial.json', 11015],
  'b-11016': ['rps_action_preference_strategy_b_confirmation.json', 11016],
};
const preferenceArchives = new Map();
let preferenceLoadId = 0;
export async function loadActionPreferenceEvidence() {
  const loadId = ++preferenceLoadId;
  const chart = $('preference-evidence-chart'), body = $('preference-evidence-table');
  chart.replaceChildren(); body.replaceChildren();
  if ($('preference-event-status')) {
    for (const id of ['preference-event-teacher', 'preference-event-dn',
      'preference-event-student', 'preference-event-update']) $(id).textContent = '';
    $('preference-event-status').textContent = 'Loading recorded round…';
  }
  try {
    const [file, seed] = preferenceEvidence[$('preference-evidence-seed').value] ?? [];
    if (!file) throw Error('unknown archived student');
    if (!preferenceArchives.has(file)) preferenceArchives.set(file, data(`data/${file}`));
    const archive = await preferenceArchives.get(file);
    if (loadId !== preferenceLoadId) return;
    const pair = archive.pilot.pairs?.find(x => x.seed === seed) ??
      (archive.protocol.studentSeed === seed ? archive.pilot : null);
    const rounds = [0, 6, 12, 24, 48, 96, 288];
    const arms = ['trained', 'alone', 'untrained', 'shuffled', 'generic'];
    if (!pair || !['passed', 'pilot_passed_unconfirmed'].includes(archive.status) ||
        (archive.status === 'pilot_passed_unconfirmed' && !file.includes('teaching_v3')) ||
        pair.passed !== true ||
        arms.some(arm => !pair.byArm[arm] || !same(pair.byArm[arm].checkpoints.map(p => p.rounds), rounds) ||
          pair.byArm[arm].checkpoints.some(p => !Number.isInteger(p.wins) || p.wins < 0 || p.wins > 72 ||
            (p.rows && p.rows.filter(r => r.reward === 1).length !== p.wins))))
      throw Error('saved frozen checkpoints failed validation');
    const training = pair.byArm.trained.training;
    const sampleIndex = training.findIndex(row => row.decoded >= 0);
    const sample = training[sampleIndex];
    if (!sample || sample.target !== sample.decoded || sample.cue === null ||
        !Array.isArray(sample.rates) || sample.rates.length !== 3 ||
        !Array.isArray(sample.dnCounts) || sample.dnCounts.length !== 2 ||
        sample.dnCounts.reduce((n, count) => n + Number(count > 0), 0) !== (sample.target === 2 ? 2 : 1))
      throw Error('recorded teacher preference or measured DN counts are invalid');
    $('preference-event-teacher').textContent = `After previous opponent move ${move(sample.cue)}, Fly A's frozen reward-trained policy preferred ${move(sample.target)}.`;
    $('preference-event-dn').textContent = `Measured DNge175 spikes: neuron 3141 = ${sample.dnCounts[0]}, neuron 3165 = ${sample.dnCounts[1]}. Fly B decoded ${move(sample.decoded)} from those counts.`;
    $('preference-event-student').textContent = `Fly B had already played ${move(sample.action)} against ${move(sample.opponent)}: ${sample.reward > 0 ? 'win (+1)' : sample.reward < 0 ? 'loss (−1)' : 'tie (0)'}. Its recorded MBON rates (Rock/Paper/Scissors) were ${sample.rates.map(rate => rate.toFixed(2)).join(' / ')} spikes per neuron.`;
    $('preference-event-update').textContent = `After that action, Fly B's own-reward value step was ${sample.rewardStep.toFixed(4)}; decoded-preference imitation update ${sample.imitation ? 'applied' : 'not applied'}. Frozen evaluation disables both updates and all teacher messages.`;
    $('preference-event-status').textContent = `Archived student ${seed}, training round ${sampleIndex + 1}; recorded measurements, not a live run or a cell-level spike replay. Per-trial Fly A Q scores were not saved.`;
    const colors = ['#e466b8', '#5cc17a', '#66bde0', '#f5b45c', '#b29cf2'];
    const x = i => 38 + 390 * i / 6;
    const y = wins => 142 - 116 * wins / 72;
    const lines = arms.map((arm, a) => {
      const points = pair.byArm[arm].checkpoints;
      const path = points.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(p.wins).toFixed(1)}`).join(' ');
      const dots = points.map((p, i) => `<circle cx="${x(i)}" cy="${y(p.wins)}" r="2.5" fill="${colors[a]}"/>`).join('');
      return `<path d="${path}" fill="none" stroke="${colors[a]}" stroke-width="2.5"/>${dots}`;
    }).join('');
    const ticks = rounds.map((round, i) => `<text x="${x(i)}" y="161" text-anchor="middle" fill="currentColor" font-size="10">${round}</text>`).join('');
    chart.innerHTML = `<svg viewBox="0 0 460 174" style="width:100%;max-width:680px" aria-hidden="true"><path d="M38 26V142H430" fill="none" stroke="currentColor" opacity=".5"/><text x="32" y="31" text-anchor="end" fill="currentColor" font-size="10">72</text><text x="32" y="143" text-anchor="end" fill="currentColor" font-size="10">0</text>${lines}${ticks}</svg>`;
    for (let i = 0; i < rounds.length; i++) {
      const row = document.createElement('tr');
      [rounds[i], ...arms.map(arm => `${pair.byArm[arm].checkpoints[i].wins}/72`)].forEach((value, j) => {
        const cell = document.createElement(j ? 'td' : 'th');
        if (!j) cell.scope = 'row';
        cell.textContent = value; row.append(cell);
      });
      body.append(row);
    }
    const early = arm => [1, 2, 3, 4].reduce((sum, i) => sum + pair.byArm[arm].checkpoints[i].wins, 0) / 4;
    const advantage = early('trained') - Math.max(...arms.slice(1).map(early));
    const first = arm => pair.byArm[arm].checkpoints.find(p => p.wins >= 58)?.rounds ?? null;
    $('preference-evidence-status').textContent = `Archived student ${seed}: trained advice first reached 58/72 at round ${first('trained')}; early mean advantage over the strongest control was +${advantage}/72. Final disconnected wins: trained ${pair.byArm.trained.checkpoints[6].wins}/72, no message ${pair.byArm.alone.checkpoints[6].wins}/72. Pink trained, green silent, blue untrained, orange shuffled, purple generic. Saved source: data/${file}.`;
  } catch (error) {
    if (loadId !== preferenceLoadId) return;
    $('preference-evidence-status').textContent = `Archived action-preference result unavailable: ${error.message}`;
    if ($('preference-event-status')) $('preference-event-status').textContent = `Recorded round unavailable: ${error.message}`;
  }
}
let ablationArchive;
let ablationLoadId = 0;
export async function loadAblationEvidence() {
  const loadId = ++ablationLoadId;
  const chart = $('ablation-evidence-chart'), body = $('ablation-evidence-table');
  chart.replaceChildren(); body.replaceChildren();
  try {
    if (!ablationArchive) ablationArchive = data('data/rps_connectome_contribution_ablation.json');
    const archive = await ablationArchive;
    if (loadId !== ablationLoadId) return;
    const rounds = [0, 6, 12, 24, 48, 96, 288];
    const arms = ['neural', 'digital', 'silent', 'scrambled'];
    const expectedEarly = { 11017: [70, 70, 35.5, 52.75], 11018: [69, 69, 33.75, 64] };
    if (archive.status !== 'complete' || archive.pairs?.length !== 2 ||
        !same(archive.pairs.map(p => p.studentSeed), [11017, 11018]))
      throw Error('saved pair identities or completion status changed');
    for (const saved of archive.pairs) {
      if (!saved.source?.passed || !saved.receiver?.passed ||
          !saved.summary?.sameTrain || !saved.summary?.sameFrozen ||
          !saved.summary?.sameFinalWeights || !saved.summary?.transmissionPassed ||
          arms.some(arm => !same(saved.byArm?.[arm]?.checkpoints?.map(c => c.rounds), rounds) ||
            saved.byArm[arm].checkpoints.some(c => c.rows?.length !== 72 ||
              c.wins !== c.rows.filter(r => r.reward === 1).length ||
              !c.disconnected || !c.weightsUnchanged)))
        throw Error('saved receiver, transmission, or frozen checkpoints failed validation');
      const early = arms.map(arm => [1, 2, 3, 4].reduce((sum, i) =>
        sum + saved.byArm[arm].checkpoints[i].wins, 0) / 4);
      if (!same(early, expectedEarly[saved.studentSeed]) ||
          saved.byArm.neural.checkpoints.some((c, i) =>
            c.wins !== saved.byArm.digital.checkpoints[i].wins) ||
          saved.byArm.neural.training.filter(r => r.target !== null && r.decoded === r.target).length !== 276)
        throw Error('saved ablation values did not match the archived result');
    }
    const seed = Number($('ablation-evidence-seed').value);
    const pair = archive.pairs.find(p => p.studentSeed === seed);
    if (!pair) throw Error('unknown archived student');
    // Same look as the 15-run result chart on the Stats tab: white for the taught line, greys and dash
    // patterns for the controls, and each line named at its right-hand end.
    const x = i => 56 + 528 * i / 6, y = wins => 290 - 250 * wins / 72;
    const wins = arm => pair.byArm[arm].checkpoints.map(c => c.wins);
    const path = arm => wins(arm).map((w, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)} ${y(w).toFixed(1)}`).join(' ');
    const grid = [0, 24, 48, 72].map(w => `<line x1="56" x2="584" y1="${y(w)}" y2="${y(w)}" stroke="#2e2d2a"/>` +
      `<text x="48" y="${y(w) + 4}" text-anchor="end" fill="#a3abb4" font-size="12">${w}</text>`).join('');
    const ticks = rounds.map((n, i) => `<text x="${x(i)}" y="310" text-anchor="middle" fill="#a3abb4" font-size="12">${n}</text>`).join('');
    const line = (arm, stroke, width, dash = '') => `<path d="${path(arm)}" fill="none" stroke="${stroke}" stroke-width="${width}" ` +
      `stroke-linecap="round" stroke-linejoin="round"${dash ? ` stroke-dasharray="${dash}"` : ''}/>`;
    const dots = wins('neural').map((w, i) => `<circle cx="${x(i)}" cy="${y(w)}" r="4.5" fill="#ffffff" stroke="#191918" stroke-width="2"/>`).join('');
    // End labels, pushed apart where lines finish at the same score.
    const labels = [
      { text: 'Neural spikes = direct digital', w: wins('neural')[6], fill: '#ffffff', bold: true, stroke: '#ffffff', width: 3, dash: '' },
      { text: 'Scrambled neural', w: wins('scrambled')[6], fill: '#a3abb4', stroke: '#a3abb4', width: 2, dash: '2 4' },
      { text: 'No advice', w: wins('silent')[6], fill: '#a3abb4', stroke: '#8a929b', width: 2, dash: '' },
    ].map(l => ({ ...l, at: y(l.w) + 4 })).sort((a, b) => a.at - b.at);
    labels.forEach((l, i) => { if (i) l.at = Math.max(l.at, labels[i - 1].at + 18); });
    // A short sample of each line's stroke sits before its name.
    const names = labels.map(l => `<line x1="596" x2="618" y1="${l.at - 4}" y2="${l.at - 4}" stroke="${l.stroke}" stroke-width="${l.width}"` +
      `${l.dash ? ` stroke-dasharray="${l.dash}"` : ''}/><text x="626" y="${l.at}" fill="${l.fill}" font-size="13"${l.bold ? ' font-weight="700"' : ''}>${l.text}</text>`).join('');
    chart.innerHTML = `<svg viewBox="0 0 840 330" style="width:100%;max-width:880px;display:block" aria-hidden="true">${grid}` +
      `<path d="M56 290H584" fill="none" stroke="#5f5d58"/>${ticks}` +
      `<text x="320" y="327" text-anchor="middle" fill="#a3abb4" font-size="12">Training rounds (checkpoints shown evenly spaced)</text>` +
      line('silent', '#8a929b', 2) + line('scrambled', '#a3abb4', 2, '2 4') + line('neural', '#ffffff', 3) +
      line('digital', '#191918', 1.4, '5 5') + dots + names + '</svg>';
    for (let i = 0; i < rounds.length; i++) {
      const row = document.createElement('tr');
      [rounds[i], ...arms.map(arm => `${pair.byArm[arm].checkpoints[i].wins}/72`)].forEach((value, j) => {
        const cell = document.createElement(j ? 'td' : 'th');
        if (!j) cell.scope = 'row';
        cell.textContent = value; row.append(cell);
      });
      body.append(row);
    }
    const early = expectedEarly[seed];
    $('ablation-evidence-status').textContent = `Student ${seed}, early mean frozen wins/72 at rounds 6/12/24/48: neural ${early[0]}, digital ${early[1]}, silent ${early[2]}, scrambled ${early[3]}. Neural and digital actions and weights matched exactly; both finished ${pair.byArm.neural.checkpoints[6].wins}/72 disconnected. Silence also finished ${pair.byArm.silent.checkpoints[6].wins}/72.`;
  } catch (error) {
    if (loadId !== ablationLoadId) return;
    $('ablation-evidence-status').textContent = `Archived ablation unavailable: ${error.message}`;
  }
}
function showCheckpointResults(checkpoints, episodes, total = 288) {
  $('compare-learning').hidden = false;
  learningChart('compare-frozen-chart', checkpoints, 72, 'rounds');
  learningChart('compare-training-chart', episodes, 24, 'episode');
  $('compare-checkpoints').textContent = checkpoints.map(point =>
    `${point.rounds} rounds: Alone ${point.wins[0]}/72, With Teacher ${point.wins[1]}/72 (${point.wins[1] - point.wins[0] >= 0 ? '+' : ''}${point.wins[1] - point.wins[0]}), Random messages ${point.wins[2]}/72`).join(' · ');
  if (total !== 288) {
    const trained = checkpoints.filter(p => p.rounds > 0);
    if (!trained.length) return;
    const deltas = trained.map(p => p.wins[1] - p.wins[0]);
    const mean = deltas.reduce((a, b) => a + b, 0) / deltas.length;
    $('compare-interpretation').textContent = `At ${trained.map(p => p.rounds).join(', ')} rounds the taught Fly B averaged ${mean >= 0 ? '+' : ''}${mean.toFixed(1)}/72 wins versus Fly B alone. The study's early-advantage check needs the full 288 rounds, so it is not applied to this shorter run. This live run also includes a random-message control.`;
    return;
  }
  const early = checkpoints.filter(p => [6, 12, 24, 48, 96].includes(p.rounds));
  if (early.length !== 5) return;
  const deltas = early.map(p => p.wins[1] - p.wins[0]);
  const mean = deltas.reduce((a, b) => a + b, 0) / deltas.length;
  const final = checkpoints.find(p => p.rounds === 288);
  const passed = mean >= 4 && deltas[0] >= 4 && deltas[1] >= 4 && final && final.wins[1] >= final.wins[0];
  const description = deltas.every(d => d > 0) ? 'Teacher-guided student led at every early measured checkpoint.' :
    deltas.every(d => d < 0) ? 'Student without teacher led at every early measured checkpoint.' :
    deltas.every(d => Math.abs(d) < 4) ? 'Students learned at similar measured rates.' :
    'Results are mixed across measured checkpoints.';
  const threshold = arm => checkpoints.find(p => p.wins[arm] >= 58)?.rounds ?? 'not reached';
  $('compare-interpretation').textContent = `${description} Mean early difference: ${mean >= 0 ? '+' : ''}${mean.toFixed(1)}/72 wins. Predeclared single-run early-advantage criterion: ${passed ? 'met' : 'not met'}. First sampled 58/72 checkpoint: Alone ${threshold(0)}, With Teacher ${threshold(1)}. Random messages: ${threshold(2)}. This live run has alone and random-message controls; the full archived controls are in Research Evidence.`;
}
function showStudent(prefix, row, recent, round, total = 288) {
  $(`${prefix}-status`).textContent = `Training round ${round}/${total} ·${row.cue === null ? 'first round, no cue' : `previous move ${move(row.cue)}`}`;
  $(`${prefix}-action`).textContent = `${prefix === 'guided' ? '4. Final action and reward: ' : ''}${move(row.action)} vs ${move(row.opponent)} · ${row.reward > 0 ? 'win (+1)' : row.reward < 0 ? 'loss (−1)' : 'draw (0)'}`;
  $(`${prefix}-recent`).textContent = recentText(recent);
  $(`${prefix}-kc`).textContent = `${row.activeKcs} KCs active · VPN cue head ${row.selectedHead === 3 ? 'none' : move(row.selectedHead)}`;
  bars(`${prefix}-bars`, row.rates);
  $(`${prefix}-q`).textContent = `${prefix === 'guided' ? '5. ' : ''}Own-reward update: ${row.updateStep.toFixed(4)} · action values ${scores(row.learnedScores)}`;
}
function studentEventHandler(views, { preview = false } = {}) {
  const recent = [[], [], []], checkpoints = [], episodes = [], strips = [[], []];
  let total = 288;
  const wins = list => list.reduce((a, b) => a + b, 0);
  const outcome = reward => reward > 0 ? 'win' : reward < 0 ? 'loss' : 'draw';
  const showFrozenScores = (totals, byCue, label) => {
    const cueText = tallies => tallies.map((x, cue) =>
      `${move(cue)} ${x.n ? `${x.wins}/${x.n}` : 'not observed'}`).join(', ');
    const delta = totals[1] - totals[0];
    $('compare-result').textContent = `${label}: same 72-round frozen opponent test. Fly A disconnected. No advice, exploration, or learning during evaluation.`;
    $('compare-alone-score').textContent = `${totals[0]}/72 wins · ${(100 * totals[0] / 72).toFixed(1)}%`;
    $('compare-guided-score').textContent = `${totals[1]}/72 wins · ${(100 * totals[1] / 72).toFixed(1)}%`;
    $('compare-alone-cues').textContent = `By previous move: ${cueText(byCue[0])}`;
    $('compare-guided-cues').textContent = `By previous move: ${cueText(byCue[1])}`;
    $('compare-difference').textContent = `With Teacher minus Alone: ${delta >= 0 ? '+' : ''}${delta} wins out of 72. This live comparison does not by itself isolate trained-teacher information from other advice.`;
    $('compare-cards').hidden = false;
    $('compare-difference').hidden = false;
  };
  return event => {
      if (event.type === 'error') throw Error(event.message);
      if (event.type === 'phase') showPhase(event.phase);
      if (event.type === 'frame') event.nets.forEach((net, i) => views[i].frame(net));
      if (event.type === 'checkpoint') {
        checkpoints.push(event.point);
        checkpoints.sort((a, b) => a.rounds - b.rounds);
        if (!preview) showCheckpointResults(checkpoints, episodes, total);
      }
      if (event.type === 'episode') {
        episodes.push(event.point);
        if (!preview) showCheckpointResults(checkpoints, episodes, total);
      }
      if (event.type === 'round') {
        const { completed, rows, message, episode, episodes: episodeCount } = event;
        total = event.total ?? 288;
        $('compare-round').textContent = `${completed} / ${total} training rounds`;
        rows.forEach((row, i) => {
          recent[i].push(Number(row.reward === 1));
          if (i < 2) { strips[i].push(outcomeCode(row.reward)); if (strips[i].length > 24) strips[i].shift(); }
          if (recent[i].length > 24) recent[i].shift();
          if (i < 2) showStudent(i ? 'guided' : 'alone', row, recent[i], completed, total);
        });
        const guidedRow = rows[1];
        $('guided-proposal').textContent = `1. Own action: ${move(guidedRow.action)} (chosen before any message; the message cannot change it)`;
        $('guided-advice').textContent = message.preference === null ? '2. Fly A preference: none on the first cue-free round' :
          `2. Fly A preference: ${move(message.preference)} (from its frozen MBON-driven values)`;
        $('guided-dn').textContent = `3. Measured DNge175 spikes: 3141 ${message.dnCounts[0]} · 3165 ${message.dnCounts[1]} · decoded ${message.decoded === null ? 'none' : move(message.decoded)}`;
        const lesson = message.preference === null ? 'No cue yet' :
          `Fly A says ${move(message.preference)} · ${message.dnCounts[0] + message.dnCounts[1]} spikes`;
        $('live-story').textContent = runKind === 'ruleSwitch' ?
          "Fly A's spike message reaches one Fly B. All students keep learning from their own wins and losses as the new rule takes hold." :
          `Round ${completed}/${total} · ${lesson} · Taught: ${outcome(guidedRow.reward)} · Alone: ${outcome(rows[0].reward)}`;
        $('mission-taught').textContent = wins(recent[1]);
        $('mission-alone').textContent = wins(recent[0]);
        $('mission-random').textContent = wins(recent[2]);
        missionFinal = false;
        $('score-note').textContent = 'wins in last 24 rounds';
        renderStrip('alone-strip', strips[0]); renderStrip('guided-strip', strips[1]);
        showTeachMessage(message, guidedRow);
        mission(3, runKind === 'ruleSwitch' ? 'Fly A is teaching Fly B the new rule' : 'Fly A is teaching Fly B', completed / total);
        if (guidedRow.imitation) $('guided-q').textContent += ' · then imitation update toward decoded preference';
        roundDone(`round ${completed}/${total}`);
        $('compare-status').textContent = `Both students learn from their own game rewards; only the taught student also applies imitation updates. Fly A's controller is frozen. Episode ${episode + 1}/${episodeCount ?? Math.ceil(total / 24)}.`;
      }
      if (event.type === 'six_evaluating') {
        $('compare-eval-loading').hidden = false;
        setEvaluating(true);
        mission(4, 'Checking both 72-game tests', 1);
        $('live-story').textContent = 'Both Fly Bs are being tested with Fly A disconnected.';
        $('compare-status').textContent = 'Checking both frozen 72-game student scores after six training rounds…';
      }
      if (event.type === 'six_ready') {
        $('compare-eval-loading').hidden = true;
        setEvaluating(false);
        mission(3, 'Six-round comparison ready', 1);
        const point = checkpoints.find(p => p.rounds === 6);
        if (!point || !point.disconnected || !point.weightsUnchanged) throw Error('Six-round frozen checkpoint missing or invalid');
        showFrozenScores(point.wins, point.byCue, 'Live result after exactly 6 training rounds');
        [$('mission-alone').textContent, $('mission-taught').textContent, $('mission-random').textContent] = point.wins.map(String);
        $('score-note').textContent = 'test after 6 rounds · wins out of 72';
        missionFinal = true; renderPlayback();
        $('compare-status').textContent = 'Six-round comparison ready. Continue the same students to 288 rounds for the full learning curves.';
        $('live-story').textContent = 'Six rounds complete · compare the test scores below, then continue training.';
        $('compare-continue').hidden = false;
        $('compare-continue').disabled = false;
        $('compare-continue').onclick = () => {
          $('compare-continue').hidden = true;
          $('compare-continue').disabled = true;
          $('compare-continue').onclick = null;
          setPlayback({ paused: false, step: 0 });
          activeRun?.control({ continueAfterSix: true });
          $('compare-status').textContent = 'Continuing the same matched students to 288 rounds; later frozen checkpoints extend the curves.';
        };
      }
      if (event.type === 'final') {
        $('compare-eval-loading').hidden = runKind !== 'live';
        setEvaluating(runKind === 'live');
        showPhase(-1);
        mission(4, runKind === 'live' ? 'Checking both 72-game tests' : 'Final test: Fly A disconnected', 1);
        $('live-story').textContent = 'Fly A is disconnected and learning is switched off. Both Fly Bs now play new games so we can compare what they retained.';
        $('compare-status').textContent = 'Frozen evaluation: both students disconnected, with no exploration or learning.';
      }
      if (event.type === 'done') {
        $('compare-eval-loading').hidden = true;
        setEvaluating(false);
        const { totals, byCue } = event;
        for (const prefix of ['alone', 'guided']) {
          const header = $(`${prefix}-brain-header`);
          header.textContent = header.textContent.replace(' spiking now', ' recent spikes in the last training frame');
        }
        showFrozenScores(totals, byCue, `${runKind === 'live' ? 'Live result' : 'Recorded result'} after ${total} training rounds`);
        $('compare-status').textContent = 'Live comparison complete.';
        const randomWins = checkpoints.find(point => point.rounds === total)?.wins[2] ?? null;
        $('mission-taught').textContent = totals[1];
        $('mission-alone').textContent = totals[0];
        $('mission-random').textContent = randomWins ?? '–';
        $('score-note').textContent = 'final test · wins out of 72';
        missionFinal = true; renderPlayback();
        $('teach-message').hidden = true;
        $('run-next-text').textContent = `Final test, Fly A unplugged: the taught Fly B won ${totals[1]} of 72 games. Learning alone: ${totals[0]}.${randomWins === null ? '' : ` Random messages: ${randomWins}.`}`;
        $('run-next-text').textContent += ' Same idea in the real world: a partner that already adapted sends a few bytes of advice, never its model.';
        $('run-next').hidden = false;
        mission(5, `Final test: Taught ${totals[1]}/72 · Alone ${totals[0]}/72${randomWins === null ? '' : ` · Random ${randomWins}/72`}`, 1);
        showPhase(-1);
        $('live-story').textContent = runKind === 'ruleSwitch' ?
          'This is one recorded run with Fly A unplugged. The scores show what each Fly B retained; Stats has the 15-run result.' :
          preview ? `${total}-round quick run complete · one frozen test with Fly A disconnected.` :
          'Fly A was disconnected for the final test. Compare both scores to see what each Fly B retained.';
        const run = runKind === 'live' ? comparisonReady?.run : null;
        if (run) {
          Object.assign(run, { studentRounds: total, taught: totals[1], alone: totals[0] });
          renderRuns();
        }
      }
  };
}
let activeRun = null;
// Prefers a background thread; without Worker support the same engine runs here.
function openStudentRun(params, control) {
  if (typeof Worker === 'undefined') {
    const stream = runStudents({ ...params, control, runCheckpoint: inlineCheckpoint });
    return { stream, control: update => Object.assign(control, update), stop() {} };
  }
  const worker = new Worker(new URL('./rps_student_worker.js', import.meta.url), { type: 'module' });
  const queue = []; let wake = null, ended = false;
  const finish = () => { ended = true; wake?.(); };
  const push = event => { queue.push(event); wake?.(); };
  worker.onmessage = ({ data }) => { if (data.type === 'closed') finish(); else push(data); };
  worker.onerror = event => { push({ type: 'error', message: event.message || 'Training thread failed' }); finish(); };
  worker.postMessage({ params, control });
  const stream = (async function* () {
    while (true) {
      if (queue.length) { yield queue.shift(); continue; }
      if (ended) return;
      await new Promise(resolve => { wake = resolve; });
      wake = null;
    }
  })();
  return { stream, control: update => worker.postMessage({ type: 'control', control: update }),
    stop() { worker.terminate(); finish(); } };
}
async function startComparison() {
  if (!comparisonReady || comparisonRunning || challengeRunning) return;
  const { sub, agent: teacherAgent, seed, config, points, coordinates, teacherBrain,
    featureMean, featureStd } = comparisonReady;
  const token = challengeToken, plan = studentPlan();
  comparisonRunning = true;
  runKind = 'live';
  $('compare-start').disabled = true;
  setPlayback({ active: true, paused: false, step: 0, where: '' });
  setChallengeDisabled(true);
  $('compare-status').textContent = 'Initializing two separate, matched untrained Fly B brains…';
  mission(3, 'Fly A is teaching Fly B', 0);
  try {
    const aloneBrain = brain('alone', coordinates, points, sub);
    const guidedBrain = brain('guided', coordinates, points, sub);
    $('alone-brain-view').hidden = $('guided-brain-view').hidden = false;
    const views = [teacherBrain, aloneBrain, guidedBrain];
    activeRun = openStudentRun({ sub, seed, config, teacherState: JSON.parse(snapshot(teacherAgent)),
      featureMean, featureStd, rounds: plan.rounds, preview: plan.preview, pauseAtSix: plan.pauseAtSix },
      { paused: playback.paused, speed: playback.speed, step: 0, continueAfterSix: false });
    const handle = studentEventHandler(views, { preview: plan.preview });
    for await (const event of activeRun.stream) {
      if (token !== challengeToken) return;
      handle(event);
      if (event.type === 'done') activeRun.stop();
    }
  } catch (error) {
    if (token === challengeToken) {
      $('compare-status').textContent = `Comparison stopped: ${error.message}`;
      mission(3, 'Stopped'); $('live-story').textContent = error.message;
    }
  } finally {
    activeRun = null;
    if (token === challengeToken) {
      $('compare-eval-loading').hidden = true;
      setEvaluating(false);
      comparisonRunning = false;
      setPlayback({ active: false, paused: false, step: 0 });
      setChallengeDisabled(false);
      renderPattern();
    }
  }
}

const RECORDINGS = { normal: 'data/demo_replay_strategy_a_830', ruleSwitch: 'data/demo_replay_rule_switch_s5' };
const replayCache = {};
let replayAssets = null; // connectome files shared by every replay start and seek
// Display time of each recorded event at 1× (the same holds the player uses); places events on the timeline.
// Plain-language cards for the recorded rule-change stages (titles come from the saved recording).
const STAGE_CARDS = {
  'Every Fly B learned the OLD rule': 'Before any teaching, all three Fly Bs spent 96 rounds learning the OLD rule (Rock → Paper → Scissors). They are identical copies, so the race that follows is fair.',
  'The opponent just changed its rule': "Now the opponent switches to the NEW rule (Rock → Scissors → Paper), so the Fly Bs' old habits start losing. Fly A already learned the new rule. Can it help one Fly B unlearn the old one faster than the others?",
};
const stagePause = event => STAGE_CARDS[event.title] ? Math.max(event.pause ?? 1500, 5000) : (event.pause ?? 1500);
const eventMs = (event, kind) => event.type === 'teacher' ? 35 : event.type === 'teacherDone' ? 2400 :
  event.type === 'frame' ? (kind === 'ruleSwitch' ? 16 : 9) : event.type === 'stage' ? stagePause(event) : 0;
// Moves the recorded run to event `index`, either now (if it is playing) or by restarting it paused there.
function requestSeek(index) {
  if (!replayNav) return;
  if (challengeRunning && replayNav.token === challengeToken) pendingSeek = index;
  else if (!challengeRunning && !comparisonRunning) playRecordedRun(replayNav.kind, index, true);
}
function showTimeline(position) {
  const timeline = $('timeline');
  timeline.hidden = !replayNav;
  if (!replayNav) return;
  timeline.max = String(replayNav.total);
  if (!timelineDragging) timeline.value = String(position);
}
// Plays a real saved run: every event, score, and spike comes from tools/record_demo_replay.mjs
// or tools/record_rule_switch_replay.mjs. Seeking replays the saved events up to the chosen point
// without waiting, so the scores, charts, and brains match that moment exactly.
async function playRecordedRun(kind = 'normal', startAt = 0, startPaused = false) {
  const RECORDING = RECORDINGS[kind] ?? RECORDINGS.normal;
  if (challengeRunning || comparisonRunning) return;
  const token = ++challengeToken;
  pendingSeek = null;
  resetComparison();
  $('teacher-details').hidden = true; $('teacher-details-hint').hidden = false; // recorded runs did not save Fly A's neuron details
  challengeRunning = true;
  setChallengeDisabled(true);
  $('compare-start').disabled = true;
  mission(1, 'Loading the recorded run');
  setPlayback({ active: true, paused: startPaused, step: 0, where: '' });
  try {
    replayCache[kind] ??= await Promise.all([data(`${RECORDING}.json`), data(`${RECORDING}.bin`, 'arrayBuffer')]);
    const [log, buffer] = replayCache[kind];
    replayAssets ??= Promise.all([data('data/subgraph.json'),
      data('data/brain_points.bin', 'arrayBuffer'), brainCoordinates3D()]);
    const [sub, pointBytes, coordinates] = await replayAssets;
    if (token !== challengeToken) return;
    const points = new Uint16Array(pointBytes);
    const views = [brain('challenge-teacher', coordinates, points, sub),
      brain('alone', coordinates, points, sub), brain('guided', coordinates, points, sub)];
    const events = log.events;
    const starts = [];
    let total = 0;
    for (const event of events) { starts.push(total); total += eventMs(event, kind); }
    replayNav = { kind, token, starts, total, steps: { 1: 0, 2: 0,
      3: Math.max(0, events.findIndex(event => event.type === 'stage')),
      4: Math.max(0, events.findIndex(event => event.type === 'final')) } };
    const labelRun = () => {
      $('alone-brain-view').hidden = $('guided-brain-view').hidden = false;
      $('mission-mode').textContent = 'Recorded run · real saved data';
      $('mission-mode').title = log.label;
      $('run-about').textContent = `About this run: ${log.label}.`;
    };
    labelRun();
    const nets = views.map(() => ({ t: 0, lastSpike: new Float64Array(sub.meta.n_neurons).fill(-1e9) }));
    const spikes = new Uint16Array(buffer);
    const applyFrame = (frame, render) => {
      frame.nets.forEach(({ view, t, offset, count, reset }) => {
        const net = nets[view];
        if (reset) net.lastSpike.fill(-1e9);
        net.t = t;
        for (let i = 0; i < count; i++) net.lastSpike[spikes[offset + i]] = t;
        if (render) views[view].frame(net);
      });
    };
    runKind = kind;
    let handle = studentEventHandler(views), teacherRound = 0;
    // live = false fast-forwards: same updates, no waiting, spikes drawn once at the end.
    const apply = async (event, live) => {
      if (event.type === 'teacher') {
        mission(2, event.frozen ? 'Fly A is taking its test' : 'Fly A is learning the game', event.progress);
        $('live-story').textContent = kind === 'ruleSwitch' ? (event.frozen ?
          'Fly A now plays new games with learning switched off. This check shows whether it learned the opponent’s new rule.' :
          'Fly A practices against the new rule, using its wins and losses to improve its next move.') : event.story;
        applyFrame(event, live);
        teacherRound++;
        playback.where = event.frozen ? `Fly A test round ${teacherRound - 288}/72` : `Fly A round ${teacherRound}/288`;
        $('teacher-badge').textContent = event.frozen ? `Fly A test · game ${teacherRound - 288} of 72 · no learning` :
          `Fly A is learning the new rule · round ${teacherRound}/288 (sped up)`;
        if (live) { await hold(35, token); roundDone(playback.where); }
      } else if (event.type === 'teacherDone') {
        mission(2, `Fly A scored ${event.wins}/72`, 1);
        $('teacher-badge').textContent = `Fly A passed its test: won ${event.wins} of 72 games`;
        $('compare-teacher-score').textContent = `Fly A: ${event.wins}/72 frozen wins.`;
        $('live-story').textContent = 'Fly A is frozen and ready to teach.';
        if (live) { await hold(2400, token); roundDone('Fly A test'); }
      } else if (event.type === 'frame') {
        applyFrame(event, live);
        if (live) await hold(kind === 'ruleSwitch' ? 16 : 9, token);
      } else if (event.type === 'stage') {
        mission(event.step, event.title, event.progress);
        $('live-story').textContent = event.title === 'Every Fly B learned the OLD rule' ?
          'All three Fly Bs practiced the old rule and start from identical copies. That gives them a fair starting point.' :
          event.title === 'The opponent just changed its rule' ?
            'The opponent reverses its rule. Fly A already knows the new pattern; we now watch whether its advice helps one Fly B adapt faster.' : event.story;
        const card = STAGE_CARDS[event.title];
        $('stage-card').hidden = !card;
        if (card) { $('stage-card-title').textContent = event.title; $('stage-card-text').textContent = card; }
        if (live) { await hold(stagePause(event), token); roundDone(event.title); }
      } else handle(event);
    };
    const seek = async index => {
      seeking = true;
      clearComparisonDisplay();
      labelRun();
      nets.forEach(net => { net.t = 0; net.lastSpike.fill(-1e9); });
      handle = studentEventHandler(views);
      teacherRound = 0;
      for (let k = 0; k < index; k++) await apply(events[k], false);
      views.forEach((view, i) => view.frame(nets[i]));
      seeking = false;
      followKey = ''; // the next step change scrolls to the right brain again
      showTimeline(starts[index] ?? total);
    };
    let i = Math.min(Math.max(0, startAt), events.length);
    if (i) await seek(i);
    showTimeline(starts[i] ?? total);
    while (i < events.length && token === challengeToken) {
      if (pendingSeek !== null) {
        i = Math.min(Math.max(0, pendingSeek), events.length);
        pendingSeek = null;
        await seek(i);
        continue;
      }
      await apply(events[i], true);
      if (pendingSeek !== null) continue;
      i++;
      showTimeline(starts[i] ?? total);
    }
  } catch (error) {
    seeking = false;
    if (token === challengeToken) { mission(1, 'Stopped'); $('live-story').textContent = error.message; }
  } finally {
    if (token === challengeToken) {
      challengeRunning = false;
      setPlayback({ active: false, paused: false, step: 0 });
      setChallengeDisabled(false);
      renderPattern();
    }
  }
}

if (typeof document !== 'undefined') {
  if ($('evidence-speed-chart')) loadSpeedEvidence();
  if ($('replay-load')) {
  $('replay-load').onclick = loadReplay;
  $('replay-seed').onchange = () => { if (replay) { replayRound = 1; showReplay(); } };
  $('replay-arm').onchange = () => { if (replay) { replayRound = 1; showReplay(); } };
  $('replay-checkpoint').onchange = () => { if (replay) showReplay(); };
  $('replay-prev').onclick = () => { if (replay) { replayRound = Math.max(0, replayRound - 1); showReplay(); } };
  $('replay-next').onclick = () => { if (replay) { replayRound = Math.min(11, replayRound + 1); showReplay(); } };
  }
  if ($('research-evidence')) {
  $('research-evidence').ontoggle = () => { if ($('research-evidence').open) loadActionPreferenceEvidence(); };
  $('preference-evidence-seed').onchange = loadActionPreferenceEvidence;
  $('ablation-evidence').ontoggle = () => { if ($('ablation-evidence').open) loadAblationEvidence(); };
  $('ablation-evidence-seed').onchange = loadAblationEvidence;
  $('challenge-train').onclick = runChallenge;
  addEventListener('synapse:play-rule-switch', event => {
    if (challengeRunning || comparisonRunning) return;
    setPlayback({ speed: event.detail?.speed === 4 ? 4 : 1 });
    void playRecordedRun('ruleSwitch');
  });
  const togglePlayback = () => { if (playback.active) setPlayback({ paused: !playback.paused, step: 0 }); };
  const stepPlayback = () => { if (playback.active) setPlayback({ paused: true, step: 1 }); };
  $('play-toggle').onclick = togglePlayback;
  const timelineIndex = value => {
    const { starts } = replayNav;
    let lo = 0, hi = starts.length - 1;
    while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (starts[mid] <= value) lo = mid; else hi = mid - 1; }
    return lo;
  };
  $('timeline').oninput = () => { if (replayNav) requestSeek(timelineIndex(Number($('timeline').value))); };
  $('timeline').onpointerdown = () => { timelineDragging = true; };
  $('timeline').onpointerup = $('timeline').onchange = () => { timelineDragging = false; };
  [1, 2, 3, 4].forEach(step => {
    $(`mission-step-${step}`).onclick = () => { if (replayNav) requestSeek(replayNav.steps[step]); };
  });
  $('play-step').onclick = stepPlayback;
  $('compare-start').onclick = startComparison;
  $('challenge-reset').onclick = challengeReset;
  $('challenge-opponent').onchange = challengeReset;
  $('challenge-teacher-rounds').onchange = challengeReset;
  $('challenge-student-rounds').onchange = showStudentPlan;
  $('challenge-fast').onchange = showStudentPlan;
  $('challenge-seed').onchange = challengeReset;
  $('brain-reset').onclick = () => brainViews.forEach(view => view.resetView?.());
  [0, 1, 2].forEach(action => {
    $(`challenge-add-${MOVES[action].toLowerCase()}`).onclick = () => {
      if (pattern.length < 5 && !challengeRunning) { pattern.push(action); challengeReset(); }
    };
  });
  challengeReset();
  if (typeof $('brain-stage')?.after === 'function') {
    $('preference-evidence-seed').value = 'b-11015';
    document.querySelectorAll('.role-toggle').forEach(button => {
      button.onclick = () => {
        const role = button.dataset.role, hide = !hiddenRoles.has(role);
        if (hide) hiddenRoles.add(role); else hiddenRoles.delete(role);
        button.setAttribute('aria-pressed', String(!hide));
        brainViews.forEach(view => view.setHiddenRoles?.(hiddenRoles));
      };
    });
    speedButtons = [...document.querySelectorAll('.speed-option')];
    speedButtons.forEach(button => { button.onclick = () => setPlayback({ speed: Number(button.dataset.speed) }); });
    for (const type of ['wheel', 'touchmove'])
      addEventListener(type, () => { if (playback.active) followOff = true; }, { passive: true });
    document.addEventListener('keydown', event => {
      if (['PageUp', 'PageDown', 'Home', 'End'].includes(event.key) && playback.active) followOff = true;
    });
    document.addEventListener('keydown', event => {
      if (!playback.active || event.ctrlKey || event.metaKey || event.altKey ||
          /^(INPUT|SELECT|TEXTAREA|BUTTON|SUMMARY)$/.test(event.target.tagName)) return;
      if (event.key === ' ' || event.key === 'k') { event.preventDefault(); togglePlayback(); }
      else if (event.key === 'ArrowRight' || event.key === '.') { event.preventDefault(); stepPlayback(); }
    });
    renderPlayback();
    loadBrainAnatomy();
  }
  }
}
