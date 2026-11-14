// Fault-response task pilot; see docs/fault-response-pilot-protocol.md. Derived from check_rule_switch_cohort.mjs:
// the same five-arm design on a second task (random alarm order, a partial rule change). Write-once per seed.
// node tools/check_fault_response_pilot.mjs SEED_INDEX
import { readFile, open, access, mkdir } from 'node:fs/promises';
import { Network, rng, wiring } from '../js/sim.js';
import { RpsContextualBandit, RpsEpisode, result } from '../js/rps.js';
import { PreferenceStudent, comparisonRound, preferenceRound, studentWeights } from '../js/rps_challenge_students.js';
import { frozenStudentCheckpoint } from '../js/rps_challenge_checkpoints.js';

const s = Number(process.argv[2]);
if (!Number.isInteger(s) || !(s >= 0 && s <= 2 || s >= 10 && s <= 12 || s >= 20 && s <= 34))
  throw Error('Use seed index 0-2 (partial change), 10-12 (full-change pilot), or 20-34 (full-change cohort)');
// Seeds 0-2: docs/fault-response-pilot-protocol.md. Seeds 10-12: docs/fault-response-full-change-protocol.md.
// Seeds 20-34: docs/fault-response-cohort-protocol.md.
const FULL = s >= 10;
const dir = new URL(s >= 20 ? '../data/fault_response_cohort/' : FULL ? '../data/fault_response_full_change/' : '../data/fault_response_pilot/', import.meta.url);
const output = new URL(`seed_${s}.json`, dir);
try { await access(output); throw Error('Write-once result already exists'); }
catch (error) { if (error.code !== 'ENOENT') throw error; }
const read = async name => JSON.parse(await readFile(new URL('../data/' + name, import.meta.url)));
const sub = await read('subgraph.json');
const { featureMean, featureStd } = (await read('rps_contextual_bandit.json')).design;
// One of three alarms per round, in random order. need[alarm] plays the part of the opponent's move in result():
// the response need + 1 fixes the fault (+1), need itself does nothing (0), and the third makes it worse (-1).
const CHANGED_ALARM = 1;
// Fixing responses 1, 1, 2 before the change; after it 1, 0, 2 (partial) or 2, 2, 0 (full).
const A = [0, 0, 1], B = FULL ? [1, 1, 2] : [0, 2, 1];
class AlarmGame {
  constructor(need, seed) { this.need = need; this.rand = rng(seed); this.strategy = 'alarm'; this.draw(); }
  draw() { this.alarm = Math.floor(this.rand() * 3); }
  observe() { return { previousPlayer: null, previousOpponent: this.alarm, previousResult: null }; }
  play(response) {
    const opponent = this.need[this.alarm], reward = result(response, opponent);
    this.draw();
    return { opponent, reward };
  }
}
const challengeOpponent = (need, seed) => new AlarmGame(need, seed);
const teacherSeed = 30001 + 2 * s, studentSeed = 30002 + 2 * s;
const EPISODES = 4, CHECKS = [24, 48, 96];
const makeNet = seed => new Network(sub, wiring(sub, 'real', seed), { seed });
const student = weights => {
  const net = makeNet(studentSeed), agent = new PreferenceStudent(net, studentSeed, featureMean, featureStd);
  if (weights) {
    const saved = JSON.parse(weights);
    saved.heads.forEach((h, i) => { h.W.forEach((w, a) => agent.heads[i].W[a].set(w)); agent.heads[i].b.set(h.b); });
    agent.baseline = saved.baseline;
  }
  agent.explore = true;
  return { net, agent, game: null };
};
const snapshot = a => JSON.stringify({ W: a.W.map(w => [...w]), b: [...a.b], baseline: a.baseline });

async function save(result) {
  await mkdir(dir, { recursive: true });
  const file = await open(output, 'wx');
  try { await file.writeFile(JSON.stringify(result, null, 1) + '\n'); } finally { await file.close(); }
  console.log(JSON.stringify({ s, status: result.status, teacherWins: result.teacherWins, advantage48: result.advantage48, points: result.points }));
}

// 1. Teacher (same procedure as the pilot, new seeds).
const teacherNet = makeNet(teacherSeed);
const teacherAgent = new RpsContextualBandit(teacherNet, { seed: teacherSeed, featureMean, featureStd });
for (let e = 0; e < 12; e++) {
  const ep = new RpsEpisode(teacherNet, teacherAgent, challengeOpponent(B, 710000 + 100 * s + e), { rounds: 24, learn: true });
  while (!ep.done) {
    if (ep.t % 400 === 0) { const k = teacherSeed * 1e7 + e * 1000 + ep.t / 400; teacherNet.rand = rng(k + 1); teacherAgent.rand = rng(k + 501); }
    ep.step();
  }
}
teacherAgent.explore = false;
let teacherWins = 0;
for (let e = 0; e < 3; e++) {
  const ep = new RpsEpisode(teacherNet, teacherAgent, challengeOpponent(B, 770000 + 10 * s + e), { rounds: 24, learn: false });
  while (!ep.done) {
    if (ep.t % 400 === 0) { const k = teacherSeed * 1e7 + (100 + e) * 1000 + ep.t / 400; teacherNet.rand = rng(k + 1); teacherAgent.rand = rng(k + 501); }
    ep.step();
  }
  teacherWins += ep.events.filter(event => event.reward === 1).length;
}
if (teacherWins < 58) { await save({ status: 'stopped_teacher', teacherSeed, studentSeed, teacherWins }); process.exit(0); }
const teacherWeights = snapshot(teacherAgent);

// 2. Shared Strategy A history.
const shared = student(), twin = student();
for (let e = 0; e < 4; e++) {
  const games = [0, 1].map(() => challengeOpponent(A, 730000 + 100 * s + e));
  shared.game = games[0]; twin.game = games[1];
  for (let r = 0; r < 24; r++) {
    const round = comparisonRound(null, shared, twin, studentSeed * 1e7 + e * 1000 + r, true);
    while (!round.done) round.step();
  }
}
const start = studentWeights(shared.agent);
const teacher = { net: teacherNet, agent: teacherAgent, key: 0 };
const keys = (e, r) => [teacherSeed * 1e7 + (200 + e) * 1000 + r, studentSeed * 1e7 + (200 + e) * 1000 + r];
const switchGame = e => challengeOpponent(B, 750000 + 100 * s + e);

// 3. Dry run records the trained teacher's preference tape.
teacherNet.resetState();
const tape = [];
{
  const a = student(start), b = student(start);
  for (let e = 0; e < EPISODES; e++) {
    a.game = switchGame(e); b.game = switchGame(e);
    for (let r = 0; r < 24; r++) {
      const [tk, sk] = keys(e, r);
      teacher.key = tk;
      const round = preferenceRound(teacher, a, b, sk);
      while (!round.done) round.step();
      tape.push({ cue: round.rows[0].cue, preference: round.message.preference });
    }
  }
}
teacherNet.resetState();
const cued = tape.flatMap((row, i) => row.preference === null ? [] : [i]);
const values = cued.map(i => tape[i].preference), shuffleRng = rng(2026111120 + 2 * s);
for (let i = values.length - 1; i > 0; i--) {
  const j = Math.floor(shuffleRng() * (i + 1)); [values[i], values[j]] = [values[j], values[i]];
}
const shuffledTape = Array(tape.length).fill(null);
cued.forEach((index, i) => { shuffledTape[index] = values[i]; });

// Stand-in teacher that reports a fixed preference for the current round.
const tapeTeacher = targets => ({ net: makeNet(teacherSeed), key: 0, agent: { i: 0, scores: [0, 0, 0],
  decide() { const p = targets[this.i]; this.scores = [0, 1, 2].map(a => Number(a === p)); } } });
const shuffledTeacher = tapeTeacher(shuffledTape);
const untrainedNet = makeNet(teacherSeed);
const untrainedAgent = new RpsContextualBandit(untrainedNet, { seed: teacherSeed, featureMean, featureStd, readoutLearn: false });
untrainedAgent.explore = false;
const untrainedWeights = snapshot(untrainedAgent);
const untrainedTeacher = { net: untrainedNet, agent: untrainedAgent, key: 0 };

// 4. Switch.
const arms = { alone: student(start), trained: student(start), random: student(start),
  untrained: student(start), shuffled: student(start) };
const alone2 = student(start), alone3 = student(start);
const makeOpponent = seed => challengeOpponent(B, 770000 + 10 * s + (seed - 51000));
const check = other => frozenStudentCheckpoint({ sub, students: [arms.alone, other], studentSeed, featureMean, featureStd, makeOpponent });
const points = [];
const measure = async rounds => {
  const point = { rounds };
  for (const name of ['trained', 'random', 'untrained', 'shuffled']) {
    const result = await check(arms[name]);
    if (point.alone !== undefined && point.alone !== result.wins[0]) throw Error('Matched frozen checks disagreed');
    point.alone = result.wins[0]; point[name] = result.wins[1];
    // Frozen wins on the alarm whose fixing response changed, and on the two that did not.
    const split = i => ({ changed: result.byCue[i][CHANGED_ALARM].wins, changedRounds: result.byCue[i][CHANGED_ALARM].n,
      unchanged: result.wins[i] - result.byCue[i][CHANGED_ALARM].wins });
    point.byAlarm ??= {};
    point.byAlarm.alone = split(0); point.byAlarm[name] = split(1);
  }
  points.push(point);
  console.log(JSON.stringify({ s, ...point, byAlarm: undefined }));
};
await measure(0);
const messages = { trained: 0, random: 0, untrained: 0, shuffled: 0 }, untrainedPrefs = [0, 0, 0];
let done = 0;
for (let e = 0; e < EPISODES; e++) {
  for (const x of [...Object.values(arms), alone2, alone3]) x.game = switchGame(e);
  for (let r = 0; r < 24; r++) {
    const [tk, sk] = keys(e, r);
    teacher.key = untrainedTeacher.key = shuffledTeacher.key = tk;
    shuffledTeacher.agent.i = done;
    const rounds = [preferenceRound(teacher, arms.alone, arms.trained, sk, arms.random),
      preferenceRound(untrainedTeacher, alone2, arms.untrained, sk),
      preferenceRound(shuffledTeacher, alone3, arms.shuffled, sk)];
    for (const round of rounds) while (!round.done) round.step();
    if (rounds[0].message.preference !== tape[done].preference || rounds[0].rows[0].cue !== tape[done].cue ||
        rounds[2].message.preference !== shuffledTape[done])
      throw Error('Live preferences did not reproduce the recorded tape');
    if (rounds[0].message.preference !== null) { messages.trained++; messages.random++; }
    if (rounds[1].message.preference !== null) { messages.untrained++; untrainedPrefs[rounds[1].message.preference]++; }
    if (rounds[2].message.preference !== null) messages.shuffled++;
    const base = studentWeights(arms.alone.agent);
    if (studentWeights(alone2.agent) !== base || studentWeights(alone3.agent) !== base)
      throw Error('Alone copies diverged');
    done++;
    if (CHECKS.includes(done)) await measure(done);
  }
}
const students = [...Object.values(arms), alone2, alone3];
const at = rounds => points.find(p => p.rounds === rounds);
const advantage = p => p.trained - Math.max(p.alone, p.random, p.untrained, p.shuffled);
const recovered = points.find(p => p.rounds > 0 && p.trained - p.alone <= 4);
await save({ status: 'completed', teacherSeed, studentSeed, teacherWins, messages, points,
  advantage48: advantage(at(48)), advantageByCheckpoint: points.map(p => ({ rounds: p.rounds, advantage: advantage(p) })),
  aloneWithin4AtRounds: recovered ? recovered.rounds : null,
  tape: { trainedPreferences: [0, 1, 2].map(a => values.filter(v => v === a).length),
    shuffledMoved: cued.filter(i => shuffledTape[i] !== tape[i].preference).length, cuedRounds: cued.length,
    untrainedPreferences: untrainedPrefs },
  teachersFrozen: snapshot(teacherAgent) === teacherWeights && snapshot(untrainedAgent) === untrainedWeights,
  synapsesUnchanged: students.every(x => x.net.m.every(v => v === 1)) });
