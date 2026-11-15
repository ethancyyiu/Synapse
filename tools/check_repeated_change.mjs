// Study 4: does teaching survive a second rule change? Protocol: docs/repeated-change-protocol.md.
// Rule A history -> change 1 to rule B (48 rounds) -> change 2 back to rule A (48 rounds).
// Same task machinery, student, learning rule, message code and evaluation as the rule-switch studies.
// node tools/check_repeated_change.mjs SEED_INDEX   (20 = implementation check; 0-14 = final)
import { readFile, open, access, mkdir } from 'node:fs/promises';
import { Network, rng, wiring } from '../js/sim.js';
import { RpsContextualBandit, RpsEpisode } from '../js/rps.js';
import { PreferenceStudent, preferenceRound, comparisonRound, studentWeights } from '../js/rps_challenge_students.js';
import { frozenStudentCheckpoint } from '../js/rps_challenge_checkpoints.js';
import { challengeOpponent } from '../js/rps_challenge_opponent.js';

const s = Number(process.argv[2]);
if (!Number.isInteger(s) || (s !== 20 && (s < 0 || s > 14))) throw Error('Use check seed 20 or final seed 0-14');
const dir = new URL('../data/repeated_change/', import.meta.url);
const output = new URL(s === 20 ? 'smoke_seed_20.json' : `seed_${s}.json`, dir);
try { await access(output); throw Error('Write-once result already exists'); }
catch (error) { if (error.code !== 'ENOENT') throw error; }
const read = async name => JSON.parse(await readFile(new URL('../data/' + name, import.meta.url)));
const sub = await read('subgraph.json');
const { featureMean, featureStd } = (await read('rps_contextual_bandit.json')).design;
const A = { mode: 'preset', offset: 1 }, B = { mode: 'preset', offset: 2 };
const teacherSeed = 33001 + 2 * s, studentSeed = 33002 + 2 * s;
const PHASE_ROUNDS = 48, CHECKS = [48, 72, 96];
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
const started = Date.now();
async function save(result) {
  await mkdir(dir, { recursive: true });
  const file = await open(output, 'wx');
  try { await file.writeFile(JSON.stringify({ ...result, runtimeSeconds: Math.round((Date.now() - started) / 1000) }, null, 1) + '\n'); }
  finally { await file.close(); }
  console.log(JSON.stringify({ s, status: result.status, teacherBWins: result.teacherBWins, teacherAWins: result.teacherAWins, points: result.points }));
}

// 1. Two teachers on the same network seed: one learns rule B (current in phase 1), one rule A (current in phase 2).
const trainTeacher = (rule, trainBase, gateBase) => {
  const net = makeNet(teacherSeed);
  const agent = new RpsContextualBandit(net, { seed: teacherSeed, featureMean, featureStd });
  for (let e = 0; e < 12; e++) {
    const ep = new RpsEpisode(net, agent, challengeOpponent(rule, trainBase + 100 * s + e), { rounds: 24, learn: true });
    while (!ep.done) {
      if (ep.t % 400 === 0) { const k = teacherSeed * 1e7 + e * 1000 + ep.t / 400; net.rand = rng(k + 1); agent.rand = rng(k + 501); }
      ep.step();
    }
  }
  agent.explore = false;
  let wins = 0;
  for (let e = 0; e < 3; e++) {
    const ep = new RpsEpisode(net, agent, challengeOpponent(rule, gateBase + 10 * s + e), { rounds: 24, learn: false });
    while (!ep.done) {
      if (ep.t % 400 === 0) { const k = teacherSeed * 1e7 + (100 + e) * 1000 + ep.t / 400; net.rand = rng(k + 1); agent.rand = rng(k + 501); }
      ep.step();
    }
    wins += ep.events.filter(event => event.reward === 1).length;
  }
  return { net, agent, wins };
};
const tB = trainTeacher(B, 1110000, 1170000);
if (tB.wins < 58) { await save({ status: 'stopped_teacher_B', teacherSeed, studentSeed, teacherBWins: tB.wins }); process.exit(0); }
const tA = trainTeacher(A, 1130000, 1180000);
if (tA.wins < 58) { await save({ status: 'stopped_teacher_A', teacherSeed, studentSeed, teacherBWins: tB.wins, teacherAWins: tA.wins }); process.exit(0); }
const weightsB = snapshot(tB.agent), weightsA = snapshot(tA.agent);

// 2. Shared rule-A history (as in the rule-switch studies).
const shared = student(), twin = student();
for (let e = 0; e < 4; e++) {
  const games = [0, 1].map(() => challengeOpponent(A, 1150000 + 100 * s + e));
  shared.game = games[0]; twin.game = games[1];
  for (let r = 0; r < 24; r++) {
    const round = comparisonRound(null, shared, twin, studentSeed * 1e7 + e * 1000 + r, true);
    while (!round.done) round.step();
  }
}
const start = studentWeights(shared.agent);
// Global rounds 0-47: rule B; 48-95: rule A. Episodes are 24 rounds, two per phase.
const ruleOf = e => e < 2 ? B : A;
const keys = (e, r) => [teacherSeed * 1e7 + (200 + e) * 1000 + r, studentSeed * 1e7 + (200 + e) * 1000 + r];
const game = e => challengeOpponent(ruleOf(e), 1190000 + 100 * s + e);

// 3. Dry run: record each phase's current teacher preferences (frozen teachers; cue comes from the opponent).
const live = { 0: { net: tB.net, agent: tB.agent, key: 0 }, 1: { net: tA.net, agent: tA.agent, key: 0 } };
tB.net.resetState(); tA.net.resetState();
const tape = [];
{
  const a = student(start), b = student(start);
  for (let e = 0; e < 4; e++) {
    a.game = game(e); b.game = game(e);
    const t = live[e < 2 ? 0 : 1];
    for (let r = 0; r < 24; r++) {
      const [tk, sk] = keys(e, r);
      t.key = tk;
      const round = preferenceRound(t, a, b, sk);
      while (!round.done) round.step();
      tape.push({ cue: round.rows[0].cue, preference: round.message.preference });
    }
  }
}
const preferences = tape.map(row => row.preference);
// Shuffled control: the current teacher's preferences permuted among observed-cue rounds within each phase.
const fisherYates = (values, seed) => {
  const out = [...values], rand = rng(seed);
  for (let i = out.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [out[i], out[j]] = [out[j], out[i]]; }
  return out;
};
const shuffledTape = [...preferences];
for (const [lo, hi, seed] of [[0, 48, 2026111800 + s], [48, 96, 2026111900 + s]]) {
  const idx = [];
  for (let i = lo; i < hi; i++) if (preferences[i] !== null) idx.push(i);
  const values = fisherYates(idx.map(i => preferences[i]), seed);
  idx.forEach((i, k) => { shuffledTape[i] = values[k]; });
}
const tapeTeacher = values => ({ net: makeNet(teacherSeed), key: 0, agent: { i: 0, scores: [0, 0, 0],
  decide() { const p = values[this.i]; this.scores = [0, 1, 2].map(a => Number(a === p)); } } });
const teachers = { trained: tapeTeacher(preferences), shuffled: tapeTeacher(shuffledTape) };

// 4. Two changes. Every learner gets its own reward every round.
const arms = { alone: student(start), trained: student(start), random: student(start), shuffled: student(start) };
const aloneTwin = student(start);
const checkOpponent = rule => seed => challengeOpponent(rule, (rule === B ? 1170000 : 1200000) + 10 * s + (seed - 51000));
const points = [];
const messages = { trained: 0, random: 0, shuffled: 0 };
const measure = async (rounds, rule) => {
  const point = { rounds, rule: rule === B ? 'B' : 'A', messages: { ...messages } };
  for (const name of ['trained', 'random', 'shuffled']) {
    const result = await frozenStudentCheckpoint({ sub, students: [arms.alone, arms[name]], studentSeed, featureMean, featureStd, makeOpponent: checkOpponent(rule) });
    if (point.alone !== undefined && point.alone !== result.wins[0]) throw Error('Matched frozen checks disagreed');
    point.alone = result.wins[0]; point[name] = result.wins[1];
  }
  points.push(point);
  console.log(JSON.stringify({ s, ...point }));
};
await measure(0, B);
let done = 0;
for (let e = 0; e < 4; e++) {
  for (const x of [...Object.values(arms), aloneTwin]) x.game = game(e);
  for (let r = 0; r < 24; r++) {
    const [tk, sk] = keys(e, r);
    for (const t of Object.values(teachers)) { t.key = tk; t.agent.i = done; }
    const rounds = [
      preferenceRound(teachers.trained, arms.alone, arms.trained, sk, arms.random),
      preferenceRound(teachers.shuffled, aloneTwin, arms.shuffled, sk),
    ];
    for (const round of rounds) while (!round.done) round.step();
    if (rounds.some(round => round.rows[0].cue !== tape[done].cue) ||
      rounds[0].message.preference !== preferences[done] || rounds[1].message.preference !== shuffledTape[done])
      throw Error('Live advice or cue did not reproduce the fixed tape');
    if (rounds[0].message.decoded !== null) { messages.trained++; messages.random++; }
    if (rounds[1].message.decoded !== null) messages.shuffled++;
    if (studentWeights(aloneTwin.agent) !== studentWeights(arms.alone.agent)) throw Error('Alone copies diverged');
    done++;
    if (CHECKS.includes(done)) await measure(done, done <= PHASE_ROUNDS ? B : A);
  }
}
const expected = preferences.filter(p => p !== null).length;
if (Object.values(messages).some(c => c !== expected)) throw Error('Decoded message count mismatch');
const students = [...Object.values(arms), aloneTwin];
await save({ status: 'completed', check: s === 20, teacherSeed, studentSeed, teacherBWins: tB.wins, teacherAWins: tA.wins,
  messagesPerArm: expected, messages, points,
  teachersFrozen: snapshot(tB.agent) === weightsB && snapshot(tA.agent) === weightsA,
  synapsesUnchanged: students.every(x => x.net.m.every(v => v === 1)) });
