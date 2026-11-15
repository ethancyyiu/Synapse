// Study 2: when should 36 messages be sent? Protocol: docs/advice-timing-protocol.md.
// Same task, teacher, student, learning rule, message code and evaluation as
// tools/check_advice_budget_followup.mjs; only the timing of the 36 messages differs between arms.
// node tools/check_advice_timing.mjs SEED_INDEX [TAU]   (299 = pilot; 300-314 = final)
import { readFile, open, access, mkdir } from 'node:fs/promises';
import { Network, rng, wiring } from '../js/sim.js';
import { RpsContextualBandit, RpsEpisode } from '../js/rps.js';
import { PreferenceStudent, preferenceRound, comparisonRound, studentWeights } from '../js/rps_challenge_students.js';
import { frozenStudentCheckpoint } from '../js/rps_challenge_checkpoints.js';
import { challengeOpponent } from '../js/rps_challenge_opponent.js';

const s = Number(process.argv[2]);
const LOCKED_TAU = 0.2;                    // see protocol; the pilot may pass a candidate value
const TAU = process.argv[3] === undefined ? LOCKED_TAU : Number(process.argv[3]);
const pilot = s === 299;
if (!Number.isInteger(s) || (!pilot && (s < 300 || s > 314))) throw Error('Use pilot seed 299 or final seed 300-314');
if (!pilot && TAU !== LOCKED_TAU) throw Error('Final seeds must use the locked threshold');
const BUDGET = 36;
const dir = new URL('../data/advice_timing/', import.meta.url);
const output = new URL(pilot ? `pilot_seed_299_tau_${TAU}.json` : `seed_${s}.json`, dir);
try { await access(output); throw Error('Write-once result already exists'); }
catch (error) { if (error.code !== 'ENOENT') throw error; }
const read = async name => JSON.parse(await readFile(new URL('../data/' + name, import.meta.url)));
const sub = await read('subgraph.json');
const { featureMean, featureStd } = (await read('rps_contextual_bandit.json')).design;
const A = { mode: 'preset', offset: 1 }, B = { mode: 'preset', offset: 2 };
const teacherSeed = 26001 + 2 * s, studentSeed = 26002 + 2 * s;
const EPISODES = 2, CHECKS = [24, 48];
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
  try { await file.writeFile(JSON.stringify({ ...result, tau: TAU, runtimeSeconds: Math.round((Date.now() - started) / 1000) }, null, 1) + '\n'); }
  finally { await file.close(); }
  console.log(JSON.stringify({ s, status: result.status, teacherWins: result.teacherWins, requests: result.requestsSent, points: result.points }));
}

// 1. Teacher: identical to the follow-up.
const teacherNet = makeNet(teacherSeed);
const teacherAgent = new RpsContextualBandit(teacherNet, { seed: teacherSeed, featureMean, featureStd });
for (let e = 0; e < 12; e++) {
  const ep = new RpsEpisode(teacherNet, teacherAgent, challengeOpponent(B, 110000 + 100 * s + e), { rounds: 24, learn: true });
  while (!ep.done) {
    if (ep.t % 400 === 0) { const k = teacherSeed * 1e7 + e * 1000 + ep.t / 400; teacherNet.rand = rng(k + 1); teacherAgent.rand = rng(k + 501); }
    ep.step();
  }
}
teacherAgent.explore = false;
let teacherWins = 0;
for (let e = 0; e < 3; e++) {
  const ep = new RpsEpisode(teacherNet, teacherAgent, challengeOpponent(B, 170000 + 10 * s + e), { rounds: 24, learn: false });
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
  const games = [0, 1].map(() => challengeOpponent(A, 130000 + 100 * s + e));
  shared.game = games[0]; twin.game = games[1];
  for (let r = 0; r < 24; r++) {
    const round = comparisonRound(null, shared, twin, studentSeed * 1e7 + e * 1000 + r, true);
    while (!round.done) round.step();
  }
}
const start = studentWeights(shared.agent);
const teacher = { net: teacherNet, agent: teacherAgent, key: 0 };
const keys = (e, r) => [teacherSeed * 1e7 + (200 + e) * 1000 + r, studentSeed * 1e7 + (200 + e) * 1000 + r];
const switchGame = e => challengeOpponent(B, 150000 + 100 * s + e);

// 3. Trained teacher's preference tape.
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
const preferences = tape.map(row => row.preference);
const cued = tape.flatMap((row, i) => row.preference === null ? [] : [i]);
if (tape.length !== 48 || cued.length !== 46) throw Error('Unexpected advice opportunity count');
const fisherYates = (values, seed) => {
  const out = [...values], rand = rng(seed);
  for (let i = out.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [out[i], out[j]] = [out[j], out[i]]; }
  return out;
};
// Fixed schedules over the 46 observed-cue ordinals, set before training.
const schedules = {
  early36: new Set(Array.from({ length: BUDGET }, (_, i) => i)),
  spaced36: new Set(Array.from({ length: BUDGET }, (_, i) => Math.floor((i + .5) * 46 / BUDGET))),
  randomTime36: new Set(fisherYates(Array.from({ length: 46 }, (_, i) => i), 2026111600 + s).slice(0, BUDGET)),
};
if (Object.values(schedules).some(x => x.size !== BUDGET)) throw Error('Schedule size mismatch');
// Shuffled content for the yoked request control: the 46 trained preferences permuted across opportunities.
const permuted = fisherYates(cued.map(i => preferences[i]), 2026111700 + s);
const shuffledTape = [...preferences];
cued.forEach((i, k) => { shuffledTape[i] = permuted[k]; });

const tapeTeacher = values => ({ net: makeNet(teacherSeed), key: 0, agent: { i: 0, scores: [0, 0, 0],
  decide() { const p = values[this.i]; this.scores = [0, 1, 2].map(a => Number(a === p)); } } });
const T = () => tapeTeacher(preferences);
const teachers = { spaced36: T(), early36: T(), randomTime36: T(), requested: T(), requestedShuffled: tapeTeacher(shuffledTape), full46: T() };

// 4. Switch.
const armNames = ['alone', 'spaced36', 'early36', 'randomTime36', 'requested', 'requestedRandom', 'requestedShuffled', 'full46'];
const arms = Object.fromEntries(armNames.map(n => [n, student(start)]));
const twins = Array.from({ length: 5 }, () => student(start));
const makeOpponent = seed => challengeOpponent(B, 170000 + 10 * s + (seed - 51000));
const check = other => frozenStudentCheckpoint({ sub, students: [arms.alone, other], studentSeed, featureMean, featureStd, makeOpponent });
const points = [];
const messages = Object.fromEntries(armNames.filter(n => n !== 'alone').map(n => [n, 0]));
const measure = async rounds => {
  const point = { rounds, messages: { ...messages } };
  for (const name of armNames.filter(n => n !== 'alone')) {
    const result = await check(arms[name]);
    if (point.alone !== undefined && point.alone !== result.wins[0]) throw Error('Matched frozen checks disagreed');
    point.alone = result.wins[0]; point[name] = result.wins[1];
  }
  points.push(point);
  console.log(JSON.stringify({ s, ...point }));
};
// Request rule: Fly B asks for advice on an observed-cue round if, the last time it saw this cue, the gap
// between its two highest decision scores was below TAU (or it has not seen this cue yet), and fewer than
// 36 messages have been sent. Uses only Fly B's own past state; never the answer, rewards to come, or Fly A.
const lastGap = {};
const requestLog = [];
await measure(0);
let done = 0, cuedDone = 0;
for (let e = 0; e < EPISODES; e++) {
  for (const x of [...Object.values(arms), ...twins]) x.game = switchGame(e);
  for (let r = 0; r < 24; r++) {
    const [tk, sk] = keys(e, r);
    for (const t of Object.values(teachers)) { t.key = tk; t.agent.i = done; }
    const enabled = preferences[done] !== null, cue = tape[done].cue;
    const sel = name => enabled && schedules[name].has(cuedDone);
    const request = enabled && messages.requested < BUDGET && (lastGap[cue] === undefined || lastGap[cue] < TAU);
    requestLog.push({ round: done, cue, request, gapUsed: lastGap[cue] ?? null });
    const rounds = [
      preferenceRound(teachers.spaced36, arms.alone, arms.spaced36, sk, null, sel('spaced36')),
      preferenceRound(teachers.early36, twins[0], arms.early36, sk, null, sel('early36')),
      preferenceRound(teachers.randomTime36, twins[1], arms.randomTime36, sk, null, sel('randomTime36')),
      preferenceRound(teachers.requested, twins[2], arms.requested, sk, arms.requestedRandom, request),
      preferenceRound(teachers.requestedShuffled, twins[3], arms.requestedShuffled, sk, null, request),
      preferenceRound(teachers.full46, twins[4], arms.full46, sk, null, enabled),
    ];
    for (const round of rounds) while (!round.done) round.step();
    if (rounds.some(round => round.rows[0].cue !== cue)) throw Error('Cue did not reproduce the tape');
    const sentFor = [['spaced36', 0], ['early36', 1], ['randomTime36', 2], ['requested', 3], ['requestedShuffled', 4], ['full46', 5]];
    for (const [name, i] of sentFor) if (rounds[i].message.decoded !== null) messages[name]++;
    if (rounds[3].message.decoded !== null) messages.requestedRandom++;
    if (cue !== null) {
      const sc = [...rounds[3].rows[1].scores].sort((a, z) => z - a);
      lastGap[cue] = sc[0] - sc[1];
    }
    const base = studentWeights(arms.alone.agent);
    if (twins.some(x => studentWeights(x.agent) !== base)) throw Error('Alone copies diverged');
    if (enabled) cuedDone++;
    done++;
    if (CHECKS.includes(done)) await measure(done);
  }
}
const expected = { spaced36: 36, early36: 36, randomTime36: 36, full46: 46 };
if (cuedDone !== 46 || Object.entries(expected).some(([n, c]) => messages[n] !== c) ||
  messages.requestedRandom !== messages.requested || messages.requestedShuffled !== messages.requested)
  throw Error('Scheduled or decoded message count mismatch');
const students = [...Object.values(arms), ...twins];
await save({ status: 'completed', pilot, teacherSeed, studentSeed, teacherWins,
  schedules: Object.fromEntries(Object.entries(schedules).map(([k, v]) => [k, [...v].sort((a, z) => a - z)])),
  requestsSent: messages.requested, requestLog, messages, points,
  teacherFrozen: snapshot(teacherAgent) === teacherWeights,
  synapsesUnchanged: students.every(x => x.net.m.every(v => v === 1)) });
