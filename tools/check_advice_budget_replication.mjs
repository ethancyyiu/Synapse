// Study 5: fresh-seed replication of the 36-message result. Locked protocol:
// docs/advice-budget-replication-protocol.md. A copy of tools/check_advice_budget_followup.mjs restricted to
// alone, trained36, random36, shuffled36 and trained46. Write-once per seed.
// node tools/check_advice_budget_replication.mjs SEED_INDEX   (100 = reproduction check; 200-214 = final)
import { readFile, open, access, mkdir } from 'node:fs/promises';
import { Network, rng, wiring } from '../js/sim.js';
import { RpsContextualBandit, RpsEpisode } from '../js/rps.js';
import { PreferenceStudent, preferenceRound, comparisonRound, studentWeights } from '../js/rps_challenge_students.js';
import { frozenStudentCheckpoint } from '../js/rps_challenge_checkpoints.js';
import { challengeOpponent } from '../js/rps_challenge_opponent.js';

const s = Number(process.argv[2]);
if (!Number.isInteger(s) || (s !== 100 && (s < 200 || s > 214))) throw Error('Use check seed 100 or final seed 200-214');
const dir = new URL('../data/advice_budget_replication/', import.meta.url);
const output = new URL(s === 100 ? 'smoke_seed_100.json' : `seed_${s}.json`, dir);
try { await access(output); throw Error('Write-once result already exists'); }
catch (error) { if (error.code !== 'ENOENT') throw error; }
const read = async name => JSON.parse(await readFile(new URL('../data/' + name, import.meta.url)));
const archived = s === 100 ? await read('advice_budget_followup/seed_100.json') : null;
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
  try { await file.writeFile(JSON.stringify({ ...result, runtimeSeconds: Math.round((Date.now() - started) / 1000) }, null, 1) + '\n'); }
  finally { await file.close(); }
  console.log(JSON.stringify({ s, status: result.status, teacherWins: result.teacherWins, points: result.points }));
}

// 1. Teacher: identical procedure to the follow-up.
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
if (archived && archived.teacherWins !== teacherWins) throw Error('Teacher gate disagreed with the follow-up archive');
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
const preferences = tape.map(row => row.preference);
const cued = tape.flatMap((row, i) => row.preference === null ? [] : [i]);
if (tape.length !== 48 || cued.length !== 46) throw Error('Unexpected advice opportunity count');
const budgets = [36, 46];
const positions = Object.fromEntries(budgets.map(b => [b,
  new Set(Array.from({ length: b }, (_, i) => Math.floor((i + .5) * 46 / b)))]));
if (budgets.some(b => positions[b].size !== b)) throw Error('Spaced schedule repeats an opportunity');
const shuffle = (values, seed) => {
  const out = [...values], rand = rng(seed);
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1)); [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
};
const slots36 = [...positions[36]].sort((a, z) => a - z);
const values36 = shuffle(slots36.map(j => preferences[cued[j]]), 2026111515 + 100 * s + 36);
const shuffled36 = [...preferences];
slots36.forEach((j, i) => { shuffled36[cued[j]] = values36[i]; });

const tapeTeacher = values => ({ net: makeNet(teacherSeed), key: 0, agent: { i: 0, scores: [0, 0, 0],
  decide() { const p = values[this.i]; this.scores = [0, 1, 2].map(a => Number(a === p)); } } });
const teachers = { trained36: tapeTeacher(preferences), shuffled36: tapeTeacher(shuffled36), trained46: tapeTeacher(preferences) };

// 4. Switch. Each advice round has a matched no-advice twin, as in the follow-up.
const arms = { alone: student(start), trained36: student(start), random36: student(start), shuffled36: student(start), trained46: student(start) };
const aloneTwins = [student(start), student(start)];
const makeOpponent = seed => challengeOpponent(B, 170000 + 10 * s + (seed - 51000));
const check = other => frozenStudentCheckpoint({ sub, students: [arms.alone, other], studentSeed, featureMean, featureStd, makeOpponent });
const points = [];
const messages = { trained36: 0, random36: 0, shuffled36: 0, trained46: 0 };
const measure = async rounds => {
  const point = { rounds, messages: { ...messages } };
  for (const name of Object.keys(arms).filter(n => n !== 'alone')) {
    const result = await check(arms[name]);
    if (point.alone !== undefined && point.alone !== result.wins[0]) throw Error('Matched frozen checks disagreed');
    point.alone = result.wins[0]; point[name] = result.wins[1];
  }
  if (archived) {
    const old = archived.points.find(p => p.rounds === rounds);
    for (const name of ['alone', 'trained36', 'random36', 'shuffled36', 'trained46'])
      if (!old || old[name] !== point[name]) throw Error(`Did not reproduce the follow-up's ${name} at round ${rounds}`);
  }
  points.push(point);
  console.log(JSON.stringify({ s, ...point }));
};
await measure(0);
let done = 0, cuedDone = 0;
for (let e = 0; e < EPISODES; e++) {
  for (const x of [...Object.values(arms), ...aloneTwins]) x.game = switchGame(e);
  for (let r = 0; r < 24; r++) {
    const [tk, sk] = keys(e, r);
    for (const t of Object.values(teachers)) { t.key = tk; t.agent.i = done; }
    const enabled = preferences[done] !== null;
    const sel36 = enabled && positions[36].has(cuedDone), sel46 = enabled && positions[46].has(cuedDone);
    const rounds = [
      preferenceRound(teachers.trained36, arms.alone, arms.trained36, sk, arms.random36, sel36),
      preferenceRound(teachers.shuffled36, aloneTwins[0], arms.shuffled36, sk, null, sel36),
      preferenceRound(teachers.trained46, aloneTwins[1], arms.trained46, sk, null, sel46),
    ];
    for (const round of rounds) while (!round.done) round.step();
    if (rounds.some(round => round.rows[0].cue !== tape[done].cue) ||
      rounds[0].message.preference !== (sel36 ? preferences[done] : null) ||
      rounds[1].message.preference !== (sel36 ? shuffled36[done] : null) ||
      rounds[2].message.preference !== (sel46 ? preferences[done] : null))
      throw Error('Live advice or cue did not reproduce the fixed tape and schedule');
    if (rounds[0].message.decoded !== null) { messages.trained36++; messages.random36++; }
    if (rounds[1].message.decoded !== null) messages.shuffled36++;
    if (rounds[2].message.decoded !== null) messages.trained46++;
    const base = studentWeights(arms.alone.agent);
    if (aloneTwins.some(x => studentWeights(x.agent) !== base)) throw Error('Alone copies diverged');
    if (enabled) cuedDone++;
    done++;
    if (CHECKS.includes(done)) await measure(done);
  }
}
if (cuedDone !== 46 || messages.trained36 !== 36 || messages.random36 !== 36 || messages.shuffled36 !== 36 || messages.trained46 !== 46)
  throw Error('Scheduled or decoded message count mismatch');
const students = [...Object.values(arms), ...aloneTwins];
await save({ status: 'completed', check: s === 100, teacherSeed, studentSeed, teacherWins,
  schedule: { 36: slots36, 46: [...positions[46]].sort((a, z) => a - z) },
  shuffledChanged: cued.filter(i => shuffled36[i] !== preferences[i]).length, messages, points,
  teacherFrozen: snapshot(teacherAgent) === teacherWeights,
  synapsesUnchanged: students.every(x => x.net.m.every(v => v === 1)) });
