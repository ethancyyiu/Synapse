// Fixed follow-up; see docs/advice-budget-followup-protocol.md. Write-once per seed.
// node tools/check_advice_budget_followup.mjs SEED_INDEX (98/99 pilot; 100-114 final)
import { readFile, open, access, mkdir } from 'node:fs/promises';
import { Network, rng, wiring } from '../js/sim.js';
import { RpsContextualBandit, RpsEpisode } from '../js/rps.js';
import { PreferenceStudent, comparisonRound, preferenceRound, studentWeights } from '../js/rps_challenge_students.js';
import { frozenStudentCheckpoint } from '../js/rps_challenge_checkpoints.js';
import { challengeOpponent } from '../js/rps_challenge_opponent.js';

const s = Number(process.argv[2]);
if (!Number.isInteger(s) || (s !== 98 && s !== 99 && (s < 100 || s > 114))) throw Error('Use pilot seed 98/99 or final seed 100-114');
const dir = new URL('../data/advice_budget_followup/', import.meta.url);
const output = new URL(`seed_${s}.json`, dir);
try { await access(output); throw Error('Write-once result already exists'); }
catch (error) { if (error.code !== 'ENOENT') throw error; }
const read = async name => JSON.parse(await readFile(new URL('../data/' + name, import.meta.url)));
const archived = s < 100 ? null : await read(`advice_budget/seed_${s}.json`);
if (archived && archived.status !== 'completed') throw Error('Prior seed did not complete');
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
if (archived && archived.teacherWins !== teacherWins) throw Error('Teacher gate disagreed with archived seed');
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
const budgets = [12, 24, 36, 46];
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
const shuffled = {};
for (const b of budgets) {
  const slots = [...positions[b]].sort((a, z) => a - z);
  const values = shuffle(slots.map(j => preferences[cued[j]]), 2026111515 + 100 * s + b);
  shuffled[b] = [...preferences];
  slots.forEach((j, i) => { shuffled[b][cued[j]] = values[i]; });
}

// The trained tape is obtained above from real Fly A. Replayed preferences keep content
// identical across arms; each recipient still receives measured DN spikes.
const tapeTeacher = values => ({ net: makeNet(teacherSeed), key: 0, agent: { i: 0, scores: [0, 0, 0],
  decide() { const p = values[this.i]; this.scores = [0, 1, 2].map(a => Number(a === p)); } } });
const teachers = Object.fromEntries(budgets.flatMap(b => [
  [`trained${b}`, tapeTeacher(preferences)], [`shuffled${b}`, tapeTeacher(shuffled[b])]]));

// 4. Switch. Every condition has a matched no-advice twin.
const arms = { alone: student(start) };
for (const b of budgets) for (const prefix of ['trained', 'random', 'shuffled'])
  arms[`${prefix}${b}`] = student(start);
const aloneTwins = Array.from({ length: 2 * budgets.length - 1 }, () => student(start));
const makeOpponent = seed => challengeOpponent(B, 170000 + 10 * s + (seed - 51000));
const check = other => frozenStudentCheckpoint({ sub, students: [arms.alone, other], studentSeed, featureMean, featureStd, makeOpponent });
const points = [];
const measure = async rounds => {
  const point = { rounds, messages: { ...messages } };
  for (const name of Object.keys(arms).filter(n => n !== 'alone')) {
    const result = await check(arms[name]);
    if (point.alone !== undefined && point.alone !== result.wins[0]) throw Error('Matched frozen checks disagreed');
    point.alone = result.wins[0]; point[name] = result.wins[1];
  }
  if (archived) {
    const old = archived.points.find(p => p.rounds === rounds);
    if (!old || old.alone !== point.alone || old.spaced12 !== point.trained12 ||
      old.random12 !== point.random12 || old.full !== point.trained46)
      throw Error('Replicated result disagreed with archive');
  }
  points.push(point);
  console.log(JSON.stringify({ s, ...point }));
};
const messages = Object.fromEntries(Object.keys(arms).filter(n => n !== 'alone').map(n => [n, 0]));
await measure(0);
let done = 0, cuedDone = 0;
for (let e = 0; e < EPISODES; e++) {
  for (const x of [...Object.values(arms), ...aloneTwins]) x.game = switchGame(e);
  for (let r = 0; r < 24; r++) {
    const [tk, sk] = keys(e, r);
    for (const t of Object.values(teachers)) { t.key = tk; t.agent.i = done; }
    const enabled = preferences[done] !== null;
    const rounds = budgets.flatMap((b, i) => {
      const selected = enabled && positions[b].has(cuedDone);
      return [preferenceRound(teachers[`trained${b}`], i ? aloneTwins[2 * i - 1] : arms.alone,
        arms[`trained${b}`], sk, arms[`random${b}`], selected),
      preferenceRound(teachers[`shuffled${b}`], aloneTwins[2 * i],
        arms[`shuffled${b}`], sk, null, selected)];
    });
    for (const round of rounds) while (!round.done) round.step();
    if (rounds.some(round => round.rows[0].cue !== tape[done].cue) ||
      budgets.some((b, i) => {
        const selected = enabled && positions[b].has(cuedDone);
        return rounds[2 * i].message.preference !== (selected ? preferences[done] : null) ||
          rounds[2 * i + 1].message.preference !== (selected ? shuffled[b][done] : null);
      }))
      throw Error('Live advice or cue did not reproduce the fixed tape and schedule');
    budgets.forEach((b, i) => {
      if (rounds[2 * i].message.decoded !== null) {
        messages[`trained${b}`]++; messages[`random${b}`]++;
      }
      if (rounds[2 * i + 1].message.decoded !== null) messages[`shuffled${b}`]++;
    });
    const base = studentWeights(arms.alone.agent);
    if (aloneTwins.some(x => studentWeights(x.agent) !== base))
      throw Error('Alone copies diverged');
    if (enabled) cuedDone++;
    done++;
    if (CHECKS.includes(done)) await measure(done);
  }
}
if (cuedDone !== 46 || Object.entries(messages).some(([name, count]) =>
  count !== Number(name.match(/\d+$/)[0])))
  throw Error('Scheduled or decoded message count mismatch');
const students = [...Object.values(arms), ...aloneTwins];
await save({ status: 'completed', pilot: s < 100, teacherSeed, studentSeed, teacherWins,
  schedule: Object.fromEntries(budgets.map(b => [b, [...positions[b]].sort((a, z) => a - z)])),
  shuffledChanged: Object.fromEntries(budgets.map(b => [b, cued.filter(i =>
    shuffled[b][i] !== preferences[i]).length])), messages, points,
  teacherFrozen: snapshot(teacherAgent) === teacherWeights,
  synapsesUnchanged: students.every(x => x.net.m.every(v => v === 1)) });
