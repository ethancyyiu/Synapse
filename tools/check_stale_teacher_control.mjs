// Stale-teacher control pilot; see docs/stale-teacher-control-protocol.md. Derived from check_rule_switch_cohort.mjs:
// the control arm is a teacher trained on the old rule (Strategy A). Write-once per seed.
// node tools/check_stale_teacher_control.mjs SEED_INDEX
import { readFile, open, access, mkdir } from 'node:fs/promises';
import { Network, rng, wiring } from '../js/sim.js';
import { RpsContextualBandit, RpsEpisode } from '../js/rps.js';
import { PreferenceStudent, comparisonRound, preferenceRound, studentWeights } from '../js/rps_challenge_students.js';
import { frozenStudentCheckpoint } from '../js/rps_challenge_checkpoints.js';
import { challengeOpponent } from '../js/rps_challenge_opponent.js';

const s = Number(process.argv[2]);
if (!Number.isInteger(s) || s < 0 || s > 17) throw Error('Use seed index 0-2 (pilot) or 3-17 (cohort)');
// Seeds 0-2 are the pilot; 3-17 are the locked cohort (docs/stale-teacher-cohort-protocol.md).
const dir = new URL(s <= 2 ? '../data/stale_teacher_control/' : '../data/stale_teacher_cohort/', import.meta.url);
const output = new URL(`seed_${s}.json`, dir);
try { await access(output); throw Error('Write-once result already exists'); }
catch (error) { if (error.code !== 'ENOENT') throw error; }
const read = async name => JSON.parse(await readFile(new URL('../data/' + name, import.meta.url)));
const sub = await read('subgraph.json');
const { featureMean, featureStd } = (await read('rps_contextual_bandit.json')).design;
const A = { mode: 'preset', offset: 1 }, B = { mode: 'preset', offset: 2 };
const teacherSeed = 27001 + 2 * s, studentSeed = 27002 + 2 * s;
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
  console.log(JSON.stringify({ s, status: result.status, teacherWins: result.teacherWins, staleTeacherWins: result.staleTeacherWins, trainedMinusStale48: result.trainedMinusStale48, points: result.points }));
}

// 1. Two teachers, same network seed and procedure: one trained on the new rule (B), one on the old rule (A).
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
const { net: teacherNet, agent: teacherAgent, wins: teacherWins } = trainTeacher(B, 210000, 270000);
if (teacherWins < 58) { await save({ status: 'stopped_teacher', teacherSeed, studentSeed, teacherWins }); process.exit(0); }
const { net: staleNet, agent: staleAgent, wins: staleTeacherWins } = trainTeacher(A, 230000, 280000);
if (staleTeacherWins < 58) { await save({ status: 'stopped_stale_teacher', teacherSeed, studentSeed, teacherWins, staleTeacherWins }); process.exit(0); }
const teacherWeights = snapshot(teacherAgent), staleWeights = snapshot(staleAgent);

// 2. Shared Strategy A history.
const shared = student(), twin = student();
for (let e = 0; e < 4; e++) {
  const games = [0, 1].map(() => challengeOpponent(A, 250000 + 100 * s + e));
  shared.game = games[0]; twin.game = games[1];
  for (let r = 0; r < 24; r++) {
    const round = comparisonRound(null, shared, twin, studentSeed * 1e7 + e * 1000 + r, true);
    while (!round.done) round.step();
  }
}
const start = studentWeights(shared.agent);
const teacher = { net: teacherNet, agent: teacherAgent, key: 0 };
const keys = (e, r) => [teacherSeed * 1e7 + (200 + e) * 1000 + r, studentSeed * 1e7 + (200 + e) * 1000 + r];
const switchGame = e => challengeOpponent(B, 290000 + 100 * s + e);

const staleTeacher = { net: staleNet, agent: staleAgent, key: 0 };
teacherNet.resetState(); staleNet.resetState();

// 4. Switch.
const arms = { alone: student(start), trained: student(start), stale: student(start) };
const alone2 = student(start);
const makeOpponent = seed => challengeOpponent(B, 270000 + 10 * s + (seed - 51000));
const check = other => frozenStudentCheckpoint({ sub, students: [arms.alone, other], studentSeed, featureMean, featureStd, makeOpponent });
const points = [];
const measure = async rounds => {
  const point = { rounds };
  for (const name of ['trained', 'stale']) {
    const result = await check(arms[name]);
    if (point.alone !== undefined && point.alone !== result.wins[0]) throw Error('Matched frozen checks disagreed');
    point.alone = result.wins[0]; point[name] = result.wins[1];
  }
  points.push(point);
  console.log(JSON.stringify({ s, ...point }));
};
await measure(0);
const messages = { trained: 0, stale: 0 }, stalePrefs = [0, 0, 0], trainedPrefs = [0, 0, 0];
let done = 0, bothSent = 0, agreed = 0;
for (let e = 0; e < EPISODES; e++) {
  for (const x of [...Object.values(arms), alone2]) x.game = switchGame(e);
  for (let r = 0; r < 24; r++) {
    const [tk, sk] = keys(e, r);
    teacher.key = staleTeacher.key = tk;
    const rounds = [preferenceRound(teacher, arms.alone, arms.trained, sk),
      preferenceRound(staleTeacher, alone2, arms.stale, sk)];
    for (const round of rounds) while (!round.done) round.step();
    const [tp, sp] = rounds.map(round => round.message.preference);
    if (tp !== null) { messages.trained++; trainedPrefs[tp]++; }
    if (sp !== null) { messages.stale++; stalePrefs[sp]++; }
    if (tp !== null && sp !== null) { bothSent++; if (tp === sp) agreed++; }
    if (studentWeights(alone2.agent) !== studentWeights(arms.alone.agent)) throw Error('Alone copies diverged');
    done++;
    if (CHECKS.includes(done)) await measure(done);
  }
}
const students = [...Object.values(arms), alone2];
const at = rounds => points.find(p => p.rounds === rounds);
await save({ status: 'completed', teacherSeed, studentSeed, teacherWins, staleTeacherWins, messages, points,
  trainedMinusStale48: at(48).trained - at(48).stale, staleMinusAlone48: at(48).stale - at(48).alone,
  preferences: { trained: trainedPrefs, stale: stalePrefs, bothSent, agreed },
  teachersFrozen: snapshot(teacherAgent) === teacherWeights && snapshot(staleAgent) === staleWeights,
  synapsesUnchanged: students.every(x => x.net.m.every(v => v === 1)) });
