// Trust-gate pilot; see docs/trust-gate-pilot-protocol.md. Derived from check_stale_teacher_control.mjs:
// adds two trust-gated students that hear the same new-rule and old-rule teachers. Write-once per seed.
// node tools/check_trust_gate_pilot.mjs SEED_INDEX
import { readFile, open, access, mkdir } from 'node:fs/promises';
import { Network, rng, wiring } from '../js/sim.js';
import { RpsContextualBandit, RpsEpisode } from '../js/rps.js';
import { PreferenceStudent, TrustGatedStudent, comparisonRound, preferenceRound, studentWeights } from '../js/rps_challenge_students.js';
import { frozenStudentCheckpoint } from '../js/rps_challenge_checkpoints.js';
import { challengeOpponent } from '../js/rps_challenge_opponent.js';

const s = Number(process.argv[2]);
if (!Number.isInteger(s) || s < 0 || s > 17) throw Error('Use seed index 0-2 (pilot) or 3-17 (cohort)');
// Seeds 0-2 are the pilot; 3-17 are the locked cohort (docs/trust-gate-cohort-protocol.md).
const dir = new URL(s <= 2 ? '../data/trust_gate_pilot/' : '../data/trust_gate_cohort/', import.meta.url);
const output = new URL(`seed_${s}.json`, dir);
try { await access(output); throw Error('Write-once result already exists'); }
catch (error) { if (error.code !== 'ENOENT') throw error; }
const read = async name => JSON.parse(await readFile(new URL('../data/' + name, import.meta.url)));
const sub = await read('subgraph.json');
const { featureMean, featureStd } = (await read('rps_contextual_bandit.json')).design;
const A = { mode: 'preset', offset: 1 }, B = { mode: 'preset', offset: 2 };
const teacherSeed = 28001 + 2 * s, studentSeed = 28002 + 2 * s;
const EPISODES = 4, CHECKS = [24, 48, 96];
const makeNet = seed => new Network(sub, wiring(sub, 'real', seed), { seed });
const student = (weights, Kind = PreferenceStudent) => {
  const net = makeNet(studentSeed), agent = new Kind(net, studentSeed, featureMean, featureStd);
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
  console.log(JSON.stringify({ s, status: result.status, teacherWins: result.teacherWins, staleTeacherWins: result.staleTeacherWins, verdict: result.verdict, points: result.points }));
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
const { net: teacherNet, agent: teacherAgent, wins: teacherWins } = trainTeacher(B, 410000, 470000);
if (teacherWins < 58) { await save({ status: 'stopped_teacher', teacherSeed, studentSeed, teacherWins }); process.exit(0); }
const { net: staleNet, agent: staleAgent, wins: staleTeacherWins } = trainTeacher(A, 430000, 480000);
if (staleTeacherWins < 58) { await save({ status: 'stopped_stale_teacher', teacherSeed, studentSeed, teacherWins, staleTeacherWins }); process.exit(0); }
const teacherWeights = snapshot(teacherAgent), staleWeights = snapshot(staleAgent);

// 2. Shared Strategy A history.
const shared = student(), twin = student();
for (let e = 0; e < 4; e++) {
  const games = [0, 1].map(() => challengeOpponent(A, 450000 + 100 * s + e));
  shared.game = games[0]; twin.game = games[1];
  for (let r = 0; r < 24; r++) {
    const round = comparisonRound(null, shared, twin, studentSeed * 1e7 + e * 1000 + r, true);
    while (!round.done) round.step();
  }
}
const start = studentWeights(shared.agent);
const teacher = { net: teacherNet, agent: teacherAgent, key: 0 };
const keys = (e, r) => [teacherSeed * 1e7 + (200 + e) * 1000 + r, studentSeed * 1e7 + (200 + e) * 1000 + r];
const switchGame = e => challengeOpponent(B, 490000 + 100 * s + e);

const staleTeacher = { net: staleNet, agent: staleAgent, key: 0 };
teacherNet.resetState(); staleNet.resetState();
// Frozen copies of both teachers for the gated arms, so every arm hears a teacher in the same state.
const copyTeacher = from => {
  const net = makeNet(teacherSeed);
  const agent = new RpsContextualBandit(net, { seed: teacherSeed, featureMean, featureStd });
  from.W.forEach((w, a) => agent.W[a].set(w)); agent.b.set(from.b); agent.baseline = from.baseline;
  agent.explore = false; net.resetState();
  return { net, agent, key: 0 };
};
const teacherCopy = copyTeacher(teacherAgent), staleCopy = copyTeacher(staleAgent);

// 4. Switch.
const arms = { alone: student(start), trained: student(start), stale: student(start),
  gatedTrained: student(start, TrustGatedStudent), gatedStale: student(start, TrustGatedStudent) };
const aloneCopies = [student(start), student(start), student(start)];
const trustOf = () => ({ gatedTrained: arms.gatedTrained.agent.trust, gatedStale: arms.gatedStale.agent.trust });
const makeOpponent = seed => challengeOpponent(B, 470000 + 10 * s + (seed - 51000));
const check = other => frozenStudentCheckpoint({ sub, students: [arms.alone, other], studentSeed, featureMean, featureStd, makeOpponent });
const points = [];
const measure = async rounds => {
  const point = { rounds };
  for (const name of ['trained', 'stale', 'gatedTrained', 'gatedStale']) {
    const result = await check(arms[name]);
    if (point.alone !== undefined && point.alone !== result.wins[0]) throw Error('Matched frozen checks disagreed');
    point.alone = result.wins[0]; point[name] = result.wins[1];
  }
  point.trust = trustOf();
  points.push(point);
  console.log(JSON.stringify({ s, ...point }));
};
await measure(0);
const messages = { trained: 0, stale: 0 }, stalePrefs = [0, 0, 0], trainedPrefs = [0, 0, 0];
let done = 0, bothSent = 0, agreed = 0;
for (let e = 0; e < EPISODES; e++) {
  for (const x of [...Object.values(arms), ...aloneCopies]) x.game = switchGame(e);
  for (let r = 0; r < 24; r++) {
    const [tk, sk] = keys(e, r);
    teacher.key = staleTeacher.key = teacherCopy.key = staleCopy.key = tk;
    const rounds = [preferenceRound(teacher, arms.alone, arms.trained, sk),
      preferenceRound(staleTeacher, aloneCopies[0], arms.stale, sk),
      preferenceRound(teacherCopy, aloneCopies[1], arms.gatedTrained, sk),
      preferenceRound(staleCopy, aloneCopies[2], arms.gatedStale, sk)];
    for (const round of rounds) while (!round.done) round.step();
    const [tp, sp, gtp, gsp] = rounds.map(round => round.message.preference);
    if (gtp !== tp || gsp !== sp) throw Error('Gated arm heard a different message');
    if (tp !== null) { messages.trained++; trainedPrefs[tp]++; }
    if (sp !== null) { messages.stale++; stalePrefs[sp]++; }
    if (tp !== null && sp !== null) { bothSent++; if (tp === sp) agreed++; }
    if (aloneCopies.some(x => studentWeights(x.agent) !== studentWeights(arms.alone.agent))) throw Error('Alone copies diverged');
    done++;
    if (CHECKS.includes(done)) await measure(done);
  }
}
const students = [...Object.values(arms), ...aloneCopies];
const at = rounds => points.find(p => p.rounds === rounds);
// Fixed analysis (docs/trust-gate-pilot-protocol.md).
const noHarm = [48, 96].every(r => at(r).gatedStale - at(r).alone >= -4);
const benefitKept = at(48).gatedTrained - at(48).trained >= -4;
const staleHarmed = at(96).alone - at(96).stale > 4;
const protectedFromStale = at(96).gatedStale - at(96).stale > 4;
await save({ status: 'completed', teacherSeed, studentSeed, teacherWins, staleTeacherWins, messages, points,
  verdict: { noHarm, benefitKept, staleHarmed, protectedFromStale }, finalTrust: trustOf(),
  preferences: { trained: trainedPrefs, stale: stalePrefs, bothSent, agreed },
  teachersFrozen: snapshot(teacherAgent) === teacherWeights && snapshot(staleAgent) === staleWeights &&
    snapshot(teacherCopy.agent) === teacherWeights && snapshot(staleCopy.agent) === staleWeights,
  synapsesUnchanged: students.every(x => x.net.m.every(v => v === 1)) });
