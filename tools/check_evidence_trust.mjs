// Study 3: can Fly B reject an outdated teacher? Protocol: docs/evidence-trust-protocol.md.
// Copy of tools/check_trust_gate_pilot.mjs (same task, teachers, seeds formulas, evaluation) with two new
// students defined here, so shared code is unchanged:
//   EvidenceTrustStudent: trust = (rewarded matches + 1) / (all matches + 2), starting at 0.5.
//   DiscountStudent: imitation always scaled by 0.5, no trust estimate (matched baseline).
// node tools/check_evidence_trust.mjs SEED_INDEX   (29 = implementation check; 30-44 = final)
import { readFile, open, access, mkdir } from 'node:fs/promises';
import { Network, rng, wiring } from '../js/sim.js';
import { RpsContextualBandit, RpsEpisode } from '../js/rps.js';
import { PreferenceStudent, comparisonRound, preferenceRound, studentWeights } from '../js/rps_challenge_students.js';
import { frozenStudentCheckpoint } from '../js/rps_challenge_checkpoints.js';
import { challengeOpponent } from '../js/rps_challenge_opponent.js';

// Uses only what Fly B observes: its own action, the decoded advice, and its own reward.
class EvidenceTrustStudent extends PreferenceStudent {
  hits = 0; matches = 0;
  get trust() { return (this.hits + 1) / (this.matches + 2); }
  imitate(decodedAction) {
    if (this.activeHead === 3 || decodedAction === null) return 0;
    if (this.action === decodedAction) { this.matches++; this.hits += Number(this.r === 1); }
    return super.imitate(decodedAction, .03 * this.trust);
  }
}
class DiscountStudent extends PreferenceStudent {
  trust = 0.5;
  imitate(decodedAction) {
    if (this.activeHead === 3 || decodedAction === null) return 0;
    return super.imitate(decodedAction, .03 * this.trust);
  }
}

const s = Number(process.argv[2]);
if (!Number.isInteger(s) || s < 29 || s > 44) throw Error('Use seed index 29 (check) or 30-44 (final)');
const dir = new URL('../data/evidence_trust/', import.meta.url);
const output = new URL(s === 29 ? 'smoke_seed_29.json' : `seed_${s}.json`, dir);
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
const started = Date.now();
async function save(result) {
  await mkdir(dir, { recursive: true });
  const file = await open(output, 'wx');
  try { await file.writeFile(JSON.stringify({ ...result, runtimeSeconds: Math.round((Date.now() - started) / 1000) }, null, 1) + '\n'); }
  finally { await file.close(); }
  console.log(JSON.stringify({ s, status: result.status, teacherWins: result.teacherWins, staleTeacherWins: result.staleTeacherWins, points: result.points }));
}

// 1. Two teachers, as in the trust-gate study: new rule (B) and old rule (A).
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
const { agent: teacherAgent, wins: teacherWins } = trainTeacher(B, 410000, 470000);
if (teacherWins < 58) { await save({ status: 'stopped_teacher', teacherSeed, studentSeed, teacherWins }); process.exit(0); }
const { agent: staleAgent, wins: staleTeacherWins } = trainTeacher(A, 430000, 480000);
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
const keys = (e, r) => [teacherSeed * 1e7 + (200 + e) * 1000 + r, studentSeed * 1e7 + (200 + e) * 1000 + r];
const switchGame = e => challengeOpponent(B, 490000 + 100 * s + e);
// Frozen teacher copies, one per advised arm, so every arm hears a teacher in the same state.
const copyTeacher = from => {
  const net = makeNet(teacherSeed);
  const agent = new RpsContextualBandit(net, { seed: teacherSeed, featureMean, featureStd });
  from.W.forEach((w, a) => agent.W[a].set(w)); agent.b.set(from.b); agent.baseline = from.baseline;
  agent.explore = false; net.resetState();
  return { net, agent, key: 0 };
};

// 3. Switch. A no teacher; B current; C outdated; D/E evidence trust; F discount (both teachers).
const arms = { alone: student(start), trained: student(start), stale: student(start),
  evidenceStale: student(start, EvidenceTrustStudent), evidenceTrained: student(start, EvidenceTrustStudent),
  discountStale: student(start, DiscountStudent), discountTrained: student(start, DiscountStudent) };
const plan = [['trained', teacherAgent], ['stale', staleAgent], ['evidenceStale', staleAgent], ['evidenceTrained', teacherAgent],
  ['discountStale', staleAgent], ['discountTrained', teacherAgent]];
const teachers = Object.fromEntries(plan.map(([name, from]) => [name, copyTeacher(from)]));
const aloneCopies = Array.from({ length: plan.length - 1 }, () => student(start));
const trustOf = () => ({ evidenceStale: arms.evidenceStale.agent.trust, evidenceTrained: arms.evidenceTrained.agent.trust });
const makeOpponent = seed => challengeOpponent(B, 470000 + 10 * s + (seed - 51000));
const check = other => frozenStudentCheckpoint({ sub, students: [arms.alone, other], studentSeed, featureMean, featureStd, makeOpponent });
const points = [];
const measure = async rounds => {
  const point = { rounds };
  for (const [name] of plan) {
    const result = await check(arms[name]);
    if (point.alone !== undefined && point.alone !== result.wins[0]) throw Error('Matched frozen checks disagreed');
    point.alone = result.wins[0]; point[name] = result.wins[1];
  }
  point.trust = trustOf();
  points.push(point);
  console.log(JSON.stringify({ s, ...point }));
};
await measure(0);
const messages = Object.fromEntries(plan.map(([name]) => [name, 0]));
let done = 0;
for (let e = 0; e < EPISODES; e++) {
  for (const x of [...Object.values(arms), ...aloneCopies]) x.game = switchGame(e);
  for (let r = 0; r < 24; r++) {
    const [tk, sk] = keys(e, r);
    for (const t of Object.values(teachers)) t.key = tk;
    const rounds = plan.map(([name], i) => preferenceRound(teachers[name], i ? aloneCopies[i - 1] : arms.alone, arms[name], sk));
    for (const round of rounds) while (!round.done) round.step();
    const prefs = rounds.map(round => round.message.preference);
    // Arms hearing the same teacher must hear the same message.
    const byTeacher = { current: plan.flatMap(([, from], i) => from === teacherAgent ? [prefs[i]] : []),
      outdated: plan.flatMap(([, from], i) => from === staleAgent ? [prefs[i]] : []) };
    if (Object.values(byTeacher).some(list => list.some(p => p !== list[0]))) throw Error('Arms heard different messages from one teacher');
    plan.forEach(([name], i) => { if (rounds[i].message.decoded !== null) messages[name]++; });
    if (aloneCopies.some(x => studentWeights(x.agent) !== studentWeights(arms.alone.agent))) throw Error('Alone copies diverged');
    done++;
    if (CHECKS.includes(done)) await measure(done);
  }
}
const students = [...Object.values(arms), ...aloneCopies];
await save({ status: 'completed', check: s === 29, teacherSeed, studentSeed, teacherWins, staleTeacherWins, messages, points,
  finalTrust: trustOf(),
  teachersFrozen: snapshot(teacherAgent) === teacherWeights && snapshot(staleAgent) === staleWeights &&
    Object.values(teachers).every(t => [teacherWeights, staleWeights].includes(snapshot(t.agent))),
  synapsesUnchanged: students.every(x => x.net.m.every(v => v === 1)) });
