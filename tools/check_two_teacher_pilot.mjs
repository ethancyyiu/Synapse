// Two-teacher pilot; see docs/two-teacher-pilot-protocol.md. Derived from check_trust_gate_pilot.mjs:
// Fly B hears a new-rule and an old-rule teacher on every round, in two message slots. Write-once per seed.
// node tools/check_two_teacher_pilot.mjs SEED_INDEX
import { readFile, open, access, mkdir } from 'node:fs/promises';
import { Network, rng, wiring } from '../js/sim.js';
import { RpsContextualBandit, RpsEpisode } from '../js/rps.js';
import { PreferenceStudent, comparisonRound, preferenceRound, studentWeights } from '../js/rps_challenge_students.js';
import { frozenStudentCheckpoint } from '../js/rps_challenge_checkpoints.js';
import { challengeOpponent } from '../js/rps_challenge_opponent.js';

const s = Number(process.argv[2]);
if (!Number.isInteger(s) || s < 0 || s > 17) throw Error('Use seed index 0-2 (pilot) or 3-17 (cohort)');
// Seeds 0-2 are the pilot; 3-17 are the locked cohort (docs/two-teacher-cohort-protocol.md).
const dir = new URL(s <= 2 ? '../data/two_teacher_pilot/' : '../data/two_teacher_cohort/', import.meta.url);
const output = new URL(`seed_${s}.json`, dir);
try { await access(output); throw Error('Write-once result already exists'); }
catch (error) { if (error.code !== 'ENOENT') throw error; }
const read = async name => JSON.parse(await readFile(new URL('../data/' + name, import.meta.url)));
const sub = await read('subgraph.json');
const { featureMean, featureStd } = (await read('rps_contextual_bandit.json')).design;
const A = { mode: 'preset', offset: 1 }, B = { mode: 'preset', offset: 2 };
const teacherSeed = 29001 + 2 * s, studentSeed = 29002 + 2 * s;
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

// One trust value per message slot; the rule and its 0.2 rate are those of docs/trust-gate-pilot-protocol.md.
class TwoTeacherStudent extends PreferenceStudent {
  trust = [1, 1];
  gated = false;
  imitateSlot(slot, decodedAction) {
    if (this.activeHead === 3 || decodedAction === null) return 0;
    if (!this.gated) return this.imitate(decodedAction);
    if (this.action === decodedAction) this.trust[slot] += .2 * (Number(this.r === 1) - this.trust[slot]);
    return this.imitate(decodedAction, .03 * this.trust[slot]);
  }
}
// preferenceRound's schedule with a second message slot at 290-330 ms on the same two DNs.
const DN = [3141, 3165], CODES = [[1, 0], [0, 1], [1, 1]], SLOTS = [[240, 280], [290, 330]];
const decodeCounts = counts => {
  const bits = counts.map(count => Number(count > 0));
  const found = CODES.findIndex(code => code.every((bit, i) => bit === bits[i]));
  return found < 0 ? null : found;
};
function twoTeacherRound(slotTeachers, students, key) {
  const cue = students[0].game.observe().previousOpponent;
  if (students.some(x => x.game.observe().previousOpponent !== cue)) throw Error('Student opponent cues diverged');
  const vpn = net => cue === null ? [] : Array.from(net.laneVpn[cue].slice(2));
  const preferences = [null, null];
  const counts = students.map(() => [[0, 0], [0, 0]]);
  const rows = students.map(() => ({ action: null, reward: null, opponent: null }));
  for (let t = 0; t < 680; t++) {
    if (t === 0) {
      students.forEach(x => {
        x.net.resetState(); x.net.rand = rng(key + 2); x.agent.rand = rng(key + 501);
        if (cue !== null) x.net.setDrive(vpn(x.net), 150);
      });
      slotTeachers.forEach(x => { x.net.rand = rng(x.key + 1); if (cue !== null) x.net.setDrive(vpn(x.net), 150); });
    }
    if (t === 50) [...students, ...slotTeachers].forEach(x => x.net.resetCounts());
    if (t === 240) {
      students.forEach((x, i) => {
        if (cue !== null) x.net.setDrive(vpn(x.net), 0);
        rows[i].action = x.agent.decide(x.net);
        if (x.agent.activeHead !== (cue === null ? 3 : cue)) throw Error('Student cue route did not match observed VPN input');
      });
      slotTeachers.forEach((x, slot) => {
        if (cue !== null) x.net.setDrive(vpn(x.net), 0);
        x.agent.decide(x.net);
        if (cue !== null) preferences[slot] = x.agent.scores.indexOf(Math.max(...x.agent.scores));
      });
    }
    SLOTS.forEach(([from, to], slot) => {
      if (preferences[slot] === null) return;
      if (t === from) students.forEach(x => DN.forEach((id, j) => { if (CODES[preferences[slot]][j]) x.net.setDrive([id], 300); }));
      if (t === to) students.forEach((x, i) => {
        x.net.setDrive(DN, 0);
        if (decodeCounts(counts[i][slot]) !== preferences[slot]) throw Error('Measured DN preference did not decode');
      });
    });
    if (t === 280) students.forEach((x, i) => {
      const played = x.game.play(rows[i].action);
      x.agent.reward(x.net, played.reward);
      rows[i].reward = played.reward; rows[i].opponent = played.opponent;
      x.net.resetCounts();
    });
    if (t === 620) students.forEach(x => {
      x.agent.afterDopamine(x.net, true);
      [0, 1].forEach(slot => x.agent.imitateSlot(slot, preferences[slot]));
    });
    slotTeachers.forEach(x => x.net.step());
    students.forEach((x, i) => {
      x.net.step();
      SLOTS.forEach(([from, to], slot) => {
        if (t >= from && t < to) for (let n = 0; n < x.net.nSpikes; n++) {
          if (x.net.spikes[n] === DN[0]) counts[i][slot][0]++;
          else if (x.net.spikes[n] === DN[1]) counts[i][slot][1]++;
        }
      });
    });
  }
  if (rows.some(row => row.opponent !== rows[0].opponent)) throw Error('Student opponent streams diverged');
  return { preferences };
}

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
const { net: teacherNet, agent: teacherAgent, wins: teacherWins } = trainTeacher(B, 610000, 670000);
if (teacherWins < 58) { await save({ status: 'stopped_teacher', teacherSeed, studentSeed, teacherWins }); process.exit(0); }
const { net: staleNet, agent: staleAgent, wins: staleTeacherWins } = trainTeacher(A, 630000, 680000);
if (staleTeacherWins < 58) { await save({ status: 'stopped_stale_teacher', teacherSeed, studentSeed, teacherWins, staleTeacherWins }); process.exit(0); }
const teacherWeights = snapshot(teacherAgent), staleWeights = snapshot(staleAgent);

// 2. Shared Strategy A history.
const shared = student(), twin = student();
for (let e = 0; e < 4; e++) {
  const games = [0, 1].map(() => challengeOpponent(A, 650000 + 100 * s + e));
  shared.game = games[0]; twin.game = games[1];
  for (let r = 0; r < 24; r++) {
    const round = comparisonRound(null, shared, twin, studentSeed * 1e7 + e * 1000 + r, true);
    while (!round.done) round.step();
  }
}
const start = studentWeights(shared.agent);
const teacher = { net: teacherNet, agent: teacherAgent, key: 0 };
const keys = (e, r) => [teacherSeed * 1e7 + (200 + e) * 1000 + r, studentSeed * 1e7 + (200 + e) * 1000 + r];
const switchGame = e => challengeOpponent(B, 690000 + 100 * s + e);

const staleTeacher = { net: staleNet, agent: staleAgent, key: 0 };
teacherNet.resetState(); staleNet.resetState();
// A frozen copy of the new-rule teacher for the two-teacher arms, so every arm hears it in the same state.
const copyTeacher = from => {
  const net = makeNet(teacherSeed);
  const agent = new RpsContextualBandit(net, { seed: teacherSeed, featureMean, featureStd });
  from.W.forEach((w, a) => agent.W[a].set(w)); agent.b.set(from.b); agent.baseline = from.baseline;
  agent.explore = false; net.resetState();
  return { net, agent, key: 0 };
};
const teacherCopy = copyTeacher(teacherAgent);
// The new-rule teacher has slot 0 on even seed indices and slot 1 on odd ones.
const newRuleSlot = s % 2;
const slotTeachers = newRuleSlot === 0 ? [teacherCopy, staleTeacher] : [staleTeacher, teacherCopy];

// 4. Switch.
const arms = { alone: student(start), trained: student(start),
  bothUngated: student(start, TwoTeacherStudent), bothGated: student(start, TwoTeacherStudent) };
arms.bothGated.agent.gated = true;
const trustOf = () => ({ newRule: arms.bothGated.agent.trust[newRuleSlot], oldRule: arms.bothGated.agent.trust[1 - newRuleSlot] });
const makeOpponent = seed => challengeOpponent(B, 670000 + 10 * s + (seed - 51000));
const check = other => frozenStudentCheckpoint({ sub, students: [arms.alone, other], studentSeed, featureMean, featureStd, makeOpponent });
const points = [];
const measure = async rounds => {
  const point = { rounds };
  for (const name of ['trained', 'bothUngated', 'bothGated']) {
    const result = await check(arms[name]);
    if (point.alone !== undefined && point.alone !== result.wins[0]) throw Error('Matched frozen checks disagreed');
    point.alone = result.wins[0]; point[name] = result.wins[1];
  }
  point.trust = trustOf();
  points.push(point);
  console.log(JSON.stringify({ s, ...point }));
};
await measure(0);
const messages = { newRule: 0, oldRule: 0 };
let done = 0, bothSent = 0, agreed = 0;
for (let e = 0; e < EPISODES; e++) {
  for (const x of Object.values(arms)) x.game = switchGame(e);
  for (let r = 0; r < 24; r++) {
    const [tk, sk] = keys(e, r);
    teacher.key = staleTeacher.key = teacherCopy.key = tk;
    const single = preferenceRound(teacher, arms.alone, arms.trained, sk);
    while (!single.done) single.step();
    const { preferences } = twoTeacherRound(slotTeachers, [arms.bothUngated, arms.bothGated], sk);
    const tp = preferences[newRuleSlot], sp = preferences[1 - newRuleSlot];
    if (tp !== single.message.preference) throw Error('Two-teacher arms heard a different new-rule message');
    if (tp !== null) messages.newRule++;
    if (sp !== null) messages.oldRule++;
    if (tp !== null && sp !== null) { bothSent++; if (tp === sp) agreed++; }
    done++;
    if (CHECKS.includes(done)) await measure(done);
  }
}
const students = Object.values(arms);
const at = rounds => points.find(p => p.rounds === rounds);
// Fixed analysis (docs/two-teacher-pilot-protocol.md).
const choosingHelps = at(48).bothGated - at(48).bothUngated > 4;
const badAdviserCostsLittle = at(48).bothGated - at(48).trained >= -4;
await save({ status: 'completed', teacherSeed, studentSeed, teacherWins, staleTeacherWins, newRuleSlot, messages, points,
  verdict: { choosingHelps, badAdviserCostsLittle }, finalTrust: trustOf(),
  agreement: { bothSent, agreed },
  teachersFrozen: snapshot(teacherAgent) === teacherWeights && snapshot(staleAgent) === staleWeights &&
    snapshot(teacherCopy.agent) === teacherWeights,
  synapsesUnchanged: students.every(x => x.net.m.every(v => v === 1)) });
