// Locked rule-switch pilot; see docs/rule-switch-pilot-protocol.md. Write-once per seed.
// node tools/check_rule_switch_pilot.mjs SEED_INDEX
import { readFile, open, access, mkdir } from 'node:fs/promises';
import { Network, rng, wiring } from '../js/sim.js';
import { RpsContextualBandit, RpsEpisode } from '../js/rps.js';
import { PreferenceStudent, comparisonRound, preferenceRound, studentWeights } from '../js/rps_challenge_students.js';
import { frozenStudentCheckpoint } from '../js/rps_challenge_checkpoints.js';
import { challengeOpponent } from '../js/rps_challenge_opponent.js';

const s = Number(process.argv[2]);
if (![0, 1, 2].includes(s)) throw Error('Use seed index 0, 1, or 2');
const dir = new URL('../data/rule_switch_pilot/', import.meta.url);
const output = new URL(`seed_${s}.json`, dir);
try { await access(output); throw Error('Write-once result already exists'); }
catch (error) { if (error.code !== 'ENOENT') throw error; }
const read = async name => JSON.parse(await readFile(new URL('../data/' + name, import.meta.url)));
const sub = await read('subgraph.json');
const { featureMean, featureStd } = (await read('rps_contextual_bandit.json')).design;
const A = { mode: 'preset', offset: 1 }, B = { mode: 'preset', offset: 2 };
const teacherSeed = 24001 + 2 * s, studentSeed = 24002 + 2 * s;
const makeNet = seed => new Network(sub, wiring(sub, 'real', seed), { seed });
const student = weights => {
  const net = makeNet(studentSeed), agent = new PreferenceStudent(net, studentSeed, featureMean, featureStd);
  if (weights) {
    const saved = JSON.parse(weights);
    saved.heads.forEach((h, i) => { h.W.forEach((w, a) => agent.heads[i].W[a].set(w)); agent.heads[i].b.set(h.b); });
    agent.baseline = saved.baseline;
  }
  return { net, agent, game: null };
};

async function save(result) {
  await mkdir(dir, { recursive: true });
  const file = await open(output, 'wx');
  try { await file.writeFile(JSON.stringify(result, null, 1) + '\n'); } finally { await file.close(); }
  console.log(JSON.stringify({ s, status: result.status, teacherWins: result.teacherWins, advantage: result.advantage, wins: result.wins }));
}

const teacherNet = makeNet(teacherSeed);
const teacherAgent = new RpsContextualBandit(teacherNet, { seed: teacherSeed, featureMean, featureStd });
for (let e = 0; e < 12; e++) {
  const ep = new RpsEpisode(teacherNet, teacherAgent, challengeOpponent(B, 84000 + 100 * s + e), { rounds: 24, learn: true });
  while (!ep.done) {
    if (ep.t % 400 === 0) { const k = teacherSeed * 1e7 + e * 1000 + ep.t / 400; teacherNet.rand = rng(k + 1); teacherAgent.rand = rng(k + 501); }
    ep.step();
  }
}
teacherAgent.explore = false;
let teacherWins = 0;
for (let e = 0; e < 3; e++) {
  const ep = new RpsEpisode(teacherNet, teacherAgent, challengeOpponent(B, 96000 + 10 * s + e), { rounds: 24, learn: false });
  while (!ep.done) {
    if (ep.t % 400 === 0) { const k = teacherSeed * 1e7 + (100 + e) * 1000 + ep.t / 400; teacherNet.rand = rng(k + 1); teacherAgent.rand = rng(k + 501); }
    ep.step();
  }
  teacherWins += ep.events.filter(event => event.reward === 1).length;
}
if (teacherWins < 58) { await save({ status: 'stopped_teacher', teacherSeed, studentSeed, teacherWins }); process.exit(0); }
teacherNet.resetState();

const shared = student(), twin = student();
shared.agent.explore = twin.agent.explore = true;
for (let e = 0; e < 4; e++) {
  const games = [0, 1].map(() => challengeOpponent(A, 85000 + 100 * s + e));
  shared.game = games[0]; twin.game = games[1];
  for (let r = 0; r < 24; r++) {
    const round = comparisonRound(null, shared, twin, studentSeed * 1e7 + e * 1000 + r, true);
    while (!round.done) round.step();
  }
}
const start = studentWeights(shared.agent);
const alone = student(start), trained = student(start), random = student(start);
[alone, trained, random].forEach(x => { x.agent.explore = true; });
const teacher = { net: teacherNet, agent: teacherAgent, key: 0 };

const makeOpponent = seed => challengeOpponent(B, 96000 + 10 * s + (seed - 51000));
const check = other => frozenStudentCheckpoint({ sub, students: [alone, other], studentSeed, featureMean, featureStd, makeOpponent });
const points = [];
const measure = async rounds => {
  const [taught, control] = [await check(trained), await check(random)];
  if (taught.wins[0] !== control.wins[0]) throw Error('Matched frozen checks disagreed');
  points.push({ rounds, alone: taught.wins[0], trained: taught.wins[1], random: control.wins[1] });
};
await measure(0);
let done = 0, messages = 0;
for (let e = 0; e < 2; e++) {
  const games = [0, 1, 2].map(() => challengeOpponent(B, 86000 + 100 * s + e));
  alone.game = games[0]; trained.game = games[1]; random.game = games[2];
  for (let r = 0; r < 24; r++) {
    teacher.key = teacherSeed * 1e7 + (200 + e) * 1000 + r;
    const round = preferenceRound(teacher, alone, trained, studentSeed * 1e7 + (200 + e) * 1000 + r, random);
    while (!round.done) round.step();
    if (round.message.preference !== null) messages++;
    done++;
    if ([6, 12, 24, 48].includes(done)) await measure(done);
  }
}
const last = points.at(-1);
await save({ status: 'completed', teacherSeed, studentSeed, teacherWins, messages, points,
  wins: { alone: last.alone, trained: last.trained, random: last.random },
  advantage: last.trained - Math.max(last.alone, last.random),
  synapsesUnchanged: [alone, trained, random].every(x => x.net.m.every(v => v === 1)) });
