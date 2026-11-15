// Study 1, teaching comparison. Protocol: docs/connectome-contribution-2-protocol.md. Write-once per seed.
// Rule-switch task as in the advice-budget follow-up (48 post-switch rounds, advice on all 46 observed-cue rounds).
//   Real connectome: alone, neural (A: measured DN spikes), digital (B: same preference, no spikes),
//                    random (C: random content through the same spike channel).
//   Rewired connectome ('shuffled' wiring for teacher and students): aloneS, neuralS, digitalS (D).
// node tools/check_connectome_contribution_2.mjs SEED_INDEX   (20 = implementation check; 0-14 = final)
import { readFile, open, access, mkdir } from 'node:fs/promises';
import { Network, rng, wiring } from '../js/sim.js';
import { RpsContextualBandit, RpsEpisode } from '../js/rps.js';
import { PreferenceStudent, preferenceRound, comparisonRound, studentWeights } from '../js/rps_challenge_students.js';
import { frozenStudentCheckpoint } from '../js/rps_challenge_checkpoints.js';
import { challengeOpponent } from '../js/rps_challenge_opponent.js';

const s = Number(process.argv[2]);
if (!Number.isInteger(s) || (s !== 20 && (s < 0 || s > 14))) throw Error('Use check seed 20 or final seed 0-14');
const dir = new URL('../data/connectome_contribution_2/', import.meta.url);
const output = new URL(s === 20 ? 'smoke_seed_20.json' : `seed_${s}.json`, dir);
try { await access(output); throw Error('Write-once result already exists'); }
catch (error) { if (error.code !== 'ENOENT') throw error; }
const read = async name => JSON.parse(await readFile(new URL('../data/' + name, import.meta.url)));
const sub = await read('subgraph.json');
const { featureMean, featureStd } = (await read('rps_contextual_bandit.json')).design;
const A = { mode: 'preset', offset: 1 }, B = { mode: 'preset', offset: 2 };
const teacherSeed = 37001 + 2 * s, studentSeed = 37002 + 2 * s;
const EPISODES = 2, CHECKS = [24, 48];
const snapshot = a => JSON.stringify({ W: a.W.map(w => [...w]), b: [...a.b], baseline: a.baseline });
const started = Date.now();
async function save(result) {
  await mkdir(dir, { recursive: true });
  const file = await open(output, 'wx');
  try { await file.writeFile(JSON.stringify({ ...result, runtimeSeconds: Math.round((Date.now() - started) / 1000) }, null, 1) + '\n'); }
  finally { await file.close(); }
  console.log(JSON.stringify({ s, status: result.status, worlds: result.worlds && Object.fromEntries(Object.entries(result.worlds).map(([k, v]) => [k, v.status])) }));
}

// One "world" = one wiring variant: teacher, shared history, tape, and the arms that live in it.
async function world(variant, armNames) {
  const makeNet = seed => new Network(sub, wiring(sub, variant, seed), { seed });
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
  const teacherNet = makeNet(teacherSeed);
  const teacherAgent = new RpsContextualBandit(teacherNet, { seed: teacherSeed, featureMean, featureStd });
  for (let e = 0; e < 12; e++) {
    const ep = new RpsEpisode(teacherNet, teacherAgent, challengeOpponent(B, 1310000 + 100 * s + e), { rounds: 24, learn: true });
    while (!ep.done) {
      if (ep.t % 400 === 0) { const k = teacherSeed * 1e7 + e * 1000 + ep.t / 400; teacherNet.rand = rng(k + 1); teacherAgent.rand = rng(k + 501); }
      ep.step();
    }
  }
  teacherAgent.explore = false;
  let teacherWins = 0;
  for (let e = 0; e < 3; e++) {
    const ep = new RpsEpisode(teacherNet, teacherAgent, challengeOpponent(B, 1330000 + 10 * s + e), { rounds: 24, learn: false });
    while (!ep.done) {
      if (ep.t % 400 === 0) { const k = teacherSeed * 1e7 + (100 + e) * 1000 + ep.t / 400; teacherNet.rand = rng(k + 1); teacherAgent.rand = rng(k + 501); }
      ep.step();
    }
    teacherWins += ep.events.filter(event => event.reward === 1).length;
  }
  if (teacherWins < 58) return { status: 'stopped_teacher', teacherWins };
  const teacherWeights = snapshot(teacherAgent);
  // Shared rule-A history.
  const shared = student(), twin = student();
  for (let e = 0; e < 4; e++) {
    const games = [0, 1].map(() => challengeOpponent(A, 1350000 + 100 * s + e));
    shared.game = games[0]; twin.game = games[1];
    for (let r = 0; r < 24; r++) {
      const round = comparisonRound(null, shared, twin, studentSeed * 1e7 + e * 1000 + r, true);
      while (!round.done) round.step();
    }
  }
  const start = studentWeights(shared.agent);
  const keys = (e, r) => [teacherSeed * 1e7 + (200 + e) * 1000 + r, studentSeed * 1e7 + (200 + e) * 1000 + r];
  const switchGame = e => challengeOpponent(B, 1370000 + 100 * s + e);
  // Teacher preference tape.
  const teacher = { net: teacherNet, agent: teacherAgent, key: 0 };
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
  const preferences = tape.map(x => x.preference);
  const tapeTeacher = () => ({ net: makeNet(teacherSeed), key: 0, agent: { i: 0, scores: [0, 0, 0],
    decide() { const p = preferences[this.i]; this.scores = [0, 1, 2].map(a => Number(a === p)); } } });
  const [aloneName, neuralName, digitalName, randomName] = armNames;
  const arms = { [aloneName]: student(start), [neuralName]: student(start), [digitalName]: student(start) };
  if (randomName) arms[randomName] = student(start);
  const twinB = student(start);
  const tNeural = tapeTeacher(), tDigital = tapeTeacher();
  const makeOpponent = seed => challengeOpponent(B, 1330000 + 10 * s + (seed - 51000));
  const points = [];
  const counts = { [neuralName]: { sent: 0, decoded: 0 }, [digitalName]: { delivered: 0 } };
  if (randomName) counts[randomName] = { sent: 0, decoded: 0 };
  const measure = async rounds => {
    const point = { rounds };
    for (const name of Object.keys(arms).filter(n => n !== aloneName)) {
      const result = await frozenStudentCheckpoint({ sub, students: [arms[aloneName], arms[name]], studentSeed, featureMean, featureStd, makeOpponent,
        // Frozen checks rebuild each student's network; in the rewired world they must use the same wiring.
        ...(variant === 'real' ? {} : { engine: { Network, wiring: (g, _variant, seed) => wiring(g, variant, seed) } }) });
      if (point[aloneName] !== undefined && point[aloneName] !== result.wins[0]) throw Error('Matched frozen checks disagreed');
      point[aloneName] = result.wins[0]; point[name] = result.wins[1];
    }
    points.push(point);
  };
  await measure(0);
  let done = 0;
  for (let e = 0; e < EPISODES; e++) {
    for (const x of [...Object.values(arms), twinB]) x.game = switchGame(e);
    for (let r = 0; r < 24; r++) {
      const [tk, sk] = keys(e, r);
      tNeural.key = tDigital.key = tk; tNeural.agent.i = tDigital.agent.i = done;
      const pref = preferences[done];
      const rounds = [
        preferenceRound(tNeural, arms[aloneName], arms[neuralName], sk, randomName ? arms[randomName] : null, true),
        preferenceRound(tDigital, twinB, arms[digitalName], sk, null, false),   // no spikes sent
      ];
      for (const round of rounds) while (!round.done) round.step();
      if (rounds.some(round => round.rows[0].cue !== tape[done].cue)) throw Error('Cue did not reproduce the tape');
      // Digital delivery: the same preference applied to Fly B's learner directly, in the same order as the
      // neural path (after its own-reward update), with no DN drive.
      if (pref !== null) { arms[digitalName].agent.imitate(pref); counts[digitalName].delivered++; }
      if (rounds[0].message.preference !== null) {
        counts[neuralName].sent++;
        if (rounds[0].message.decoded === pref) counts[neuralName].decoded++;
        if (randomName) { counts[randomName].sent++; if (rounds[0].rows[2].decodedBit !== null) counts[randomName].decoded++; }
      }
      if (studentWeights(twinB.agent) !== studentWeights(arms[aloneName].agent)) throw Error('Alone copies diverged');
      done++;
      if (CHECKS.includes(done)) await measure(done);
    }
  }
  const students = [...Object.values(arms), twinB];
  return { status: 'completed', teacherWins, counts, points, teacherFrozen: snapshot(teacherAgent) === teacherWeights,
    synapsesUnchanged: students.every(x => x.net.m.every(v => v === 1)) };
}

const real = await world('real', ['alone', 'neural', 'digital', 'random']);
const rewired = await world('shuffled', ['aloneS', 'neuralS', 'digitalS']);
await save({ status: real.status === 'completed' ? 'completed' : real.status, check: s === 20, teacherSeed, studentSeed,
  worlds: { real, rewired } });
