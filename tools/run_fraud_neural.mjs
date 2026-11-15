// Separate synthetic fraud-response study, using the existing fault-response
// controllers and measured DN spike channel. See docs/synapse-fraud-neural-protocol.md.
import { readFile, mkdir, open, access } from 'node:fs/promises';
import { Network, rng, wiring } from '../js/sim.js';
import { RpsContextualBandit, RpsEpisode } from '../js/rps.js';
import { PreferenceStudent, comparisonRound, preferenceRound, studentWeights } from '../js/rps_challenge_students.js';
import { frozenStudentCheckpoint } from '../js/rps_challenge_checkpoints.js';
import { ACTIONS, balancedCues, FraudResponseGame, NEW_NEED, OLD_NEED } from '../js/fraud_neural_game.js';

const [mode, seedText] = process.argv.slice(2);
const seed = Number(seedText);
if (mode !== 'probe' && !(mode === 'seed' && Number.isInteger(seed) && [401, 420, 421, 422, 423, 424].includes(seed)))
  throw Error('Use probe or seed 401/420/421/422/423/424');
const sub = JSON.parse(await readFile(new URL('../data/subgraph.json', import.meta.url)));
const { featureMean, featureStd } = JSON.parse(await readFile(new URL('../data/rps_contextual_bandit.json', import.meta.url))).design;
const makeNet = s => new Network(sub, wiring(sub, 'real', s), { seed: s });
const teacherSeed = 90001 + 2 * (mode === 'probe' ? 400 : seed);
const studentSeed = teacherSeed + 1;
const game = (need, gameSeed, sequence = null) => new FraudResponseGame(need, gameSeed, sequence);
const outputDir = new URL('../data/fraud_neural/', import.meta.url);
const output = new URL(`seed_${seed}.json`, outputDir);
const snapshot = agent => JSON.stringify({ W: agent.W.map(w => [...w]), b: [...agent.b], baseline: agent.baseline });
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

if (mode === 'probe') {
  const ep = new RpsEpisode(teacherNet, teacherAgent, game(NEW_NEED, 910400), { rounds: 24, learn: true });
  while (!ep.done) ep.step();
  teacherAgent.explore = false;
  const alone = student(), guided = student();
  alone.game = game(NEW_NEED, 920400); guided.game = game(NEW_NEED, 920400);
  const teacher = { net: teacherNet, agent: teacherAgent, key: 930400 };
  const round = preferenceRound(teacher, alone, guided, 940400);
  while (!round.done) round.step();
  const { preference, decoded, dnCounts } = round.message;
  if (preference === null || decoded !== preference || !dnCounts.some(Boolean) ||
      !round.rows[1].imitation || sub.neurons.role.length !== 6900)
    throw Error('End-to-end DN teaching probe failed');
  console.log(JSON.stringify({ status: 'passed', neurons: sub.neurons.role.length,
    synapses: sub.edges.pre.length, cue: round.rows[1].cue, sent: ACTIONS[preference],
    decoded: ACTIONS[decoded], dnCounts, studentImitation: round.rows[1].imitation }));
  process.exit(0);
}

try { await access(output); throw Error('Write-once fraud neural result already exists'); }
catch (error) { if (error.code !== 'ENOENT') throw error; }
const save = async result => {
  await mkdir(outputDir, { recursive: true });
  const file = await open(output, 'wx');
  try { await file.writeFile(JSON.stringify(result, null, 2) + '\n'); }
  finally { await file.close(); }
  console.log(JSON.stringify({ seed, status: result.status, teacherWins: result.teacherWins,
    advantage48: result.advantage48, points: result.points?.map(({ rounds, ...arms }) => ({ rounds, ...arms })) }));
};

// Train Fly A on the updated synthetic policy with the existing neural episode.
const teacherEpisodeWins = [];
for (let e = 0; e < 12; e++) {
  const ep = new RpsEpisode(teacherNet, teacherAgent, game(NEW_NEED, 910000 + 100 * seed + e),
    { rounds: 24, learn: true });
  while (!ep.done) {
    if (ep.t % 400 === 0) {
      const k = teacherSeed * 1e7 + e * 1000 + ep.t / 400;
      teacherNet.rand = rng(k + 1); teacherAgent.rand = rng(k + 501);
    }
    ep.step();
  }
  teacherEpisodeWins.push(ep.events.filter(event => event.reward === 1).length);
}
teacherAgent.explore = false;
let teacherWins = 0;
for (let e = 0; e < 3; e++) {
  const ep = new RpsEpisode(teacherNet, teacherAgent, game(NEW_NEED, 970000 + 10 * seed + e),
    { rounds: 24, learn: false });
  while (!ep.done) {
    if (ep.t % 400 === 0) {
      const k = teacherSeed * 1e7 + (100 + e) * 1000 + ep.t / 400;
      teacherNet.rand = rng(k + 1); teacherAgent.rand = rng(k + 501);
    }
    ep.step();
  }
  teacherWins += ep.events.filter(event => event.reward === 1).length;
}
if (teacherWins < 58) {
  await save({ status: 'stopped_teacher', seed, teacherSeed, studentSeed, teacherWins,
    teacherEpisodeWins, teacherGate: 58 });
  process.exit(0);
}
const teacherWeights = snapshot(teacherAgent);

// Identical old-policy history for every Fly B arm.
const shared = student(), twin = student();
for (let e = 0; e < 4; e++) {
  shared.game = game(OLD_NEED, 930000 + 100 * seed + e);
  twin.game = game(OLD_NEED, 930000 + 100 * seed + e);
  for (let r = 0; r < 24; r++) {
    const round = comparisonRound(null, shared, twin, studentSeed * 1e7 + e * 1000 + r, true);
    while (!round.done) round.step();
  }
}
const start = studentWeights(shared.agent);
const teacher = { net: teacherNet, agent: teacherAgent, key: 0 };
const keys = (e, r) => [teacherSeed * 1e7 + (200 + e) * 1000 + r,
  studentSeed * 1e7 + (200 + e) * 1000 + r];
const switchGame = e => game(NEW_NEED, 950000 + 100 * seed + e);

// Record Fly A's preferences once. Shuffling changes the advice, not the
// students' cue schedule or spike-channel transmission mechanism.
teacherNet.resetState();
const tape = [];
{
  const a = student(start), b = student(start);
  for (let e = 0; e < 4; e++) {
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
const positions = tape.flatMap((row, i) => row.preference === null ? [] : [i]);
const values = positions.map(i => tape[i].preference), shuffleRng = rng(2026111500 + seed);
for (let i = values.length - 1; i > 0; i--) {
  const j = Math.floor(shuffleRng() * (i + 1));
  [values[i], values[j]] = [values[j], values[i]];
}
const shuffledTape = Array(tape.length).fill(null);
positions.forEach((position, i) => { shuffledTape[position] = values[i]; });
const tapeTeacher = targets => ({ net: makeNet(teacherSeed), key: 0, agent: {
  index: 0, scores: [0, 0, 0],
  decide() { const preference = targets[this.index]; this.scores = [0, 1, 2].map(a => Number(a === preference)); },
} });
const shuffledTeacher = tapeTeacher(shuffledTape);
const arms = { alone: student(start), trained: student(start), random: student(start), shuffled: student(start) };
const extraAlone = student(start);
const check = other => frozenStudentCheckpoint({ sub, students: [arms.alone, other],
  studentSeed, featureMean, featureStd,
  makeOpponent: () => game(NEW_NEED, 990000 + seed, balancedCues(990000 + seed)) });
const points = [];
const measure = async rounds => {
  const point = { rounds };
  for (const name of ['trained', 'random', 'shuffled']) {
    const result = await check(arms[name]);
    if (point.alone !== undefined && point.alone !== result.wins[0])
      throw Error('Matched frozen checks diverged');
    point.alone = result.wins[0]; point[name] = result.wins[1];
  }
  points.push(point);
  console.log(JSON.stringify({ seed, checkpoint: point }));
};
await measure(0);
const messages = { trained: 0, random: 0, shuffled: 0 };
const decoded = { trained: 0, random: 0, shuffled: 0 };
const spikeCounts = { trained: [0, 0], random: [0, 0], shuffled: [0, 0] };
const teachingTrace = [];
let done = 0;
for (let e = 0; e < 4; e++) {
  for (const arm of [...Object.values(arms), extraAlone]) arm.game = switchGame(e);
  for (let r = 0; r < 24; r++) {
    const [tk, sk] = keys(e, r);
    teacher.key = shuffledTeacher.key = tk;
    shuffledTeacher.agent.index = done;
    const rounds = [preferenceRound(teacher, arms.alone, arms.trained, sk, arms.random),
      preferenceRound(shuffledTeacher, extraAlone, arms.shuffled, sk)];
    for (const round of rounds) while (!round.done) round.step();
    if (rounds[0].message.preference !== tape[done].preference ||
        rounds[0].rows[0].cue !== tape[done].cue ||
        rounds[1].message.preference !== shuffledTape[done])
      throw Error('Teacher preference tape did not reproduce');
    for (const [name, row, sent] of [
      ['trained', rounds[0].rows[1], rounds[0].message.preference],
      ['random', rounds[0].rows[2], rounds[0].rows[2].decodedBit],
      ['shuffled', rounds[1].rows[1], rounds[1].message.preference],
    ]) {
      if (sent !== null) {
        messages[name]++;
        if (row.decodedBit !== null && row.dnCounts.some(Boolean)) decoded[name]++;
        spikeCounts[name][0] += row.dnCounts[0];
        spikeCounts[name][1] += row.dnCounts[1];
      }
    }
    if (teachingTrace.length < 6) {
      const row = rounds[0].rows[1], msg = rounds[0].message;
      teachingTrace.push({ round: done + 1, cue: row.cue, sentAction: msg.preference,
        dnCounts: msg.dnCounts, decodedAction: msg.decoded,
        flyBAction: row.action, syntheticReward: row.reward, imitationUpdate: row.imitation });
    }
    if (studentWeights(extraAlone.agent) !== studentWeights(arms.alone.agent))
      throw Error('Alone controls diverged');
    done++;
    if ([24, 48, 96].includes(done)) await measure(done);
  }
}
if (Object.keys(messages).some(name => messages[name] !== decoded[name]))
  throw Error('Some DN messages did not decode');
const at = rounds => points.find(p => p.rounds === rounds);
const advantage48 = at(48).trained - Math.max(at(48).alone, at(48).random, at(48).shuffled);
const recoveryAt = Object.fromEntries(Object.keys(arms).map(name =>
  [name, points.find(point => point[name] >= 58)?.rounds ?? null]));
await save({ status: 'completed', protocol: 'docs/synapse-fraud-neural-protocol.md', seed,
  teacherSeed, studentSeed, neurons: sub.neurons.role.length, synapses: sub.edges.pre.length,
  teacherWins, teacherGate: 58, teacherEpisodeWins, points, recoveryAt,
  advantage48, messages, decoded, spikeCounts, teachingTrace,
  trainedStudentWeights: studentWeights(arms.trained.agent),
  teacherFrozen: snapshot(teacherAgent) === teacherWeights,
  synapsesUnchanged: Object.values(arms).every(arm => arm.net.m.every(value => value === 1)) });
