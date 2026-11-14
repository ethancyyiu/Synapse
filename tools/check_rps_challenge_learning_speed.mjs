// One write-once, prospectively locked Strategy-A learning-speed measurement.
// Run only in a copy without data/rps_challenge_learning_speed.json.
import assert from 'node:assert/strict';
import { readFile, open } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import { Network, rng, wiring } from '../js/sim.js';
import { RpsContextualBandit, RpsEpisode } from '../js/rps.js';
import { ChallengeCueHeadBandit, comparisonRound, studentWeights } from '../js/rps_challenge_students.js';
import { CHECKPOINT_ROUNDS, frozenStudentCheckpoint } from '../js/rps_challenge_checkpoints.js';
import { challengeOpponent } from '../js/rps_challenge.js';

const root = fileURLToPath(new URL('..', import.meta.url));
const output = join(root, 'data/rps_challenge_learning_speed.json');
const sub = JSON.parse(await readFile(join(root, 'data/subgraph.json'), 'utf8'));
const calibration = JSON.parse(await readFile(join(root, 'data/rps_contextual_bandit.json'), 'utf8'));
const featureMean = calibration.design.featureMean, featureStd = calibration.design.featureStd;
const config = { mode: 'preset', offset: 1 }, teacherSeed = 830, studentSeed = 10830;
const makeOpponent = seed => challengeOpponent(config, seed);
const weights = agent => JSON.stringify({ W: agent.W.map(w => [...w]), b: [...agent.b], baseline: agent.baseline });
const hashState = objects => {
  const hash = createHash('sha256');
  for (const object of objects) {
    const net = object.net;
    for (const array of [net.v, net.g, net.refr, net.trace, net.count, net.lastSpike,
      ...net.ring, net.drive, net.noise, net.noiseKick, net.m, net.spikes])
      hash.update(Buffer.from(array.buffer, array.byteOffset, array.byteLength));
    hash.update(JSON.stringify({ t: net.t, nSpikes: net.nSpikes,
      controller: object.agent.heads ? studentWeights(object.agent) : weights(object.agent),
      game: object.game ? { round: object.game.round, previousOpponent: object.game.previousOpponent,
        previousPlayer: object.game.previousPlayer, previousResult: object.game.previousResult } : null }));
  }
  return hash.digest('hex');
};
const runA = () => {
  const net = new Network(sub, wiring(sub, 'real', teacherSeed), { seed: teacherSeed });
  const agent = new RpsContextualBandit(net, { seed: teacherSeed, featureMean, featureStd,
    readoutLearn: true });
  const totals = [];
  for (let phase = 0; phase < 2; phase++) {
    const count = phase ? 3 : 12;
    if (phase) agent.explore = false;
    for (let episode = 0; episode < count; episode++) {
      const keyEpisode = phase ? 100 + episode : episode;
      const opponent = makeOpponent((phase ? 51000 : 41000) + episode);
      const ep = new RpsEpisode(net, agent, opponent, { rounds: 24, learn: !phase });
      while (!ep.done) {
        if (ep.t % 400 === 0) {
          const key = teacherSeed * 10000000 + keyEpisode * 1000 + ep.t / 400;
          net.rand = rng(key + 1); agent.rand = rng(key + 501);
        }
        ep.step();
      }
      if (phase) totals.push(ep.events.filter(e => e.reward === 1).length);
    }
  }
  assert.ok(net.m.every(x => x === 1));
  return { net, agent, frozenWins: totals.reduce((a, b) => a + b, 0) };
};
const trainedTeacher = runA();
function newTeacher() {
  const net = new Network(sub, wiring(sub, 'real', teacherSeed), { seed: teacherSeed });
  const agent = new RpsContextualBandit(net, { seed: teacherSeed, featureMean, featureStd,
    readoutLearn: false });
  agent.W.forEach((w, a) => w.set(trainedTeacher.agent.W[a]));
  agent.b.set(trainedTeacher.agent.b);
  agent.baseline = trainedTeacher.agent.baseline;
  agent.explore = false;
  return { net, agent, key: 0 };
}
const newStudent = () => {
  const net = new Network(sub, wiring(sub, 'real', studentSeed), { seed: studentSeed });
  return { net, agent: new ChallengeCueHeadBandit(net, studentSeed, featureMean, featureStd), game: null };
};
async function runStudents(checkpointsEnabled) {
  const teacher = newTeacher(), alone = newStudent(), guided = newStudent();
  assert.equal(studentWeights(alone.agent), studentWeights(guided.agent));
  assert.deepEqual([...alone.net.v], [...guided.net.v]);
  const initialWeights = studentWeights(alone.agent);
  const checkpoints = [], trace = [], episodes = [];
  const evaluate = async completed => {
    const before = hashState([teacher, alone, guided]);
    const result = await frozenStudentCheckpoint({ sub, students: [alone, guided],
      studentSeed, featureMean, featureStd, makeOpponent });
    assert.equal(hashState([teacher, alone, guided]), before, 'checkpoint modified training state');
    checkpoints.push({ rounds: completed, ...result });
  };
  if (checkpointsEnabled) await evaluate(0);
  for (let episode = 0; episode < 12; episode++) {
    const games = [makeOpponent(41000 + episode), makeOpponent(41000 + episode)];
    alone.game = games[0]; guided.game = games[1];
    const trainingWins = [0, 0];
    for (let r = 0; r < 24; r++) {
      teacher.key = teacherSeed * 10000000 + episode * 1000 + r;
      const key = studentSeed * 10000000 + episode * 1000 + r;
      const round = comparisonRound(teacher, alone, guided, key, true);
      while (!round.done) round.step();
      const rows = round.rows;
      assert.equal(rows[0].opponent, rows[1].opponent);
      rows.forEach((row, i) => { trainingWins[i] += Number(row.reward === 1); });
      trace.push(rows.map(row => ({ cue: row.cue, opponent: row.opponent,
        candidate: row.candidate, action: row.action, reward: row.reward,
        decodedBit: row.decodedBit, dnCounts: row.dnCounts, updateStep: row.updateStep })));
      const completed = episode * 24 + r + 1;
      if (checkpointsEnabled && CHECKPOINT_ROUNDS.includes(completed)) await evaluate(completed);
    }
    episodes.push({ episode: episode + 1, wins: trainingWins });
  }
  assert.notEqual(studentWeights(alone.agent), initialWeights);
  assert.ok([alone, guided].every(s => s.net.m.every(x => x === 1)));
  assert.equal(weights(teacher.agent), weights(trainedTeacher.agent));
  return { trace, checkpoints, episodes, finalWeights: [studentWeights(alone.agent),
    studentWeights(guided.agent)], teacherWeightsUnchanged: true, synapsesUnchanged: true };
}
const measured = await runStudents(true);
const control = await runStudents(false);
assert.deepEqual(measured.trace, control.trace, 'checkpoints changed training trajectory');
assert.deepEqual(measured.finalWeights, control.finalWeights, 'checkpoints changed final weights');
const early = measured.checkpoints.filter(p => [6, 12, 24, 48, 96].includes(p.rounds));
const deltas = early.map(p => p.wins[1] - p.wins[0]);
const meanEarlyDifference = deltas.reduce((a, b) => a + b, 0) / deltas.length;
const final = measured.checkpoints.at(-1);
const earlyCriterionMet = meanEarlyDifference >= 4 && deltas[0] >= 4 && deltas[1] >= 4 &&
  final.wins[1] >= final.wins[0];
const artifact = { protocol: { date: '', config, teacherSeed, studentSeed,
  checkpoints: CHECKPOINT_ROUNDS, trainOpponentSeeds: [41000, 41011],
  evalOpponentSeeds: [51000, 51002], earlyCriterion: 'mean Δ at 6,12,24,48,96 >= 4; Δ6>=4; Δ12>=4; Δ288>=0',
  proficiency: 58 }, teacherFrozenWins: trainedTeacher.frozenWins,
  initialStudentsMatched: true, trajectoryIdenticalWithAndWithoutCheckpoints: true,
  finalWeightsIdenticalWithAndWithoutCheckpoints: true,
  checkpoints: measured.checkpoints, trainingEpisodes: measured.episodes,
  meanEarlyDifference, earlyCriterionMet, ...{
    teacherWeightsUnchanged: measured.teacherWeightsUnchanged,
    synapsesUnchanged: measured.synapsesUnchanged },
  finalControllerWeights: measured.finalWeights };
const handle = await open(output, 'wx');
try { await handle.writeFile(JSON.stringify(artifact, null, 2) + '\n'); }
finally { await handle.close(); }
console.log(JSON.stringify({ teacherFrozenWins: artifact.teacherFrozenWins,
  checkpoints: artifact.checkpoints.map(p => [p.rounds, ...p.wins]), meanEarlyDifference,
  earlyCriterionMet, trajectoryIdentical: true }));
