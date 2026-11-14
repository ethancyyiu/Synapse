// Prospective five-arm Strategy-A control test. Run once per stage: initial, then confirm only if initial passes.
import assert from 'node:assert/strict';
import { readFile, open, access } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import { Network, rng, wiring } from '../js/sim.js';
import { RpsContextualBandit, RpsEpisode } from '../js/rps.js';
import { ChallengeCueHeadBandit, comparisonRound, studentWeights } from '../js/rps_challenge_students.js';
import { CHECKPOINT_ROUNDS, frozenStudentCheckpoint } from '../js/rps_challenge_checkpoints.js';
import { challengeOpponent } from '../js/rps_challenge.js';

const stage = process.argv[2];
if (!['initial', 'confirm'].includes(stage)) throw Error('Use initial or confirm');
const root = fileURLToPath(new URL('..', import.meta.url));
const file = name => join(root, name);
const output = file(`data/rps_teacher_specific_speed_${stage}.json`);
try { await access(output); throw Error(`Refusing to overwrite ${output}`); } catch (error) {
  if (error.code !== 'ENOENT') throw error;
}
if (stage === 'confirm') {
  const initial = JSON.parse(await readFile(file('data/rps_teacher_specific_speed_initial.json')));
  if (!initial.passed) throw Error('Initial gates did not pass; confirmation is prohibited');
}
const sub = JSON.parse(await readFile(file('data/subgraph.json')));
const calibration = JSON.parse(await readFile(file('data/rps_contextual_bandit.json'))).design;
const { featureMean, featureStd } = calibration;
const pairs = stage === 'initial' ? [[830, 10830]] : [[915, 10915], [916, 10916]];
const config = { mode: 'preset', offset: 1 };
const makeOpponent = seed => challengeOpponent(config, seed);
const names = ['correct', 'alone', 'untrained', 'shuffled', 'generic'];
const approve = (scores, candidate) => Number(scores[candidate] >
  Math.max(...scores.filter((_, action) => action !== candidate)));
const teacherWeights = agent => JSON.stringify({ W: agent.W.map(w => [...w]), b: [...agent.b],
  baseline: agent.baseline });
const freshStudent = seed => {
  const net = new Network(sub, wiring(sub, 'real', seed), { seed });
  return { net, agent: new ChallengeCueHeadBandit(net, seed, featureMean, featureStd), game: null };
};
const stateHash = objects => {
  const hash = createHash('sha256');
  for (const object of objects) {
    const net = object.net;
    for (const array of [net.v, net.g, net.refr, net.trace, net.count, net.lastSpike,
      ...net.ring, net.drive, net.noise, net.noiseKick, net.m, net.spikes])
      hash.update(Buffer.from(array.buffer, array.byteOffset, array.byteLength));
    hash.update(JSON.stringify({ t: net.t, nSpikes: net.nSpikes,
      weights: object.agent.heads ? studentWeights(object.agent) : teacherWeights(object.agent),
      game: object.game ? [object.game.round, object.game.previousOpponent,
        object.game.previousPlayer, object.game.previousResult] : null }));
  }
  return hash.digest('hex');
};
function trainTeacher(seed) {
  const net = new Network(sub, wiring(sub, 'real', seed), { seed });
  const agent = new RpsContextualBandit(net, { seed, featureMean, featureStd, readoutLearn: true });
  let evalRows = [];
  for (let phase = 0; phase < 2; phase++) {
    if (phase) agent.explore = false;
    for (let episode = 0; episode < (phase ? 3 : 12); episode++) {
      const episodeKey = phase ? 100 + episode : episode;
      const ep = new RpsEpisode(net, agent,
        makeOpponent((phase ? 51000 : 41000) + episode), { rounds: 24, learn: !phase });
      while (!ep.done) {
        if (ep.t % 400 === 0) {
          const key = seed * 10000000 + episodeKey * 1000 + ep.t / 400;
          net.rand = rng(key + 1); agent.rand = rng(key + 501);
        }
        ep.step();
      }
      if (phase) evalRows.push(...ep.events);
    }
  }
  assert.ok(net.m.every(x => x === 1));
  const wins = evalRows.filter(x => x.reward === 1).length;
  const byCue = [0, 1, 2].map(cue => {
    const xs = evalRows.filter(x => x.observation.previousOpponent === cue);
    return { wins: xs.filter(x => x.reward === 1).length, n: xs.length };
  });
  return { agent, seed, wins, byCue, valid: wins >= 58 &&
    byCue.every(x => x.wins / x.n >= .70) };
}
function newTeacher(trained, untrained = false) {
  const net = new Network(sub, wiring(sub, 'real', trained.seed), { seed: trained.seed });
  const agent = new RpsContextualBandit(net, { seed: trained.seed, featureMean, featureStd,
    readoutLearn: false });
  if (!untrained) {
    agent.W.forEach((w, a) => w.set(trained.agent.W[a]));
    agent.b.set(trained.agent.b);
    agent.baseline = trained.agent.baseline;
  }
  agent.explore = false;
  return { net, agent, key: 0 };
}
function sourceTape(trained) {
  const source = newTeacher(trained), occurrence = [0, 0, 0], entries = [];
  for (let episode = 0; episode < 12; episode++) {
    const game = makeOpponent(41000 + episode);
    for (let r = 0; r < 24; r++) {
      const cue = game.observe().previousOpponent;
      const vpn = cue === null ? [] : Array.from(source.net.laneVpn[cue].slice(2));
      const key = trained.seed * 10000000 + episode * 1000 + r;
      source.net.rand = rng(key + 1);
      if (vpn.length) source.net.setDrive(vpn, 150);
      let bit = null, candidate = null;
      for (let t = 0; t < 680; t++) {
        if (t === 50) source.net.resetCounts();
        if (t === 240) {
          if (vpn.length) source.net.setDrive(vpn, 0);
          source.agent.decide(source.net);
          if (cue !== null) {
            candidate = occurrence[cue]++ % 3;
            bit = approve(source.agent.scores, candidate);
          }
        }
        source.net.step();
      }
      entries.push({ cue, candidate, bit });
      game.play(0); // Strategy A depends only on the previous opponent move.
    }
  }
  const shuffled = Array(entries.length).fill(null), counts = [];
  for (let cue = 0; cue < 3; cue++) {
    const positions = entries.flatMap((e, i) => e.cue === cue ? [i] : []);
    const bits = positions.map(i => entries[i].bit);
    const random = rng(2026110601 + trained.seed + cue);
    for (let i = bits.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [bits[i], bits[j]] = [bits[j], bits[i]];
    }
    positions.forEach((position, i) => { shuffled[position] = bits[i]; });
    counts.push({ cue, n: bits.length, ones: bits.filter(Boolean).length,
      moved: positions.filter((position, i) => entries[position].bit !== bits[i]).length });
  }
  return { shuffled, counts, valid: counts.every(x => x.ones > 0 && x.ones < x.n) };
}
const short = row => ({ cue: row.cue, opponent: row.opponent, candidate: row.candidate,
  action: row.action, reward: row.reward, bit: row.decodedBit, dnCounts: row.dnCounts,
  selectedHead: row.selectedHead, updateStep: row.updateStep });
function summary(rows) {
  const observed = rows.filter(row => row.cue !== null);
  const messaged = observed.filter(row => row.bit !== null);
  const correctProposal = row => row.candidate === (row.opponent + 1) % 3;
  const positive = messaged.filter(correctProposal), negative = messaged.filter(row => !correctProposal(row));
  return { wins: rows.filter(row => row.reward === 1).length,
    approvals: observed.filter(row => row.bit === 1).length,
    rejections: observed.filter(row => row.bit === 0).length,
    changed: observed.filter(row => row.action !== row.candidate).length,
    helpfulChanges: observed.filter(row => row.action !== row.candidate && row.reward === 1 &&
      row.candidate !== (row.opponent + 1) % 3).length,
    harmfulChanges: observed.filter(row => row.action !== row.candidate && row.reward !== 1 &&
      correctProposal(row)).length,
    adviceBalancedAccuracy: positive.length && negative.length ?
      (positive.filter(row => row.bit === 1).length / positive.length +
        negative.filter(row => row.bit === 0).length / negative.length) / 2 : null,
    rewardedUpdates: observed.filter(row => row.reward === 1 && row.updateStep !== 0).length,
    byCue: [0, 1, 2].map(cue => {
      const xs = observed.filter(row => row.cue === cue);
      return { n: xs.length, wins: xs.filter(row => row.reward === 1).length,
        rejections: xs.filter(row => row.bit === 0).length,
        rewardedUpdates: xs.filter(row => row.reward === 1 && row.updateStep !== 0).length };
    }) };
}
async function runPair(teacherSeed, studentSeed) {
  const trained = trainTeacher(teacherSeed);
  const result = { teacherSeed, studentSeed,
    teacher: { wins: trained.wins, byCue: trained.byCue, valid: trained.valid },
    source: null, checkpoints: [], training: {}, validity: {}, passed: false };
  if (!trained.valid) { result.status = 'teacher_gate_failed'; return result; }
  const tape = sourceTape(trained);
  result.source = { counts: tape.counts, valid: tape.valid };
  if (!tape.valid) { result.status = 'source_gate_failed'; return result; }
  const arms = Object.fromEntries(names.map(name => [name, freshStudent(studentSeed)]));
  const dummy = Object.fromEntries(['untrained', 'shuffled', 'generic'].map(name =>
    [name, freshStudent(studentSeed)]));
  const initial = studentWeights(arms.alone.agent);
  assert.ok(Object.values(arms).every(s => studentWeights(s.agent) === initial &&
    s.net.m.every(x => x === 1) && JSON.stringify([...s.net.v]) ===
      JSON.stringify([...arms.alone.net.v])));
  for (const name of names) result.training[name] = [];
  const correctSource = newTeacher(trained), untrainedSource = newTeacher(trained, true);
  const priorCorrectReject = [0, 0, 0], genericReject = [0, 0, 0];
  const live = [correctSource, untrainedSource, ...Object.values(arms), ...Object.values(dummy)];
  async function checkpoint(rounds) {
    const before = stateHash(live);
    const pairs = [[arms.correct, arms.alone], [arms.untrained, arms.shuffled],
      [arms.generic, arms.alone]];
    const frozen = [];
    for (const pair of pairs) frozen.push(await frozenStudentCheckpoint({ sub,
      students: pair, studentSeed, featureMean, featureStd, makeOpponent }));
    assert.equal(stateHash(live), before, 'frozen checkpoint altered live state');
    const schedule = JSON.stringify(frozen[0].schedule);
    assert.ok(frozen.every(x => x.disconnected && x.weightsUnchanged &&
      JSON.stringify(x.schedule) === schedule));
    const wins = { correct: frozen[0].wins[0], alone: frozen[0].wins[1],
      untrained: frozen[1].wins[0], shuffled: frozen[1].wins[1],
      generic: frozen[2].wins[0] };
    const byCue = { correct: frozen[0].byCue[0], alone: frozen[0].byCue[1],
      untrained: frozen[1].byCue[0], shuffled: frozen[1].byCue[1],
      generic: frozen[2].byCue[0] };
    assert.equal(frozen[2].wins[1], wins.alone, 'duplicate Alone evaluation disagreed');
    const rejectionGap = priorCorrectReject.map((value, cue) => value - genericReject[cue]);
    assert.ok(rejectionGap.every(value => value >= 0 && value <= 1),
      'causal generic exposure matching failed');
    result.checkpoints.push({ rounds, wins, byCue, rejectionGap,
      messagesAbsent: true, weightsUnchanged: true });
  }
  await checkpoint(0);
  assert.ok(Object.values(result.checkpoints[0].wins).every(x => x === 24));
  for (let episode = 0; episode < 12; episode++) {
    const games = Object.fromEntries([...names, ...Object.keys(dummy).map(x => `dummy_${x}`)]
      .map(name => [name, makeOpponent(41000 + episode)]));
    for (const name of names) arms[name].game = games[name];
    for (const name of Object.keys(dummy)) dummy[name].game = games[`dummy_${name}`];
    for (let r = 0; r < 24; r++) {
      const index = episode * 24 + r;
      const key = studentSeed * 10000000 + episode * 1000 + r;
      const teacherKey = teacherSeed * 10000000 + episode * 1000 + r;
      correctSource.key = untrainedSource.key = teacherKey;
      let genericBit = null;
      const genericRound = comparisonRound(null, dummy.generic, arms.generic, key, true,
        ({ cue }) => {
          genericBit = Number(priorCorrectReject[cue] <= genericReject[cue]);
          if (genericBit === 0) genericReject[cue]++;
          return genericBit;
        });
      while (!genericRound.done) genericRound.step();
      const correctRound = comparisonRound(correctSource, arms.alone, arms.correct, key, true);
      while (!correctRound.done) correctRound.step();
      const cue = correctRound.rows[1].cue;
      if (cue !== null && correctRound.rows[1].decodedBit === 0) priorCorrectReject[cue]++;
      const untrainedRound = comparisonRound(untrainedSource, dummy.untrained,
        arms.untrained, key, true);
      while (!untrainedRound.done) untrainedRound.step();
      const shuffledRound = comparisonRound(null, dummy.shuffled, arms.shuffled, key, true,
        () => tape.shuffled[index]);
      while (!shuffledRound.done) shuffledRound.step();
      const recorded = { alone: correctRound.rows[0], correct: correctRound.rows[1],
        untrained: untrainedRound.rows[1], shuffled: shuffledRound.rows[1],
        generic: genericRound.rows[1] };
      const opponent = recorded.alone.opponent;
      for (const name of names) {
        const row = recorded[name];
        assert.equal(row.opponent, opponent, 'opponent mismatch');
        assert.equal(row.selectedHead, row.cue === null ? 3 : row.cue);
        if (name === 'alone') assert.ok(row.decodedBit === null && row.dnCounts.every(x => x === 0));
        else if (cue !== null) {
          const bit = row.decodedBit;
          assert.ok(bit === 0 || bit === 1);
          assert.ok(row.dnCounts[bit] > row.dnCounts[1 - bit]);
        }
        result.training[name].push(short(row));
      }
      if (CHECKPOINT_ROUNDS.includes(index + 1)) await checkpoint(index + 1);
    }
  }
  assert.equal(teacherWeights(correctSource.agent), teacherWeights(trained.agent));
  assert.ok(live.every(x => x.net.m.every(value => value === 1)));
  if (stage === 'initial') {
    const prior = JSON.parse(await readFile(file('data/rps_challenge_learning_speed.json')));
    assert.deepEqual(result.checkpoints.map(x => [x.rounds, x.wins.alone, x.wins.correct]),
      prior.checkpoints.map(x => [x.rounds, ...x.wins]), 'original two-arm result did not reproduce');
  }
  result.trainingSummary = Object.fromEntries(names.map(name => [name, summary(result.training[name])]));
  result.validity = { matchedInitialization: true, matchedOpponents: true,
    dnDecoded: true, cueRouting: true, weightIntegrity: true, checkpointIsolation: true,
    genericExposureMatched: true, originalTwoArmReproduced: stage === 'initial' };
  const first58 = name => result.checkpoints.find(x => x.wins[name] >= 58)?.rounds ?? null;
  result.first58 = Object.fromEntries(names.map(name => [name, first58(name)]));
  const early = result.checkpoints.filter(x => [6, 12, 24, 48].includes(x.rounds));
  result.earlyMean = Object.fromEntries(names.map(name => [name,
    early.reduce((sum, x) => sum + x.wins[name], 0) / early.length]));
  const controls = names.filter(name => name !== 'correct');
  result.difference = Object.fromEntries(controls.map(name =>
    [name, result.earlyMean.correct - result.earlyMean[name]]));
  result.passed = controls.every(name => result.difference[name] >= 4 &&
    (result.first58[name] === null || result.first58.correct < result.first58[name])) &&
    result.first58.correct !== null && result.first58.correct <= 48 &&
    result.checkpoints.at(-1).wins.correct >= 58;
  result.status = result.passed ? 'passed_initial_or_confirmation_pair' : 'failed_teacher_specific_speed';
  return result;
}
const artifact = { protocol: { stage, pairs, config, studentRounds: 288,
  checkpoints: CHECKPOINT_ROUNDS, teacherAndStudentTrainOpponentSeeds: [41000, 41011],
  evalOpponentSeeds: [51000, 51002], proficiency: 58,
  primary: 'mean frozen wins at 6,12,24,48: correct >= each control +4; first 58 by <=48 and earlier than each control; final >=58',
  generic: 'one-cue-occurrence lag behind completed correct-arm per-cue rejection counts; gap 0..1 at every checkpoint',
  shuffle: 'within-cue pre-run trained-neural source tape using balanced public candidates and fixed RNG; preserves source-tape cue frequencies',
  confirmation: 'run 915/10915 and 916/10916 only if 830/10830 passes every initial gate' },
  pairs: [], passed: false, status: 'not_started' };
for (const [teacherSeed, studentSeed] of pairs) {
  const result = await runPair(teacherSeed, studentSeed);
  artifact.pairs.push(result);
  if (!result.passed) break;
}
artifact.passed = artifact.pairs.length === pairs.length && artifact.pairs.every(x => x.passed);
artifact.status = artifact.passed ? 'passed' : artifact.pairs.at(-1).status;
const handle = await open(output, 'wx');
try { await handle.writeFile(JSON.stringify(artifact, null, 2) + '\n'); }
finally { await handle.close(); }
console.log(JSON.stringify({ stage, status: artifact.status,
  pairs: artifact.pairs.map(x => ({ teacher: x.teacher, source: x.source,
    first58: x.first58, earlyMean: x.earlyMean, difference: x.difference,
    checkpoints: x.checkpoints.map(c => [c.rounds, c.wins]), passed: x.passed })) }));
