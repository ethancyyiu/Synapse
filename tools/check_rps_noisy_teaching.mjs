// Locked, three-seed noisy Strategy A early-learning comparison.
// node tools/check_rps_noisy_teaching.mjs check
// node tools/check_rps_noisy_teaching.mjs run  (write once)
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, open, access } from 'node:fs/promises';
import { Network, rng, wiring } from '../js/sim.js';
import { RpsContextualBandit, RpsEpisode, result } from '../js/rps.js';
import { PreferenceStudent, studentWeights } from '../js/rps_challenge_students.js';
import { challengeOpponent } from '../js/rps_challenge_opponent.js';

const stage = process.argv[2];
if (!['check', 'run'].includes(stage)) throw Error('Use check or run');
const teacherSeed = 931, studentSeeds = [12341, 12342, 12343];
const config = { mode: 'preset', offset: 1, noise: 0.3 };
const checkpoints = [0, 12, 24], arms = ['trained', 'alone', 'shuffled', 'untrained'];
const DN = [3141, 3165], codes = [[1, 0], [0, 1], [1, 1]];
const output = new URL('../data/rps_noisy_teaching.json', import.meta.url);
if (stage === 'run') {
  try { await access(output); throw Error('Refusing to overwrite noisy teaching result'); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
}
const read = async name => JSON.parse(await readFile(new URL(`../data/${name}`, import.meta.url)));
const [sub, calibration, pilot, protocolText] = await Promise.all([
  read('subgraph.json'), read('rps_contextual_bandit.json'),
  read('rps_noisy_demo_feasibility.json'),
  readFile(new URL('../docs/noisy-rps-teaching-protocol.md', import.meta.url)),
]);
const { featureMean, featureStd } = calibration.design;
const protocolSha256 = createHash('sha256').update(protocolText).digest('hex');
const makeNet = seed => new Network(sub, wiring(sub, 'real', seed), { seed });
const makeOpponent = seed => challengeOpponent(config, seed);
const teacherWeights = agent => JSON.stringify({ W: agent.W.map(row => [...row]),
  b: [...agent.b], baseline: agent.baseline });
const decode = counts => {
  const bits = counts.map(n => Number(n > 0));
  const match = codes.findIndex(code => code.every((bit, i) => bit === bits[i]));
  return match < 0 ? null : match;
};
const newStudent = seed => {
  const net = makeNet(seed);
  return { net, agent: new PreferenceStudent(net, seed, featureMean, featureStd) };
};

function studentRound(student, game, key, target, learn) {
  const { net, agent } = student, cue = game.observe().previousOpponent;
  const vpn = cue === null ? [] : Array.from(net.laneVpn[cue].slice(2));
  const dnCounts = [0, 0];
  net.resetState(); net.rand = rng(key + 2); agent.rand = rng(key + 501);
  if (vpn.length) net.setDrive(vpn, 150);
  let action = null, scores = null, reward = null, opponent = null, decoded = null;
  let rewardStep = 0, imitation = 0;
  for (let t = 0; t < 680; t++) {
    if (t === 50) net.resetCounts();
    if (t === 240) {
      if (vpn.length) net.setDrive(vpn, 0);
      action = agent.decide(net); scores = [...agent.scores];
      assert.equal(agent.activeHead, cue === null ? 3 : cue, 'VPN cue routing');
      if (target !== null) DN.forEach((id, i) => { if (codes[target][i]) net.setDrive([id], 300); });
    }
    if (t === 280) {
      assert.notEqual(action, null, 'student must decide before opponent plays');
      net.setDrive(DN, 0);
      decoded = decode(dnCounts);
      assert.equal(decoded, target, 'decoded message must match measured DN spikes');
      ({ opponent, reward } = game.play(action));
      assert.equal(reward, result(action, opponent));
      agent.reward(net, reward); net.resetCounts();
    }
    if (t === 620) {
      rewardStep = learn ? agent.alpha * (reward - scores[action]) : 0;
      agent.afterDopamine(net, learn);
      if (learn && decoded !== null) imitation = agent.imitate(decoded);
    }
    net.step();
    if (t >= 240 && t < 280) for (let i = 0; i < net.nSpikes; i++) {
      if (net.spikes[i] === DN[0]) dnCounts[0]++;
      else if (net.spikes[i] === DN[1]) dnCounts[1]++;
    }
  }
  assert.ok(net.m.every(value => value === 1), 'KC→MBON multipliers stay frozen');
  return { cue, action, opponent, reward, target, decoded, dnCounts,
    rewardStep, imitation };
}

function teacherEpisode(net, agent, opponentSeed, episodeKey, learn) {
  const ep = new RpsEpisode(net, agent, makeOpponent(opponentSeed), { rounds: 24, learn });
  while (!ep.done) {
    if (ep.t % 400 === 0) {
      const round = ep.t / 400, key = teacherSeed * 10000000 + episodeKey * 1000 + round;
      net.rand = rng(key + 1); agent.rand = rng(key + 501);
    }
    ep.step();
  }
  return ep.events.map(event => ({ cue: event.observation.previousOpponent,
    action: event.action, opponent: event.opponent, reward: event.reward }));
}
function trainTeacher() {
  const net = makeNet(teacherSeed);
  const agent = new RpsContextualBandit(net, { seed: teacherSeed, featureMean, featureStd,
    readoutLearn: true });
  for (let ep = 0; ep < 12; ep++) teacherEpisode(net, agent, 41000 + ep, ep, true);
  const trained = teacherWeights(agent);
  agent.explore = false;
  const frozen = [];
  for (let ep = 0; ep < 3; ep++) frozen.push(...teacherEpisode(net, agent, 51000 + ep, 100 + ep, false));
  assert.equal(teacherWeights(agent), trained, 'frozen teacher weights');
  assert.ok(net.m.every(value => value === 1), 'teacher KC→MBON multipliers');
  const wins = frozen.filter(row => row.reward === 1).length;
  assert.equal(wins, pilot.teacher.wins, 'exact archived teacher replay');
  return { state: JSON.parse(trained), wins, hash: createHash('sha256').update(trained).digest('hex') };
}
function sourceTape(trainedState, untrained = false) {
  const net = makeNet(teacherSeed);
  const agent = new RpsContextualBandit(net, { seed: teacherSeed, featureMean, featureStd,
    readoutLearn: false });
  if (!untrained) {
    trainedState.W.forEach((row, action) => agent.W[action].set(row));
    agent.b.set(trainedState.b); agent.baseline = trainedState.baseline;
  }
  agent.explore = false;
  const initial = teacherWeights(agent), game = makeOpponent(41000), rows = [];
  for (let round = 0; round < 24; round++) {
    const cue = game.observe().previousOpponent;
    const vpn = cue === null ? [] : Array.from(net.laneVpn[cue].slice(2));
    net.rand = rng(teacherSeed * 10000000 + round + 1);
    if (vpn.length) net.setDrive(vpn, 150);
    let preference = null, scores = null, rates = null;
    for (let t = 0; t < 680; t++) {
      if (t === 50) net.resetCounts();
      if (t === 240) {
        if (vpn.length) net.setDrive(vpn, 0);
        agent.decide(net); scores = [...agent.scores]; rates = [...agent.rates];
        if (cue !== null) preference = scores.indexOf(Math.max(...scores));
      }
      net.step();
    }
    assert.notEqual(scores, null, 'teacher scores before opponent move');
    const played = game.play(0); // Only advances the public opponent sequence after advice.
    rows.push({ cue, preference, scores, rates, opponent: played.opponent });
  }
  assert.equal(teacherWeights(agent), initial, 'teacher source frozen');
  assert.ok(net.m.every(value => value === 1));
  return rows;
}
function shuffledTargets(tape) {
  const random = rng(2026111001), history = [];
  return tape.map((row, index) => {
    if (row.preference === null) return { target: null, sourceIndex: null };
    const sourceIndex = history.length ? history[Math.floor(random() * history.length)] : index;
    const target = tape[sourceIndex].preference;
    history.push(index); // The current teacher preference is available only to future rounds.
    assert.ok(sourceIndex <= index, 'no future message in causal shuffle');
    return { target, sourceIndex };
  });
}
function frozen(seed, weights) {
  const s = newStudent(seed), saved = JSON.parse(weights);
  saved.heads.forEach((head, h) => {
    head.W.forEach((row, a) => s.agent.heads[h].W[a].set(row));
    s.agent.heads[h].b.set(head.b);
  });
  s.agent.baseline = saved.baseline;
  s.agent.explore = false; s.agent.readoutLearn = false;
  const start = studentWeights(s.agent), rows = [];
  for (let ep = 0; ep < 3; ep++) {
    const game = makeOpponent(51000 + ep);
    for (let round = 0; round < 24; round++) {
      const key = seed * 10000000 + (100 + ep) * 1000 + round;
      rows.push(studentRound(s, game, key, null, false));
    }
  }
  assert.equal(studentWeights(s.agent), start, 'frozen evaluation unchanged');
  assert.ok(rows.every(row => row.decoded === null && row.dnCounts.every(n => n === 0) &&
    row.imitation === 0 && row.rewardStep === 0));
  return { wins: rows.filter(row => row.reward === 1).length,
    rows: rows.map(({ cue, action, opponent, reward }) => ({ cue, action, opponent, reward })) };
}
function trainArm(seed, name, targets) {
  const s = newStudent(seed), initial = studentWeights(s.agent), game = makeOpponent(41000);
  const points = [{ rounds: 0, ...frozen(seed, initial) }], training = [];
  for (let round = 0; round < 24; round++) {
    const key = seed * 10000000 + round;
    training.push(studentRound(s, game, key, targets[round], true));
    if (checkpoints.includes(round + 1))
      points.push({ rounds: round + 1, ...frozen(seed, studentWeights(s.agent)) });
  }
  return { name, seed, initial, finalWeightHash: createHash('sha256').update(studentWeights(s.agent)).digest('hex'),
    training, checkpoints: points, synapsesFrozen: s.net.m.every(value => value === 1) };
}
const schedule = rows => rows.map(row => [row.cue, row.opponent]);

if (stage === 'check') {
  // One focused preflight: exact noisy opponent pairing, two measured DN codes, and no learning.
  for (const opponentSeed of [41000, 51000]) {
    const pair = [makeOpponent(opponentSeed), makeOpponent(opponentSeed)];
    for (let round = 0; round < 24; round++)
      assert.equal(pair[0].play(round % 3).opponent, pair[1].play((round + 1) % 3).opponent);
  }
  for (const target of [0, 1]) {
    const s = newStudent(studentSeeds[0]), game = makeOpponent(41000), initial = studentWeights(s.agent);
    game.play(0); // Establish an observed cue before the reward-free probe.
    const row = studentRound(s, game, studentSeeds[0] * 10000000 + 7000 + target, target, false);
    assert.equal(row.decoded, target);
    assert.ok(row.dnCounts.some(n => n > 0));
    assert.equal(studentWeights(s.agent), initial);
  }
  console.log('Focused correctness check: matched opponent draws, both measured DN codes, frozen weights, and decision-before-outcome timing pass');
  process.exit(0);
}

const artifact = { protocol: { teacherSeed, studentSeeds, config, checkpoints, arms,
  protocolSha256, teacherTrainingRounds: 288, studentTrainingRounds: 24,
  trainingOpponentSeed: 41000, frozenOpponentSeeds: [51000, 51001, 51002],
  earlyGate: 'round-12 trained wins >= strongest control + 4 for every seed' },
  teacher: null, source: null, pairs: [], validity: {}, gate: null, status: 'started' };
async function save() {
  const handle = await open(output, 'wx');
  try { await handle.writeFile(JSON.stringify(artifact, null, 2) + '\n'); }
  finally { await handle.close(); }
  console.log(JSON.stringify({ status: artifact.status, teacher: artifact.teacher?.wins,
    gate: artifact.gate, pairs: artifact.pairs.map(pair => ({ seed: pair.seed,
      round12: Object.fromEntries(arms.map(name => [name, pair.byArm[name].checkpoints[1].wins])),
      round24: Object.fromEntries(arms.map(name => [name, pair.byArm[name].checkpoints[2].wins])) })) }));
}
const teacher = trainTeacher();
artifact.teacher = { wins: teacher.wins, weightHash: teacher.hash, exactPilotReplay: true };
const trainedTape = sourceTape(teacher.state), untrainedTape = sourceTape(null, true);
const pilotTargets = pilot.student.messages.map(row => row.preference);
artifact.validity.sourceReplay = trainedTape.every((row, i) => row.preference === pilotTargets[i]);
if (!artifact.validity.sourceReplay) {
  artifact.status = 'stopped_source_replay'; await save(); process.exit(0);
}
const shuffled = shuffledTargets(trainedTape);
const changed = shuffled.filter((row, i) => row.target !== null &&
  row.target !== trainedTape[i].preference).length;
artifact.source = { trained: trainedTape, untrained: untrainedTape, shuffled, changed };
artifact.validity.shuffled = changed >= 6;
if (!artifact.validity.shuffled) {
  artifact.status = 'stopped_shuffled_control'; await save(); process.exit(0);
}
const targets = {
  trained: trainedTape.map(row => row.preference),
  alone: Array(24).fill(null),
  shuffled: shuffled.map(row => row.target),
  untrained: untrainedTape.map(row => row.preference),
};
for (const seed of studentSeeds) {
  const byArm = Object.fromEntries(arms.map(name => [name, trainArm(seed, name, targets[name])]));
  const initial = byArm.trained.initial;
  assert.ok(arms.every(name => byArm[name].initial === initial && byArm[name].synapsesFrozen));
  assert.ok(arms.every(name => byArm[name].checkpoints[0].wins === byArm.trained.checkpoints[0].wins));
  assert.ok(arms.every(name => JSON.stringify(schedule(byArm[name].training)) ===
    JSON.stringify(schedule(byArm.trained.training))));
  for (const cp of checkpoints) {
    const reference = byArm.trained.checkpoints.find(point => point.rounds === cp);
    assert.ok(arms.every(name => JSON.stringify(schedule(byArm[name].checkpoints.find(point => point.rounds === cp).rows)) ===
      JSON.stringify(schedule(reference.rows))));
  }
  for (const name of arms) {
    const rows = byArm[name].training;
    assert.ok(rows.every((row, i) => row.target === targets[name][i] &&
      (row.target === null ? row.decoded === null && row.dnCounts.every(n => n === 0) && row.imitation === 0 :
        row.decoded === row.target && row.dnCounts.some(n => n > 0) && row.imitation === 1)));
  }
  artifact.pairs.push({ seed, byArm });
}
artifact.validity.matched = true;
artifact.validity.messageDecoding = true;
artifact.validity.noFutureLeak = shuffled.every((row, i) => row.sourceIndex === null || row.sourceIndex <= i);
artifact.gate = artifact.pairs.every(pair => {
  const trained = pair.byArm.trained.checkpoints[1].wins;
  const strongest = Math.max(...arms.filter(name => name !== 'trained').map(name => pair.byArm[name].checkpoints[1].wins));
  return trained - strongest >= 4;
});
artifact.status = artifact.gate ? 'passed' : 'failed';
await save();
