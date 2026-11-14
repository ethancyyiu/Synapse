// Opt-in staged no-teaching prerequisite. Each stage writes a new result once.
// node tools/check_rps_noisy_guided_prerequisite.mjs receiver
// node tools/check_rps_noisy_guided_prerequisite.mjs learn
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { open, readFile } from 'node:fs/promises';
import { Network, rng, wiring } from '../js/sim.js';
import { Opponent, result } from '../js/rps.js';
import { ChallengeCueHeadBandit, studentWeights } from '../js/rps_challenge_students.js';

const stage = process.argv[2];
if (!['receiver', 'learn'].includes(stage)) throw Error('Use receiver or learn stage');
const path = name => new URL(`../data/${name}`, import.meta.url);
const read = async name => JSON.parse(await readFile(path(name), 'utf8'));
const writeOnce = async (name, value) => {
  const file = await open(path(name), 'wx');
  try { await file.writeFile(`${JSON.stringify(value, null, 2)}\n`); }
  finally { await file.close(); }
};
const receiverFile = 'rps_noisy_guided_receiver.json';
const learningFile = 'rps_noisy_guided_learning.json';
const seeds = [10917, 10918], teacherSeeds = [861, 862];
const checkpoints = [0, 6, 12, 24, 48, 96, 288];
const targets = [3141, 3165], count = (xs, x) => xs.filter(y => y === x).length;
const avg = xs => xs.reduce((s, x) => s + x, 0) / xs.length;
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const sub = await read('subgraph.json');
const { featureMean, featureStd } = (await read('rps_contextual_bandit.json')).design;
const centroids = (await read('rps_readout_information.json')).centroids;
const prior = await read('rps_noisy_reward_gate_prerequisites.json');
assert.deepEqual(prior.teachers.map(x => x.seed), teacherSeeds);
assert.equal(prior.gates.teacher, true);
assert.equal(prior.gates.source, true);
assert.ok(prior.source.every(x => x.byCue.every(c => c.sourceGate &&
  c.accuracy >= .80 && c.accuracy - c.shuffledP95 >= .15 &&
  c.accuracy - c.untrainedAccuracy >= .15)));
const wireCache = new Map();
function fresh(seed, params = null, explore = false) {
  if (!wireCache.has(seed)) wireCache.set(seed, wiring(sub, 'real', seed));
  const net = new Network(sub, wireCache.get(seed), { seed });
  const agent = new ChallengeCueHeadBandit(net, seed, featureMean, featureStd);
  if (params) {
    params.heads.forEach((h, i) => {
      h.W.forEach((row, action) => agent.heads[i].W[action].set(row));
      agent.heads[i].b.set(h.b);
    });
    agent.baseline = params.baseline;
  }
  agent.explore = explore;
  if (!explore) agent.readoutLearn = false;
  return { net, agent };
}
const snapshot = agent => JSON.parse(studentWeights(agent));
const decodeCue = rates => {
  const features = rates.map(Math.log1p);
  const distances = centroids.map(c => c.reduce((s, x, i) => s + (features[i] - x) ** 2, 0));
  return distances.indexOf(Math.min(...distances));
};
const opponent = seed => new Opponent({ strategy: 'previous_opponent', offset: 2, noise: .5, seed });

function receiverRound(net, agent, cue, key, bit) {
  const vpn = Array.from(net.laneVpn[cue].slice(2)), dnCounts = [0, 0];
  net.rand = rng(key + 2); agent.rand = rng(key + 501);
  net.setDrive(vpn, 150);
  let row;
  for (let t = 0; t < 680; t++) {
    if (t === 50) net.resetCounts();
    if (t === 240) {
      net.setDrive(vpn, 0); agent.decide(net);
      row = { cue, estimatedCue: agent.activeHead, rates: [...agent.rates],
        recognized: decodeCue(agent.rates) === cue, bit, dnCounts: null, decodedBit: null };
      if (bit !== null) net.setDrive([targets[bit]], 300);
    }
    if (t === 280) {
      if (bit !== null) net.setDrive([targets[bit]], 0);
      row.dnCounts = [...dnCounts];
      row.decodedBit = dnCounts[0] === dnCounts[1] ? null : Number(dnCounts[1] > dnCounts[0]);
      net.resetCounts();
    }
    net.step();
    if (t >= 240 && t < 280) for (let j = 0; j < net.nSpikes; j++) {
      if (net.spikes[j] === targets[0]) dnCounts[0]++;
      else if (net.spikes[j] === targets[1]) dnCounts[1]++;
    }
  }
  return row;
}

if (stage === 'receiver') {
  const probes = [];
  for (const seed of seeds) for (let cue = 0; cue < 3; cue++)
    for (let repeat = 0; repeat < 12; repeat++) for (const bit of [null, 0, 1]) {
      const { net, agent } = fresh(seed);
      const before = studentWeights(agent), key = seed * 10000000 +
        (300 + 12 * cue + repeat) * 1000;
      net.resetState();
      const first = receiverRound(net, agent, cue, key, bit);
      const retained = [...first.dnCounts];
      net.resetState();
      const second = receiverRound(net, agent, (cue + 2) % 3, key + 1, null);
      probes.push({ seed, cue, repeat, bit, first, second,
        bufferPreserved: same(first.dnCounts, retained),
        weightsUnchanged: studentWeights(agent) === before && net.m.every(x => x === 1) });
    }
  const summary = seeds.flatMap(seed => [0, 1, 2].map(cue => {
    const xs = probes.filter(p => p.seed === seed && p.cue === cue);
    const base = xs.filter(p => p.bit === null);
    return { seed, cue, branches: [null, 0, 1].map(bit => {
      const ys = xs.filter(p => p.bit === bit);
      const paired = ys.map(p => base.find(b => b.repeat === p.repeat));
      return { bit, n: ys.length,
        firstRecognized: count(ys.map(p => p.first.recognized), true),
        secondRecognized: count(ys.map(p => p.second.recognized), true),
        firstNewErrors: bit === null ? null : count(ys.map((p, i) =>
          paired[i].first.recognized && !p.first.recognized), true),
        secondNewErrors: bit === null ? null : count(ys.map((p, i) =>
          paired[i].second.recognized && !p.second.recognized), true),
        firstRescues: bit === null ? null : count(ys.map((p, i) =>
          !paired[i].first.recognized && p.first.recognized), true),
        secondRescues: bit === null ? null : count(ys.map((p, i) =>
          !paired[i].second.recognized && p.second.recognized), true),
        decoded: bit === null ? null : count(ys.map(p => p.first.decodedBit === bit), true),
        targetDominance: bit === null ? null : count(ys.map(p =>
          p.first.dnCounts[bit] > p.first.dnCounts[1 - bit]), true),
        intendedMeanGain: bit === null ? null : avg(ys.map(p => p.first.dnCounts[bit])) -
          avg(base.map(p => p.first.dnCounts[bit])) };
    }) };
  }));
  const gates = {
    probesAndWeights: probes.length === 216 && probes.every(p =>
      p.bufferPreserved && p.weightsUnchanged),
    recognition: summary.every(s => s.branches.every(b =>
      b.n === 12 && b.firstRecognized >= 10 && b.secondRecognized >= 10)),
    noRecognitionLoss: summary.every(s => s.branches.slice(1).every(b =>
      b.firstRecognized >= s.branches[0].firstRecognized &&
      b.secondRecognized >= s.branches[0].secondRecognized)),
    bit: summary.every(s => s.branches.slice(1).every(b =>
      b.decoded === 12 && b.targetDominance === 12 && b.intendedMeanGain >= 1)),
  };
  const passed = Object.values(gates).every(Boolean);
  const artifact = { protocol: { source: 'docs/lab-notebook.md prospective epsilon-greedy amendment',
    seeds, teacherSeeds, branches: [null, 0, 1], cues: [0, 1, 2], repeats: 12,
    secondCue: '(first+2)%3', firstKey: 's*10000000+(300+12*cue+repeat)*1000',
    secondKey: 'firstKey+1', dnTargets: targets, dnHz: 300, dnWindowMs: [240, 280],
    cueHz: 150, cueWindowMs: [0, 240], fixedCentroidDecoder: true },
    status: passed ? 'passed_receiver' : 'stopped_receiver', passed,
    gates, summary, probes };
  await writeOnce(receiverFile, artifact);
  console.log(JSON.stringify({ status: artifact.status, gates, summary }));
  process.exit(0);
}

const receipt = await read(receiverFile);
if (!receipt.passed || receipt.status !== 'passed_receiver' ||
    !Object.values(receipt.gates).every(Boolean) || receipt.probes.length !== 216)
  throw Error('Locked receiver did not pass; independent training is prohibited');

// The opponent process depends on its own previous move and fixed random draws,
// never on the student's selected action. Check held-out coverage before training.
const planned = new Map();
for (const seed of seeds) {
  const sequences = [0, 1, 2].map(i => {
    const game = opponent(52000 + 1000 * seed + 3 * i), rows = [];
    for (let r = 0; r < 24; r++) {
      const cue = game.observe().previousOpponent, played = game.play(0);
      rows.push({ cue, opponent: played.opponent });
    }
    return rows;
  });
  planned.set(seed, sequences);
}

function round(net, agent, game, seed, episodeKey, r, learn, expected = null) {
  net.resetState();
  const cue = game.observe().previousOpponent;
  const vpn = cue === null ? [] : Array.from(net.laneVpn[cue].slice(2));
  const key = seed * 10000000 + episodeKey * 1000 + r;
  net.rand = rng(key + 2); agent.rand = rng(key + 501);
  if (vpn.length) net.setDrive(vpn, 150);
  let row;
  for (let t = 0; t < 680; t++) {
    if (t === 50) net.resetCounts();
    if (t === 240) {
      if (vpn.length) net.setDrive(vpn, 0);
      const action = agent.decide(net);
      row = { cue, estimatedCue: agent.activeHead, action, rates: [...agent.rates],
        scores: [...agent.scores], laneCounts: [...agent.laneCounts],
        dnCounts: [0, 0], updateStep: 0 };
      assert.equal(row.estimatedCue, cue === null ? 3 : cue, 'VPN cue routing failed');
    }
    if (t === 280) {
      const played = game.play(row.action);
      agent.reward(net, played.reward);
      row.opponent = played.opponent; row.reward = played.reward;
      assert.equal(row.reward, result(row.action, row.opponent));
      if (expected) assert.deepEqual({ cue, opponent: row.opponent }, expected);
      net.resetCounts();
    }
    if (t === 620) {
      const old = snapshot(agent);
      const step = learn ? agent.alpha * (row.reward - row.scores[row.action]) : 0;
      agent.afterDopamine(net, learn);
      row.updateStep = step;
      const next = snapshot(agent);
      assert.equal(next.baseline, old.baseline);
      for (let h = 0; h < 4; h++) for (let a = 0; a < 3; a++) {
        const selected = learn && h === row.estimatedCue && a === row.action;
        const expectedB = old.heads[h].b[a] + (selected ? step : 0);
        assert.ok(Math.abs(next.heads[h].b[a] - expectedB) < 1e-10);
        for (let j = 0; j < 3; j++) {
          const expectedW = old.heads[h].W[a][j] + (selected ? step * agent.f[j] : 0);
          assert.ok(Math.abs(next.heads[h].W[a][j] - expectedW) < 1e-10);
        }
      }
      assert.ok(net.m.every(x => x === 1), 'KC-to-MBON weights changed');
    }
    net.step();
  }
  return row;
}

const stateStamp = (net, agent, game) => {
  const hash = createHash('sha256');
  hash.update(studentWeights(agent));
  hash.update(JSON.stringify({ t: net.t, round: game.round, observe: game.observe(),
    netRand: String(net.rand), agentRand: String(agent.rand) }));
  for (const array of [net.v, net.g, net.refr, net.trace, net.count, net.lastSpike,
    ...net.ring, net.drive, net.noise, net.noiseKick, net.m])
    hash.update(Buffer.from(array.buffer, array.byteOffset, array.byteLength));
  return hash.digest('hex');
};
function summarize(rows) {
  const observed = rows.filter(x => x.cue !== null);
  return { wins: count(rows.map(x => x.reward), 1), observed: observed.length,
    expectedOptimal: count(observed.map(x => x.action === x.cue), true),
    byCue: [0, 1, 2].map(cue => {
      const xs = observed.filter(x => x.cue === cue);
      return { cue, n: xs.length, wins: count(xs.map(x => x.reward), 1),
        expectedOptimal: count(xs.map(x => x.action === cue), true),
        actions: [0, 1, 2].map(a => count(xs.map(x => x.action), a)) };
    }) };
}
function frozen(seed, params) {
  const rows = [];
  for (let i = 0; i < 3; i++) {
    const { net, agent } = fresh(seed, params, false);
    const game = opponent(52000 + 1000 * seed + 3 * i);
    for (let r = 0; r < 24; r++)
      rows.push({ evalIndex: i, round: r, ...round(net, agent, game, seed,
        100 + i, r, false, planned.get(seed)[i][r]) });
    assert.equal(studentWeights(agent), JSON.stringify(params));
  }
  return { summary: summarize(rows), rows, weightsUnchanged: true, disconnected: true };
}
const artifact = { protocol: { source: 'docs/lab-notebook.md prospective epsilon-greedy amendment',
  seeds, teacherSeeds, opponent: 'previous_opponent offset 2 noise 0.5',
  actionSelection: 'existing epsilon-greedy proposal, epsilon 0.20, tie to first maximum',
  balancedOverride: false, budget: '12x24', checkpoints,
  trainingOpponentSeed: '41000+1000*s+e',
  evaluationOpponentSeed: '52000+1000*s+3*i',
  networkRandom: 'rng(s*10000000+episodeKey*1000+round+2)',
  actionRandom: 'rng(s*10000000+episodeKey*1000+round+501)',
  receiverArtifact: receiverFile }, students: [], sourceGateFromArchive: true,
  status: 'not_started', passed: false };
for (const seed of seeds) {
  let params = null, rounds = 0, failed = null;
  const record = { seed, checkpoints: [], training: [], gates: {} };
  const initial = frozen(seed, snapshot(fresh(seed).agent));
  record.checkpoints.push({ rounds: 0, ...initial });
  record.gates.coverage = initial.summary.observed === 69 &&
    initial.summary.byCue.every(c => c.n >= 15);
  record.gates.initialHeadroom = initial.summary.expectedOptimal <= 36;
  if (!record.gates.coverage || !record.gates.initialHeadroom)
    failed = 'initial_or_coverage';
  for (let e = 0; e < 12 && !failed; e++) {
    const { net, agent } = fresh(seed, params, true);
    const game = opponent(41000 + 1000 * seed + e);
    for (let r = 0; r < 24 && !failed; r++) {
      const row = round(net, agent, game, seed, e, r, true);
      record.training.push({ episode: e, round: r, ...row });
      rounds++;
      if (checkpoints.includes(rounds)) {
        const stamp = stateStamp(net, agent, game);
        const netRand = net.rand, actionRand = agent.rand, opponentRand = game.rand;
        const evalResult = frozen(seed, snapshot(agent));
        assert.equal(stateStamp(net, agent, game), stamp,
          'frozen evaluation altered training state');
        assert.equal(net.rand, netRand);
        assert.equal(agent.rand, actionRand);
        assert.equal(game.rand, opponentRand);
        record.checkpoints.push({ rounds, ...evalResult });
        if (rounds === 12) {
          record.gates.earlyHeadroom = evalResult.summary.expectedOptimal <= 55;
          if (!record.gates.earlyHeadroom) failed = 'early_headroom';
        }
      }
    }
    params = snapshot(agent);
  }
  if (!failed) {
    const first = record.checkpoints[0].summary.expectedOptimal;
    const last = record.checkpoints.at(-1).summary;
    record.gates.finalLearning = last.byCue.every(c => c.expectedOptimal / c.n >= .70) &&
      (last.expectedOptimal - first) / 69 >= .20;
    if (!record.gates.finalLearning) failed = 'independent_learning';
  }
  record.trainingRounds = rounds;
  record.status = failed ?? 'passed_independent_learning';
  record.passed = !failed;
  artifact.students.push(record);
  if (failed) break;
}
artifact.passed = artifact.students.length === seeds.length && artifact.students.every(s => s.passed);
artifact.status = artifact.passed ? 'passed_no_teaching_prerequisite' :
  `stopped_${artifact.students.at(-1).status}`;
await writeOnce(learningFile, artifact);
console.log(JSON.stringify({ status: artifact.status, students: artifact.students.map(s =>
  ({ seed: s.seed, status: s.status, gates: s.gates,
    curve: s.checkpoints.map(c => ({ rounds: c.rounds, wins: c.summary.wins,
      expectedOptimal: c.summary.expectedOptimal, byCue: c.summary.byCue })) })) }));
