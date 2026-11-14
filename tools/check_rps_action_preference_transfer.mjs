// Opt-in action-preference spike teaching; locked protocol in docs/lab-notebook.md.
// Run once in a copy without the write-once output artifact.
import assert from 'node:assert/strict';
import { readFile, open } from 'node:fs/promises';
import { Network, rng, wiring } from '../js/sim.js';
import { Opponent, RpsContextualBandit, RpsEpisode, result } from '../js/rps.js';
import { ChallengeCueHeadBandit, studentWeights } from '../js/rps_challenge_students.js';

const read = async name => JSON.parse(await readFile(new URL(`../data/${name}`, import.meta.url)));
const output = new URL('../data/rps_action_preference_transfer_corrected.json', import.meta.url);
const sub = await read('subgraph.json');
const { featureMean, featureStd } = (await read('rps_contextual_bandit.json')).design;
const oldCentroids = (await read('rps_readout_information.json')).centroids;
const DN = [3141, 3165], codes = [[1, 0], [0, 1], [1, 1]];
const seeds = [11000, 11001, 11002], teacherSeed = 830;
const arms = ['trained', 'alone', 'untrained', 'shuffled', 'generic'];
const checkpoints = [0, 6, 12, 24, 48, 96, 288];
const protocol = { teacherSeed, calibrationSeed: seeds[0], heldoutSeeds: seeds.slice(1),
  code: codes, dn: DN, dnHz: 300, dnWindow: [240, 280], cueHz: 150,
  cueWindow: [280, 520], countReset: 330, decision: 520, update: 620, end: 680,
  repeats: 12, probeConditions: ['none', 0, 1, 2],
  probeKey: 'seed*10000000+(100+48*cue+12*condition+repeat)*1000',
  trainingKey: 'seed*10000000+episode*1000+round',
  teacherTraining: '12*24 Strategy A, opponent seeds 41000+episode',
  heldout: '3*24 Strategy A, opponent seeds 51000..51002',
  studentRewardAlpha: .03, imitationAlpha: .03, epsilon: .20,
  checkpointRounds: checkpoints, arms,
  receiverFloor: '>=11/12 decoded per cue/action; >=10/12 pooled cue recognition per condition; VPN 12/12; <=2/12 loss vs none',
  sourceFloor: 'archived 61/72 teacher and >=70% per cue preference on source tape',
  aloneFloor: 'initial <=50/72; final >=58/72 and >=70% per observed cue for both',
  pilotGate: 'both seeds: mean frozen 6/12/24/48 >=4/72 over every control, first 58/72 <=48 and earlier than every control, final >=58/72 and >=70% per cue' };
const artifact = { protocol, status: 'started', teacher: null, receiver: null,
  synthetic: null, independent: null, source: null, pilot: null };
async function finish(status) {
  artifact.status = status;
  const file = await open(output, 'wx');
  try { await file.writeFile(`${JSON.stringify(artifact, null, 2)}\n`); }
  finally { await file.close(); }
  console.log(JSON.stringify({ status, teacher: artifact.teacher, receiver: artifact.receiver?.summary,
    synthetic: artifact.synthetic, independent: artifact.independent?.summary,
    pilot: artifact.pilot?.summary }));
}
const win = (a, b) => result(a, b) === 1;
const opponent = seed => new Opponent({ strategy: 'previous_opponent', offset: 1, seed });
const makeNet = seed => new Network(sub, wiring(sub, 'real', seed), { seed });
const teacherSnapshot = a => JSON.stringify({ W: a.W.map(w => [...w]), b: [...a.b], baseline: a.baseline });
function teacherCopy(trained, untrained = false) {
  const net = makeNet(teacherSeed);
  const agent = new RpsContextualBandit(net, { seed: teacherSeed, featureMean, featureStd, readoutLearn: false });
  if (!untrained) {
    trained.W.forEach((w, a) => agent.W[a].set(w));
    agent.b.set(trained.b); agent.baseline = trained.baseline;
  }
  agent.explore = false;
  return { net, agent };
}
function trainTeacher() {
  const net = makeNet(teacherSeed);
  const agent = new RpsContextualBandit(net, { seed: teacherSeed, featureMean, featureStd });
  const rows = [];
  for (let phase = 0; phase < 2; phase++) {
    agent.explore = phase === 0;
    for (let episode = 0; episode < (phase ? 3 : 12); episode++) {
      const ep = new RpsEpisode(net, agent, opponent((phase ? 51000 : 41000) + episode),
        { rounds: 24, learn: phase === 0 });
      while (!ep.done) {
        if (ep.t % 400 === 0) {
          const key = teacherSeed * 10000000 + (phase ? 100 + episode : episode) * 1000 + ep.t / 400;
          net.rand = rng(key + 1); agent.rand = rng(key + 501);
        }
        ep.step();
      }
      if (phase) rows.push(...ep.events);
    }
  }
  const wins = rows.filter(r => r.reward === 1).length;
  const byCue = [0, 1, 2].map(cue => {
    const x = rows.filter(r => r.observation.previousOpponent === cue);
    return { cue, wins: x.filter(r => r.reward === 1).length, n: x.length };
  });
  return { agent, result: { wins, byCue, synapsesFrozen: net.m.every(x => x === 1),
    archivedResultReproduced: wins === 61 && byCue.every(x => x.wins / x.n >= .70) } };
}
function decode(counts) {
  const bits = counts.map(x => Number(x > 0));
  return codes.findIndex(code => code.every((value, i) => value === bits[i]));
}
function oldCue(rates) {
  const f = rates.map(Math.log1p);
  const d = oldCentroids.map(c => c.reduce((s, v, j) => s + (v - f[j]) ** 2, 0));
  return d.indexOf(Math.min(...d));
}
function receiverProbe(seed, cue, condition, repeat) {
  const net = makeNet(seed), agent = new ChallengeCueHeadBandit(net, seed, featureMean, featureStd);
  agent.explore = false; agent.readoutLearn = false;
  const initial = studentWeights(agent);
  const key = seed * 10000000 + (100 + 48 * cue + 12 * condition + repeat) * 1000;
  const counts = [0, 0], vpn = Array.from(net.laneVpn[cue].slice(2));
  net.resetState(); net.rand = rng(key + 2); agent.rand = rng(key + 501);
  for (let t = 0; t < 680; t++) {
    if (t === 240 && condition > 0) DN.forEach((id, i) => {
      if (codes[condition - 1][i]) net.setDrive([id], 300);
    });
    if (t === 280) { net.setDrive(DN, 0); net.setDrive(vpn, 150); }
    if (t === 330) net.resetCounts();
    if (t === 520) { net.setDrive(vpn, 0); agent.decide(net); }
    net.step();
    if (t >= 240 && t < 280) for (let i = 0; i < net.nSpikes; i++) {
      if (net.spikes[i] === DN[0]) counts[0]++;
      else if (net.spikes[i] === DN[1]) counts[1]++;
    }
  }
  return { seed, cue, condition: condition ? condition - 1 : null, repeat, counts,
    decoded: decode(counts), routedCue: agent.activeHead, pooledCue: oldCue(agent.rates),
    rates: [...agent.rates], laneCounts: [...agent.laneCounts],
    weightsUnchanged: initial === studentWeights(agent) && net.m.every(x => x === 1) };
}
function receiverBatch(seed) {
  const rows = [];
  for (let cue = 0; cue < 3; cue++) for (let condition = 0; condition < 4; condition++)
    for (let repeat = 0; repeat < 12; repeat++) rows.push(receiverProbe(seed, cue, condition, repeat));
  const conditions = [0, 1, 2].flatMap(cue => [null, 0, 1, 2].map(message => {
    const x = rows.filter(r => r.cue === cue && r.condition === message);
    return { cue, message, n: x.length, decoded: x.filter(r => r.decoded === (message ?? -1)).length,
      pooled: x.filter(r => r.pooledCue === cue).length,
      vpn: x.filter(r => r.routedCue === cue).length,
      targetSpikes: x.map(r => r.counts) };
  }));
  const passed = rows.length === 144 && rows.every(r => r.weightsUnchanged) &&
    conditions.every(x => x.n === 12 && x.vpn === 12 && x.pooled >= 10 &&
      x.decoded >= (x.message === null ? 12 : 11) &&
      x.pooled >= conditions.find(y => y.cue === x.cue && y.message === null).pooled - 2) &&
    rows.every(r => r.condition === null ? r.counts.every(x => x === 0) :
      codes[r.condition].every((bit, i) => bit ? r.counts[i] > 0 : r.counts[i] === 0));
  return { seed, passed, conditions, rows };
}

class PreferenceStudent extends ChallengeCueHeadBandit {
  imitate(decodedAction, enabled = true) {
    if (!enabled || this.activeHead === 3 || decodedAction === null) return 0;
    if (![0, 1, 2].includes(decodedAction)) throw Error('Invalid decoded teacher action');
    const h = this.heads[this.activeHead], q = this.logits(this.f);
    const mx = Math.max(...q), ex = q.map(x => Math.exp(x - mx));
    const z = ex.reduce((a, b) => a + b, 0);
    for (let a = 0; a < 3; a++) {
      const g = .03 * (Number(a === decodedAction) - ex[a] / z);
      h.b[a] += g;
      for (let j = 0; j < 3; j++) h.W[a][j] += g * this.f[j];
    }
    return 1;
  }
}
function student(seed, frozen = false, source = null) {
  const net = makeNet(seed), agent = new PreferenceStudent(net, seed, featureMean, featureStd);
  if (source) {
    const saved = JSON.parse(studentWeights(source.agent));
    saved.heads.forEach((h, i) => { h.W.forEach((w, a) => agent.heads[i].W[a].set(w)); agent.heads[i].b.set(h.b); });
  }
  agent.explore = !frozen; agent.readoutLearn = !frozen;
  return { net, agent };
}
function studentRound(s, game, key, target, learn) {
  const cue = game.observe().previousOpponent, vpn = cue === null ? [] : Array.from(s.net.laneVpn[cue].slice(2));
  const net = s.net, agent = s.agent, dnCounts = [0, 0];
  net.resetState(); net.rand = rng(key + 2); agent.rand = rng(key + 501);
  let row;
  for (let t = 0; t < 680; t++) {
    if (t === 240 && target !== null) DN.forEach((id, i) => {
      if (codes[target][i]) net.setDrive([id], 300);
    });
    if (t === 280) { net.setDrive(DN, 0); if (vpn.length) net.setDrive(vpn, 150); }
    if (t === 330) net.resetCounts();
    if (t === 520) {
      if (vpn.length) net.setDrive(vpn, 0);
      const action = agent.decide(net), decoded = decode(dnCounts), played = game.play(action);
      agent.reward(net, played.reward);
      row = { cue, action, opponent: played.opponent, reward: played.reward,
        rates: [...agent.rates], scores: [...agent.scores], routedCue: agent.activeHead,
        target, decoded, dnCounts: [...dnCounts], rewardStep: 0, imitation: 0 };
      net.resetCounts();
    }
    if (t === 620) {
      row.rewardStep = learn ? agent.alpha * (row.reward - row.scores[row.action]) : 0;
      agent.afterDopamine(net, learn);
      row.imitation = agent.imitate(row.decoded < 0 ? null : row.decoded, learn);
    }
    net.step();
    if (t >= 240 && t < 280) for (let i = 0; i < net.nSpikes; i++) {
      if (net.spikes[i] === DN[0]) dnCounts[0]++;
      else if (net.spikes[i] === DN[1]) dnCounts[1]++;
    }
  }
  if (row.routedCue !== (cue === null ? 3 : cue) || !net.m.every(x => x === 1) ||
      (target === null ? row.decoded !== -1 : row.decoded !== target) ||
      row.reward !== result(row.action, row.opponent)) throw Error('Student round validity failed');
  return row;
}
function frozen(seed, source) {
  const s = student(seed, true, source), rows = [], original = studentWeights(s.agent);
  for (let episode = 0; episode < 3; episode++) {
    const game = opponent(51000 + episode);
    for (let r = 0; r < 24; r++) rows.push(studentRound(s, game,
      seed * 10000000 + (100 + episode) * 1000 + r, null, false));
  }
  if (studentWeights(s.agent) !== original || rows.some(r => r.dnCounts.some(Boolean) || r.imitation))
    throw Error('Frozen evaluation changed weights or used DN messages');
  return { wins: rows.filter(r => r.reward === 1).length,
    byCue: [0, 1, 2].map(cue => {
      const x = rows.filter(r => r.cue === cue);
      return { cue, wins: x.filter(r => r.reward === 1).length, n: x.length,
        actions: [0, 1, 2].map(a => x.filter(r => r.action === a).length) };
    }), actions: [0, 1, 2].map(a => rows.filter(r => r.action === a).length),
    disconnected: true, weightsUnchanged: true };
}
function teacherTape(trained, untrained = false) {
  const source = teacherCopy(trained, untrained), rows = [];
  const initial = teacherSnapshot(source.agent);
  for (let episode = 0; episode < 12; episode++) {
    const game = opponent(41000 + episode);
    for (let r = 0; r < 24; r++) {
      const cue = game.observe().previousOpponent;
      const vpn = cue === null ? [] : Array.from(source.net.laneVpn[cue].slice(2));
      const key = teacherSeed * 10000000 + episode * 1000 + r;
      source.net.rand = rng(key + 1);
      if (vpn.length) source.net.setDrive(vpn, 150);
      let preference = null, scores = null, rates = null;
      for (let t = 0; t < 680; t++) {
        if (t === 50) source.net.resetCounts();
        if (t === 240) {
          if (vpn.length) source.net.setDrive(vpn, 0);
          source.agent.decide(source.net);
          scores = [...source.agent.scores]; rates = [...source.agent.rates];
          if (cue !== null) preference = scores.indexOf(Math.max(...scores));
        }
        source.net.step();
      }
      const played = game.play(0);
      rows.push({ cue, preference, scores, rates, opponent: played.opponent,
        winning: cue === null ? null : win(preference, played.opponent) });
    }
  }
  const byCue = [0, 1, 2].map(cue => {
    const x = rows.filter(r => r.cue === cue);
    return { cue, n: x.length, correct: x.filter(r => r.winning).length,
      preferences: [0, 1, 2].map(a => x.filter(r => r.preference === a).length) };
  });
  return { rows, byCue, frozenWeights: initial === teacherSnapshot(source.agent) &&
    source.net.m.every(x => x === 1), passed: byCue.every(x => x.correct / x.n >= .70) };
}
function shuffled(tape) {
  const observed = tape.rows.map((row, index) => row.cue === null ? null : index).filter(x => x !== null);
  const values = observed.map(i => tape.rows[i].preference), random = rng(2026110602);
  for (let i = values.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1)); [values[i], values[j]] = [values[j], values[i]];
  }
  const target = Array(tape.rows.length).fill(null);
  observed.forEach((index, i) => { target[index] = values[i]; });
  return { target, moved: observed.filter((index, i) => target[index] !== tape.rows[index].preference).length };
}
function trainArm(seed, name, targets = null) {
  const s = student(seed), initial = studentWeights(s.agent), points = [];
  const training = [], initialFrozen = frozen(seed, s);
  points.push({ rounds: 0, ...initialFrozen });
  for (let episode = 0; episode < 12; episode++) {
    const game = opponent(41000 + episode);
    for (let r = 0; r < 24; r++) {
      const index = episode * 24 + r;
      const row = studentRound(s, game, seed * 10000000 + episode * 1000 + r,
        targets ? targets[index] : null, true);
      training.push(row);
      if (checkpoints.includes(index + 1)) points.push({ rounds: index + 1, ...frozen(seed, s) });
    }
  }
  return { name, seed, initial, finalWeights: studentWeights(s.agent),
    training, checkpoints: points, frozenSynapses: s.net.m.every(x => x === 1) };
}
const trained = trainTeacher();
artifact.teacher = trained.result;
if (!trained.result.archivedResultReproduced || !trained.result.synapsesFrozen) {
  await finish('stopped_teacher'); process.exit(0);
}
artifact.receiver = { batches: [], summary: null };
for (const seed of seeds) {
  const batch = receiverBatch(seed);
  artifact.receiver.batches.push(batch);
  if (!batch.passed) {
    artifact.receiver.summary = { passed: false, stoppedAt: seed };
    await finish('stopped_receiver'); process.exit(0);
  }
}
artifact.receiver.summary = { passed: true, seeds };
const example = artifact.receiver.batches[0].rows.find(r => r.condition === 0 && r.counts[0] > 0);
const synthetic = student(seeds[0]), a = synthetic.agent;
a.activeHead = 0; a.W = a.heads[0].W; a.b = a.heads[0].b; a.f = [1, 0, 0];
const decodedExample = decode(example.counts), before = a.logits(a.f);
const unchanged = studentWeights(a); a.imitate(decodedExample, false);
assert.equal(studentWeights(a), unchanged);
a.imitate(decodedExample, true);
const after = a.logits(a.f);
artifact.synthetic = { decodedExample, before, after, disabledUnchanged: true,
  passed: after[decodedExample] - Math.max(...after.filter((_, i) => i !== decodedExample)) >
    before[decodedExample] - Math.max(...before.filter((_, i) => i !== decodedExample)) };
if (!artifact.synthetic.passed) { await finish('stopped_synthetic'); process.exit(0); }
artifact.independent = { arms: seeds.slice(1).map(seed => trainArm(seed, 'alone')), summary: null };
artifact.independent.summary = { passed: artifact.independent.arms.every(arm =>
  arm.checkpoints[0].wins <= 50 && arm.checkpoints.at(-1).wins >= 58 &&
  arm.checkpoints.at(-1).byCue.every(x => x.wins / x.n >= .70) && arm.frozenSynapses) };
if (!artifact.independent.summary.passed) { await finish('stopped_independent'); process.exit(0); }
const correct = teacherTape(trained.agent), untrained = teacherTape(trained.agent, true);
const shuffle = shuffled(correct);
artifact.source = { trainedByCue: correct.byCue, untrainedByCue: untrained.byCue,
  shuffledMoved: shuffle.moved, frozenWeights: correct.frozenWeights && untrained.frozenWeights,
  passed: correct.passed && correct.frozenWeights && untrained.frozenWeights && shuffle.moved > 0 };
if (!artifact.source.passed) { await finish('stopped_source'); process.exit(0); }
const generic = correct.rows.map((row, i) => row.cue === null ? null : i % 3);
const target = { trained: correct.rows.map(r => r.preference),
  untrained: untrained.rows.map(r => r.preference), shuffled: shuffle.target, generic };
const pilot = seeds.slice(1).map((seed, i) => {
  const byArm = { alone: artifact.independent.arms[i] };
  for (const name of arms.filter(x => x !== 'alone')) byArm[name] = trainArm(seed, name, target[name]);
  const first58 = Object.fromEntries(arms.map(name => [name,
    byArm[name].checkpoints.find(x => x.wins >= 58)?.rounds ?? null]));
  const early = Object.fromEntries(arms.map(name => [name,
    [6, 12, 24, 48].reduce((sum, n) => sum + byArm[name].checkpoints.find(x => x.rounds === n).wins, 0) / 4]));
  const controls = arms.filter(x => x !== 'trained');
  const passed = controls.every(name => early.trained - early[name] >= 4 &&
      (first58[name] === null || first58.trained < first58[name])) &&
    first58.trained !== null && first58.trained <= 48 &&
    byArm.trained.checkpoints.at(-1).wins >= 58 &&
    byArm.trained.checkpoints.at(-1).byCue.every(x => x.wins / x.n >= .70) &&
    Object.values(byArm).every(x => x.frozenSynapses && x.checkpoints.length === 7 &&
      x.training.every(row => row.target === null ? row.decoded === -1 : row.target === row.decoded));
  return { seed, byArm, first58, early, passed };
});
artifact.pilot = { pairs: pilot, summary: { passed: pilot.every(x => x.passed),
  seeds: pilot.map(x => ({ seed: x.seed, passed: x.passed, first58: x.first58, early: x.early,
    final: Object.fromEntries(arms.map(name => [name, x.byArm[name].checkpoints.at(-1).wins])) })) } };
await finish(artifact.pilot.summary.passed ? 'pilot_passed_unconfirmed' : 'pilot_failed');
