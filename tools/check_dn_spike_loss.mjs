// One locked, opt-in measurement-buffer spike-loss pilot. Historical runners are untouched.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, open, access } from 'node:fs/promises';
import { Network, rng, wiring } from '../js/sim.js';
import { Opponent, result } from '../js/rps.js';
import { ChallengeCueHeadBandit, studentWeights } from '../js/rps_challenge_students.js';

const root = new URL('../', import.meta.url);
const read = async path => JSON.parse(await readFile(new URL(path, root)));
const protocolSha256 = createHash('sha256').update(await readFile(
  new URL('docs/dn-spike-loss-protocol.md', root))).digest('hex');
const output = new URL('data/dn_spike_loss.json', root);
try { await access(output); throw Error('Write-once result already exists'); }
catch (e) { if (e.code !== 'ENOENT') throw e; }
const sub = await read('data/subgraph.json');
const { featureMean, featureStd } = (await read('data/rps_contextual_bandit.json')).design;
const prior = await read('data/rps_connectome_contribution_ablation.json');
assert.equal(prior.status, 'complete');
const DN = [3141, 3165], codes = [[1, 0], [0, 1], [1, 1]];
const checkpoints = [0, 6, 12, 24, 48, 96, 288], losses = [0, .1, .3];
const assignments = [
  { studentSeed: 23001, priorIndex: 0 },
  { studentSeed: 23002, priorIndex: 1 },
];
const netFor = seed => new Network(sub, wiring(sub, 'real', seed), { seed });
const opponent = seed => new Opponent({ strategy: 'previous_opponent', offset: 2, seed });
const decoded = counts => codes.findIndex(code => code.every((bit, i) => bit === Number(counts[i] > 0)));
function thinSpikes(spikes, n, probability, random, raw, kept) {
  for (let i = 0; i < n; i++) {
    const lane = spikes[i] === DN[0] ? 0 : spikes[i] === DN[1] ? 1 : -1;
    if (lane < 0) continue;
    raw[lane]++;
    if (probability === 0 || random() >= probability) kept[lane]++;
  }
}
class Student extends ChallengeCueHeadBandit {
  imitate(action) {
    if (this.activeHead === 3 || action === null) return 0;
    assert.ok(action >= 0 && action <= 2);
    const h = this.heads[this.activeHead], q = this.logits(this.f);
    const m = Math.max(...q), e = q.map(x => Math.exp(x - m)), z = e.reduce((a, b) => a + b, 0);
    for (let a = 0; a < 3; a++) {
      const g = .03 * (Number(a === action) - e[a] / z);
      h.b[a] += g;
      for (let j = 0; j < 3; j++) h.W[a][j] += g * this.f[j];
    }
    return 1;
  }
}
function student(seed, frozen = false) {
  const net = netFor(seed), agent = new Student(net, seed, featureMean, featureStd);
  agent.explore = !frozen;
  agent.readoutLearn = !frozen;
  return { net, agent };
}
function checkThinning() {
  const net = netFor(23001), raw0 = [0, 0], keep0 = [0, 0], raw1 = [0, 0], keep1 = [0, 0];
  net.resetState(); net.rand = rng(2300119001); net.setDrive([DN[0]], 300);
  for (let t = 0; t < 40; t++) {
    net.step();
    thinSpikes(net.spikes, net.nSpikes, 0, rng(t + 1), raw0, keep0);
    thinSpikes(net.spikes, net.nSpikes, 1, rng(t + 1), raw1, keep1);
  }
  assert.ok(raw0[0] > 0 && raw0[1] === 0);
  assert.deepEqual(raw0, raw1);
  assert.deepEqual(keep0, raw0);
  assert.deepEqual(keep1, [0, 0]);
  assert.equal(decoded(keep0), 0);
  assert.equal(decoded(keep1), -1);
  assert.ok(net.m.every(v => v === 1));
  const archived = prior.pairs[0].byArm.neural.checkpoints.at(-1);
  const weights = JSON.parse(prior.pairs[0].byArm.neural.finalWeights);
  for (const row of archived.rows) {
    const h = weights.heads[row.cue === null ? 3 : row.cue];
    const q = h.W.map((w, a) => w.reduce((sum, x, j) => sum + x * row.f[j], h.b[a]));
    assert.deepEqual(q, row.scores);
    assert.equal(q.indexOf(Math.max(...q)), row.action);
    assert.equal(result(row.action, row.opponent), row.reward);
  }
  return { measuredDnSpikes: raw0, zeroLossDecoded: 0, fullDiagnosticLossDecoded: null,
    archivedFrozenRowsReproduced: archived.rows.length, coreWeightsUnchanged: true };
}
function round(s, game, key, target, loss, delivery, direct = null, learn = true) {
  const cue = game.observe().previousOpponent, net = s.net, agent = s.agent;
  const vpn = cue === null ? [] : [...net.laneVpn[cue].slice(2)];
  const raw = [0, 0], kept = [0, 0], random = rng(key + 19001);
  net.resetState(); net.rand = rng(key + 2); agent.rand = rng(key + 501);
  if (vpn.length) net.setDrive(vpn, 150);
  let row;
  for (let t = 0; t < 680; t++) {
    if (t === 50) net.resetCounts();
    if (t === 240) {
      if (vpn.length) net.setDrive(vpn, 0);
      const action = agent.decide(net);
      row = { cue, action, scores: [...agent.scores], f: [...agent.f],
        routedCue: agent.activeHead, target, decoded: null, raw: null, kept: null,
        opponent: null, reward: null, imitation: 0 };
      if (delivery === 'neural' && target !== null) DN.forEach((id, j) => {
        if (codes[target][j]) net.setDrive([id], 300);
      });
    }
    if (t === 280) {
      net.setDrive(DN, 0);
      const code = delivery === 'digital' ? direct : decoded(kept);
      row.decoded = code < 0 ? null : code;
      row.raw = [...raw]; row.kept = [...kept];
      const played = game.play(row.action);
      row.opponent = played.opponent; row.reward = played.reward;
      agent.reward(net, played.reward); net.resetCounts();
    }
    if (t === 620) {
      agent.afterDopamine(net, learn);
      row.imitation = learn ? agent.imitate(row.decoded) : 0;
    }
    net.step();
    if (t >= 240 && t < 280)
      thinSpikes(net.spikes, net.nSpikes, loss, random, raw, kept);
  }
  assert.equal(row.routedCue, cue === null ? 3 : cue);
  assert.equal(row.reward, result(row.action, row.opponent));
  assert.ok(net.m.every(v => v === 1));
  if (delivery !== 'neural' || target === null) assert.deepEqual(raw, [0, 0]);
  if (delivery === 'neural' && target !== null && loss === 0) assert.equal(row.decoded, target);
  return row;
}
function featureTape(seed, evaluationSeeds) {
  const s = student(seed, true), before = studentWeights(s.agent), rows = [];
  for (let e = 0; e < 3; e++) {
    const game = opponent(evaluationSeeds[e]);
    for (let r = 0; r < 24; r++) {
      const row = round(s, game, seed * 10000000 + (100 + e) * 1000 + r,
        null, 0, 'silent', null, false);
      rows.push({ episode: e, round: r, cue: row.cue, routedCue: row.routedCue,
        f: row.f, opponent: row.opponent });
    }
  }
  assert.equal(studentWeights(s.agent), before);
  assert.equal(rows.length, 72);
  return rows;
}
function frozen(weightsText, tape, rounds) {
  const weights = JSON.parse(weightsText);
  const rows = tape.map(t => {
    const h = weights.heads[t.routedCue];
    const scores = h.W.map((w, a) => w.reduce((sum, x, j) => sum + x * t.f[j], h.b[a]));
    const action = scores.indexOf(Math.max(...scores));
    return { episode: t.episode, round: t.round, cue: t.cue, action,
      opponent: t.opponent, reward: result(action, t.opponent) };
  });
  return { rounds, wins: rows.filter(r => r.reward === 1).length,
    byCue: [0, 1, 2].map(cue => {
      const x = rows.filter(r => r.cue === cue);
      return { cue, wins: x.filter(r => r.reward === 1).length, n: x.length };
    }), actions: [0, 1, 2].map(a => rows.filter(r =>
      r.cue !== null && r.action === a).length),
    disconnected: true, learning: false, rows };
}
function train(seed, trainingSeeds, evaluationTape, targets, delivery, loss, directTape = null) {
  const s = student(seed), initial = studentWeights(s.agent), points = [];
  const training = [];
  points.push(frozen(initial, evaluationTape, 0));
  for (let e = 0; e < 12; e++) {
    const game = opponent(trainingSeeds[e]);
    for (let r = 0; r < 24; r++) {
      const i = e * 24 + r, key = seed * 10000000 + e * 1000 + r;
      const target = targets?.[i] ?? null;
      const x = round(s, game, key, target, loss, delivery, directTape?.[i] ?? null);
      training.push({ episode: e, round: r, cue: x.cue, action: x.action,
        opponent: x.opponent, reward: x.reward, target, decoded: x.decoded,
        raw: x.raw, kept: x.kept, imitation: x.imitation });
      if (checkpoints.includes(i + 1))
        points.push(frozen(studentWeights(s.agent), evaluationTape, i + 1));
    }
  }
  assert.equal(points.length, 7);
  return { initial, finalWeights: studentWeights(s.agent), training,
    checkpoints: points, neuralWeightsUnchanged: s.net.m.every(v => v === 1) };
}
const correctness = checkThinning();
const artifact = { protocolSha256, status: 'started', correctness,
  assignments, lossRates: losses, checkpoints, pairs: [] };
for (const { studentSeed, priorIndex } of assignments) {
  const source = prior.pairs[priorIndex];
  assert.ok(source.teacher.passed && source.source.passed && source.receiver.passed);
  const assignment = source.assignment;
  assert.equal(source.source.rows.length, 288);
  const trained = source.source.rows.map(r => r.preference);
  const shuffled = source.source.scrambledTarget;
  assert.equal(shuffled.length, 288);
  const evalTape = featureTape(studentSeed, assignment.evaluationSeeds);
  const pair = { studentSeed, teacherSeed: source.teacherSeed,
    trainingSeeds: assignment.trainingSeeds, evaluationSeeds: assignment.evaluationSeeds,
    sourceArtifactPair: source.name, byLoss: {} };
  artifact.pairs.push(pair);
  const silent = train(studentSeed, assignment.trainingSeeds, evalTape, null, 'silent', 0);
  for (const loss of losses) {
    const neural = train(studentSeed, assignment.trainingSeeds, evalTape, trained, 'neural', loss);
    const digitalTape = neural.training.map(r => r.decoded);
    const digital = train(studentSeed, assignment.trainingSeeds, evalTape,
      trained, 'digital', loss, digitalTape);
    const scrambled = train(studentSeed, assignment.trainingSeeds, evalTape,
      shuffled, 'neural', loss);
    const arms = { neural, digital, scrambled, silent };
    const base = neural.initial;
    assert.ok(Object.values(arms).every(a => a.initial === base && a.neuralWeightsUnchanged));
    const schedule = JSON.stringify(neural.training.map(r => [r.cue, r.opponent]));
    assert.ok(Object.values(arms).every(a =>
      JSON.stringify(a.training.map(r => [r.cue, r.opponent])) === schedule));
    assert.ok(digital.training.every((r, i) => r.decoded === neural.training[i].decoded &&
      r.action === neural.training[i].action && r.reward === neural.training[i].reward &&
      r.imitation === neural.training[i].imitation && r.raw.every(x => x === 0)));
    assert.equal(digital.finalWeights, neural.finalWeights);
    const summary = Object.fromEntries(Object.entries(arms).map(([name, a]) => {
      const observed = a.training.filter(r => r.target !== null);
      const early = [6, 12, 24, 48].reduce((v, round) =>
        v + a.checkpoints.find(x => x.rounds === round).wins, 0) / 4;
      const last = a.checkpoints.at(-1);
      return [name, { early, curve: a.checkpoints.map(x => x.wins), final: last.wins,
        finalByCue: last.byCue, finalActions: last.actions,
        sent: observed.length, delivered: observed.filter(r => r.decoded !== null).length,
        exact: observed.filter(r => r.decoded === r.target).length,
        wrong: observed.filter(r => r.decoded !== null && r.decoded !== r.target).length,
        erased: observed.filter(r => r.decoded === null).length,
        rawSpikes: observed.reduce((v, r) => v + r.raw[0] + r.raw[1], 0),
        retainedSpikes: observed.reduce((v, r) => v + r.kept[0] + r.kept[1], 0),
        imitationUpdates: a.training.reduce((v, r) => v + r.imitation, 0) }];
    }));
    const n = summary.neural, sh = summary.scrambled, si = summary.silent, d = summary.digital;
    const survives = n.early - sh.early >= 4 && n.early - si.early >= 4 &&
      n.final >= 58 && n.finalByCue.every(x => x.wins / x.n >= .7) &&
      n.finalActions.every(x => x >= 12);
    pair.byLoss[String(loss)] = { arms, summary,
      gates: { neuralSurvives: survives, neuralOverDigital: n.early - d.early >= 4,
        matchedDigital: n.early === d.early && n.final === d.final &&
          neural.finalWeights === digital.finalWeights } };
    console.log(JSON.stringify({ studentSeed, loss, summary, gates: pair.byLoss[String(loss)].gates }));
  }
}
artifact.status = 'complete';
artifact.allStudentsSurvive = Object.fromEntries(losses.map(loss =>
  [String(loss), artifact.pairs.every(p => p.byLoss[String(loss)].gates.neuralSurvives)]));
artifact.neuralBeatsDigital = artifact.pairs.every(p => losses.every(loss =>
  p.byLoss[String(loss)].gates.neuralOverDigital));
const file = await open(output, 'wx');
try { await file.writeFile(JSON.stringify(artifact) + '\n'); }
finally { await file.close(); }
console.log(JSON.stringify({ status: artifact.status, protocolSha256,
  allStudentsSurvive: artifact.allStudentsSurvive,
  neuralBeatsDigital: artifact.neuralBeatsDigital }));
