// Locked causal comparison of measured DN spikes and matched digital preference delivery.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, open, access } from 'node:fs/promises';
import { Network, rng, wiring } from '../js/sim.js';
import { Opponent, RpsContextualBandit, RpsEpisode, result } from '../js/rps.js';
import { ChallengeCueHeadBandit, studentWeights } from '../js/rps_challenge_students.js';

const assignments = {
  pair1: { teacherSeed: 11013, studentSeed: 11017,
    archive: 'rps_action_preference_strategy_b_initial.json',
    trainingSeeds: Array.from({ length: 12 }, (_, i) => 82000 + i),
    evaluationSeeds: [92000, 92001, 92003] },
  pair2: { teacherSeed: 11014, studentSeed: 11018,
    archive: 'rps_action_preference_strategy_b_confirmation.json',
    trainingSeeds: Array.from({ length: 12 }, (_, i) => 82012 + i),
    evaluationSeeds: [92004, 92006, 92009] },
};
let teacherSeed, studentSeed, trainingSeeds, evaluationSeeds;
const read = async name => JSON.parse(await readFile(new URL('../data/' + name, import.meta.url)));
const output = new URL('../data/rps_connectome_contribution_ablation.json', import.meta.url);
try { await access(output); throw Error('Write-once ablation result already exists'); }
catch (error) { if (error.code !== 'ENOENT') throw error; }
const protocolText = await readFile(new URL('../docs/connectome-contribution-ablation-protocol.md', import.meta.url));
const protocolSha256 = createHash('sha256').update(protocolText).digest('hex');
const sub = await read('subgraph.json');
const { featureMean, featureStd } = (await read('rps_contextual_bandit.json')).design;
const centroids = (await read('rps_readout_information.json')).centroids;
const DN = [3141, 3165], codes = [[1, 0], [0, 1], [1, 1]];
const arms = ['neural', 'digital', 'silent', 'scrambled'];
const checkpoints = [0, 6, 12, 24, 48, 96, 288];
const protocol = { assignments, lockedDocument: 'docs/connectome-contribution-ablation-protocol.md',
  protocolSha256, unchangedFrom: 'rps_action_preference_strategy_b_initial.json',
  teacher: 'Archived independently reward-trained Strategy B controllers 11013/11014',
  receiver: 'v3 matched-stream post-sensory forced codes, 12 repeats per cue/branch; all gates required',
  code: codes, dn: DN, dnHz: 300, cueWindow: [0, 240], dnWindow: [240, 280],
  rewardTime: 280, updateTime: 620, roundEnd: 680,
  student: 'cue heads, own reward alpha .03, decoded-action softmax imitation alpha .03, epsilon .20',
  training: '12*24 Strategy B, new predetermined opponent seeds, keyed streams',
  checkpoints, evaluation: '72 disconnected frozen rounds, new predetermined opponent seeds',
  controls: arms, sourceGate: 'trained >=70% winning preference per cue; scrambled <=60% per cue; all actions; >=50% shuffle changed',
  transmissionGate: '>=99% exact decoded targets and no off-code DN counts',
  similarityBand: 'neural versus digital early mean and final differences each within +/-2 wins/72',
  teacherInformationGate: 'neural early mean >=4 wins/72 above both silent and scrambled for both students' };
const artifact = { protocol, status: 'started', pairs: [] };
async function finish(status) {
  artifact.status = status;
  const file = await open(output, 'wx');
  try { await file.writeFile(JSON.stringify(artifact, null, 2) + '\n'); }
  finally { await file.close(); }
  console.log(JSON.stringify({ status, pairs: artifact.pairs.map(p => ({
    studentSeed: p.studentSeed, stop: p.stop, receiver: p.receiver?.passed,
    source: p.source?.passed, summary: p.summary })) }));
}
function decode(counts) {
  const bits = counts.map(x => Number(x > 0));
  return codes.findIndex(code => code.every((value, i) => value === bits[i]));
}
const win = (a, b) => result(a, b) === 1;
const opponent = seed => new Opponent({ strategy: 'previous_opponent', offset: 2, seed });
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
  const rows = [], trainingRows = [];
  for (let phase = 0; phase < 2; phase++) {
    agent.explore = phase === 0;
    for (let episode = 0; episode < (phase ? 3 : 12); episode++) {
      const ep = new RpsEpisode(net, agent, opponent((phase ? evaluationSeeds : trainingSeeds)[episode]),
        { rounds: 24, learn: phase === 0 });
      while (!ep.done) {
        if (ep.t % 400 === 0) {
          const key = teacherSeed * 10000000 + (phase ? 100 + episode : episode) * 1000 + ep.t / 400;
          net.rand = rng(key + 1); agent.rand = rng(key + 501);
        }
        ep.step();
      }
      if (phase) rows.push(...ep.events.map((r, round) => ({ episode, round,
        cue: r.observation.previousOpponent, action: r.action, opponent: r.opponent,
        reward: r.reward, rates: r.rates })));
      else trainingRows.push(...ep.events.map((r, round) => ({ episode, round,
        cue: r.observation.previousOpponent, action: r.action, opponent: r.opponent,
        reward: r.reward, rates: r.rates })));
    }
  }
  const wins = rows.filter(r => r.reward === 1).length;
  const byCue = [0, 1, 2].map(cue => {
    const x = rows.filter(r => r.cue === cue);
    return { cue, wins: x.filter(r => r.reward === 1).length, n: x.length };
  });
  const actions = [0, 1, 2].map(a => rows.filter(r =>
    r.cue !== null && r.action === a).length);
  return { agent, result: { wins, byCue, actions, trainingRows, frozenRows: rows,
    finalWeights: teacherSnapshot(agent), synapsesFrozen: net.m.every(x => x === 1),
    teacherValid: wins >= 58 && byCue.every(x => x.wins / x.n >= .70) &&
      actions.every(n => n >= 12) } };
}

function receiverProbe(seed, cue, repeat, message) {
  const net = new Network(sub, wiring(sub, 'real', seed), { seed });
  const agent = new ChallengeCueHeadBandit(net, seed, featureMean, featureStd);
  agent.explore = false; agent.readoutLearn = false;
  const initial = studentWeights(agent), key = seed * 10000000 +
    (700 + 12 * cue + repeat) * 1000;
  net.resetState(); net.rand = rng(key + 2); agent.rand = rng(key + 501);
  const counts = [0, 0], vpn = Array.from(net.laneVpn[cue].slice(2));
  net.setDrive(vpn, 150);
  let decision;
  for (let t = 0; t < 680; t++) {
    if (t === 50) net.resetCounts();
    if (t === 240) {
      net.setDrive(vpn, 0);
      agent.decide(net);
      const rates = [...agent.rates];
      const f = [...agent.f];
      const oldF = rates.map(Math.log1p);
      const oldD = centroids.map(c => c.reduce((sum, value, j) => sum + (value - oldF[j]) ** 2, 0));
      decision = { rates, f, laneCounts: [...agent.laneCounts], routedCue: agent.activeHead,
        oldPooledCue: oldD.indexOf(Math.min(...oldD)) };
      if (message !== null) DN.forEach((id, i) => {
        if (codes[message][i]) net.setDrive([id], 300);
      });
    }
    if (t === 280) net.setDrive(DN, 0);
    net.step();
    if (t >= 240 && t < 280) for (let i = 0; i < net.nSpikes; i++) {
      if (net.spikes[i] === DN[0]) counts[0]++;
      else if (net.spikes[i] === DN[1]) counts[1]++;
    }
  }
  return { seed, cue, repeat, message, key, counts, decoded: decode(counts),
    ...decision, weightsUnchanged: studentWeights(agent) === initial && net.m.every(x => x === 1) };
}
function receiverBatch(seed) {
  const rows = [];
  for (let cue = 0; cue < 3; cue++) for (let repeat = 0; repeat < 12; repeat++)
    for (const message of [null, 0, 1, 2]) rows.push(receiverProbe(seed, cue, repeat, message));
  const conditions = [0, 1, 2].flatMap(cue => {
    const none = rows.filter(r => r.cue === cue && r.message === null);
    return [null, 0, 1, 2].map(message => {
      const x = rows.filter(r => r.cue === cue && r.message === message);
      const decoded = x.filter(r => r.decoded === (message ?? -1)).length;
      const routed = x.filter(r => r.routedCue === cue && r.laneCounts[cue] >
        Math.max(...r.laneCounts.filter((_, i) => i !== cue))).length;
      const intended = x.every(r => message === null ? r.counts.every(c => c === 0) :
        codes[message].every((bit, i) => bit ? r.counts[i] > 0 : r.counts[i] === 0));
      const pairedExact = message === null ? 12 : x.filter(r => {
        const baseline = none.find(n => n.repeat === r.repeat);
        return JSON.stringify(r.rates) === JSON.stringify(baseline.rates) &&
          JSON.stringify(r.f) === JSON.stringify(baseline.f) &&
          r.routedCue === baseline.routedCue;
      }).length;
      const passed = x.length === 12 && decoded === 12 && routed === 12 &&
        pairedExact === 12 && intended &&
        x.every(r => r.weightsUnchanged && r.f.every(Number.isFinite));
      return { cue, message, n: x.length, decoded, routed, pairedExact, intended,
        oldPooledCorrect: x.filter(r => r.oldPooledCue === cue).length, passed };
    });
  });
  return { seed, rows, conditions, passed: conditions.every(c => c.passed) };
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

function studentRound(s, game, key, target, learn, delivery = 'neural') {
  const cue = game.observe().previousOpponent;
  const vpn = cue === null ? [] : Array.from(s.net.laneVpn[cue].slice(2));
  const net = s.net, agent = s.agent, dnCounts = [0, 0];
  net.resetState(); net.rand = rng(key + 2); agent.rand = rng(key + 501);
  if (vpn.length) net.setDrive(vpn, 150);
  let row;
  for (let t = 0; t < 680; t++) {
    if (t === 50) net.resetCounts();
    if (t === 240) {
      if (vpn.length) net.setDrive(vpn, 0);
      const action = agent.decide(net);
      row = { cue, action, rates: [...agent.rates], f: [...agent.f],
        scores: [...agent.scores], routedCue: agent.activeHead,
        laneCounts: [...agent.laneCounts], target, delivery, decoded: null, dnCounts: null,
        opponent: null, reward: null, rewardStep: 0, imitation: 0 };
      if (target !== null && delivery === 'neural') DN.forEach((id, i) => {
        if (codes[target][i]) net.setDrive([id], 300);
      });
    }
    if (t === 280) {
      net.setDrive(DN, 0);
      row.decoded = delivery === 'digital' && target !== null ? target : decode(dnCounts);
      row.dnCounts = [...dnCounts];
      const played = game.play(row.action);
      row.opponent = played.opponent; row.reward = played.reward;
      agent.reward(net, played.reward);
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
      (delivery === 'digital' && dnCounts.some(Boolean)) ||
      row.reward !== result(row.action, row.opponent))
    throw Error('Student round validity failed');
  return row;
}

function frozen(seed, source) {
  const s = student(seed, true, source), rows = [], original = studentWeights(s.agent);
  for (let episode = 0; episode < 3; episode++) {
    const game = opponent(evaluationSeeds[episode]);
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
    disconnected: true, weightsUnchanged: true,
    rows: rows.map((r, i) => ({ episode: Math.floor(i / 24), round: i % 24,
      cue: r.cue, routedCue: r.routedCue, action: r.action, opponent: r.opponent,
      reward: r.reward, rates: r.rates, f: r.f, scores: r.scores,
      dnCounts: r.dnCounts, decoded: r.decoded, rewardStep: r.rewardStep,
      imitation: r.imitation })) };
}
function teacherTape(trained, untrained = false) {
  const source = teacherCopy(trained, untrained), rows = [];
  const initial = teacherSnapshot(source.agent);
  for (let episode = 0; episode < 12; episode++) {
    const game = opponent(trainingSeeds[episode]);
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
  const values = observed.map(i => tape.rows[i].preference), random = rng(2026110702 + studentSeed);
  for (let i = values.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1)); [values[i], values[j]] = [values[j], values[i]];
  }
  const target = Array(tape.rows.length).fill(null);
  observed.forEach((index, i) => { target[index] = values[i]; });
  return { target, moved: observed.filter((index, i) => target[index] !== tape.rows[index].preference).length };
}
function trainArm(seed, name, targets = null, delivery = 'neural') {
  const s = student(seed), initial = studentWeights(s.agent), points = [];
  const training = [], initialFrozen = frozen(seed, s);
  points.push({ rounds: 0, ...initialFrozen });
  for (let episode = 0; episode < 12; episode++) {
    const game = opponent(trainingSeeds[episode]);
    for (let r = 0; r < 24; r++) {
      const index = episode * 24 + r;
      const row = studentRound(s, game, seed * 10000000 + episode * 1000 + r,
        targets ? targets[index] : null, true, delivery);
      training.push(row);
      if (checkpoints.includes(index + 1)) points.push({ rounds: index + 1, ...frozen(seed, s) });
    }
  }
  return { name, seed, initial, finalWeights: studentWeights(s.agent),
    training, checkpoints: points, frozenSynapses: s.net.m.every(x => x === 1) };
}


for (const [name, assignment] of Object.entries(assignments)) {
  ({ teacherSeed, studentSeed, trainingSeeds, evaluationSeeds } = assignment);
  const pair = { name, teacherSeed, studentSeed, assignment, stop: null };
  artifact.pairs.push(pair);
  const archived = await read(assignment.archive);
  const t = archived.teacher;
  pair.teacher = { archive: assignment.archive, originalStatus: archived.status,
    originalWins: t.wins, byCue: t.byCue, actions: t.actions,
    synapsesFrozen: t.synapsesFrozen, weights: t.finalWeights,
    passed: archived.status === 'passed' && t.teacherValid && t.synapsesFrozen &&
      t.wins >= 58 && t.byCue.every(x => x.wins / x.n >= .70) && t.actions.every(x => x >= 12) };
  if (!pair.teacher.passed) {
    pair.stop = 'archived_teacher'; await finish('stopped_prerequisite'); process.exit(0);
  }
  const trainedWeights = JSON.parse(t.finalWeights);
  const correct = teacherTape(trainedWeights), shuffle = shuffled(correct);
  const trainedTarget = correct.rows.map(r => r.preference);
  const quality = target => [0, 1, 2].map(cue => {
    const indices = correct.rows.flatMap((row, i) => row.cue === cue ? [i] : []);
    return { cue, n: indices.length,
      correct: indices.filter(i => win(target[i], correct.rows[i].opponent)).length,
      actions: [0, 1, 2].map(a => indices.filter(i => target[i] === a).length) };
  });
  const scrambledQuality = quality(shuffle.target);
  const present = target => [0, 1, 2].every(a => target.includes(a));
  pair.source = { rows: correct.rows, byCue: correct.byCue, scrambledTarget: shuffle.target,
    scrambledQuality, shuffledMoved: shuffle.moved, frozenWeights: correct.frozenWeights,
    passed: correct.passed && correct.frozenWeights && present(trainedTarget) &&
      present(shuffle.target) && shuffle.moved >= 138 &&
      scrambledQuality.every(x => x.correct / x.n <= .60) };
  if (!pair.source.passed) {
    pair.stop = 'source_or_scramble'; await finish('stopped_prerequisite'); process.exit(0);
  }
  pair.receiver = receiverBatch(studentSeed);
  if (!pair.receiver.passed) {
    pair.stop = 'receiver'; await finish('stopped_prerequisite'); process.exit(0);
  }
  const targets = { neural: trainedTarget, digital: trainedTarget,
    silent: null, scrambled: shuffle.target };
  pair.byArm = Object.fromEntries(arms.map(arm => [arm, trainArm(studentSeed, arm,
    targets[arm], arm === 'digital' ? 'digital' : 'neural')]));
  const byArm = pair.byArm, initial = byArm.neural.initial;
  const schedule = JSON.stringify(byArm.neural.training.map(r => [r.cue, r.opponent]));
  assert.ok(arms.every(arm => byArm[arm].initial === initial &&
    JSON.stringify(byArm[arm].training.map(r => [r.cue, r.opponent])) === schedule &&
    byArm[arm].frozenSynapses && byArm[arm].checkpoints.length === 7),
  'Unmatched student starts, opponent schedule, frozen synapses, or checkpoints');
  const transmission = Object.fromEntries(['neural', 'digital', 'scrambled'].map(arm => {
    const observed = byArm[arm].training.filter(r => r.target !== null);
    return [arm, { n: observed.length,
      exact: observed.filter(r => r.decoded === r.target).length,
      countsValid: observed.every(r => arm === 'digital' ?
        r.dnCounts.every(x => x === 0) :
        codes[r.target].every((bit, i) => bit ? r.dnCounts[i] > 0 : r.dnCounts[i] === 0)) }];
  }));
  const early = Object.fromEntries(arms.map(arm => [arm,
    [6, 12, 24, 48].reduce((sum, n) => sum +
      byArm[arm].checkpoints.find(x => x.rounds === n).wins, 0) / 4]));
  const final = Object.fromEntries(arms.map(arm => [arm,
    byArm[arm].checkpoints.at(-1).wins]));
  const sameTrain = byArm.neural.training.every((r, i) => {
    const d = byArm.digital.training[i];
    return r.action === d.action && r.opponent === d.opponent && r.reward === d.reward &&
      r.rewardStep === d.rewardStep && r.imitation === d.imitation &&
      JSON.stringify(r.f) === JSON.stringify(d.f) && JSON.stringify(r.scores) === JSON.stringify(d.scores);
  });
  const sameFrozen = byArm.neural.checkpoints.every((point, i) =>
    JSON.stringify(point.rows.map(r => [r.action, r.reward, r.scores])) ===
    JSON.stringify(byArm.digital.checkpoints[i].rows.map(r => [r.action, r.reward, r.scores])));
  pair.summary = { transmission, early, final,
    neuralMinusDigitalEarly: early.neural - early.digital,
    neuralMinusDigitalFinal: final.neural - final.digital,
    neuralMinusSilentEarly: early.neural - early.silent,
    neuralMinusScrambledEarly: early.neural - early.scrambled,
    sameTrain, sameFrozen,
    sameFinalWeights: byArm.neural.finalWeights === byArm.digital.finalWeights,
    similarityBand: Math.abs(early.neural - early.digital) <= 2 &&
      Math.abs(final.neural - final.digital) <= 2,
    teacherInformationEffect: early.neural - early.silent >= 4 &&
      early.neural - early.scrambled >= 4,
    transmissionPassed: ['neural', 'digital', 'scrambled'].every(arm =>
      transmission[arm].n === 276 && transmission[arm].exact / 276 >= .99 &&
      transmission[arm].countsValid) };
  if (!pair.summary.transmissionPassed) {
    pair.stop = 'transmission'; await finish('failed_transmission'); process.exit(0);
  }
}
await finish('complete');

