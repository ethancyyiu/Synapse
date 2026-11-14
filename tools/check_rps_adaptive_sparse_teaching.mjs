// Locked student-requested 25% communication study of the unchanged Strategy B preference protocol.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, open, access } from 'node:fs/promises';
import { Network, rng, wiring } from '../js/sim.js';
import { Opponent, RpsContextualBandit, RpsEpisode, result } from '../js/rps.js';
import { ChallengeCueHeadBandit, studentWeights } from '../js/rps_challenge_students.js';

const stage = process.argv[2];
if (!['initial', 'confirmation'].includes(stage)) throw Error('Use initial or confirmation');
const assignments = {
  initial: { teacherSeed: 18101, studentSeed: 18102,
    trainingSeeds: Array.from({ length: 12 }, (_, i) => 85000 + i),
    evaluationSeeds: [95000, 95001, 95003] },
  confirmation: { teacherSeed: 18103, studentSeed: 18104,
    trainingSeeds: Array.from({ length: 12 }, (_, i) => 85012 + i),
    evaluationSeeds: [95004, 95006, 95009] },
};
const { teacherSeed, studentSeed, trainingSeeds, evaluationSeeds } = assignments[stage];
const read = async name => JSON.parse(await readFile(new URL('../data/' + name, import.meta.url)));
const output = new URL('../data/rps_adaptive_sparse_teaching_' + stage + '.json', import.meta.url);
try { await access(output); throw Error('Write-once adaptive result already exists'); }
catch (error) { if (error.code !== 'ENOENT') throw error; }
if (stage === 'confirmation') {
  const first = await read('rps_adaptive_sparse_teaching_initial.json');
  assert.equal(first.status, 'passed', 'Initial adaptive stage did not pass');
}
const protocolText = await readFile(new URL('../docs/adaptive-sparse-teaching-protocol.md', import.meta.url));
const protocolSha256 = createHash('sha256').update(protocolText).digest('hex');
const sub = await read('subgraph.json');
const { featureMean, featureStd } = (await read('rps_contextual_bandit.json')).design;
const centroids = (await read('rps_readout_information.json')).centroids;
const DN = [3141, 3165], codes = [[1, 0], [0, 1], [1, 1]];
const arms = ['adaptive_trained', 'fixed_trained', 'adaptive_shuffled',
  'adaptive_untrained', 'adaptive_generic', 'alone'];
const checkpoints = [0, 6, 12, 24, 48, 96, 288];
const protocol = { stage, teacherSeed, studentSeed, trainingSeeds, evaluationSeeds,
  lockedDocument: 'docs/adaptive-sparse-teaching-protocol.md', protocolSha256,
  unchangedFrom: 'rps_action_preference_confirmation_second.json',
  teacher: '12*24 Strategy B own rewards; frozen >=58/72, >=70% per cue, >=12 each action',
  receiver: 'v3 matched-stream post-sensory forced codes, 12 repeats per cue/branch; all gates required',
  code: codes, dn: DN, dnHz: 300, cueWindow: [0, 240], dnWindow: [240, 280],
  rewardTime: 280, updateTime: 620, roundEnd: 680,
  student: 'cue heads, own reward alpha .03, decoded-action softmax imitation alpha .03, epsilon .20',
  training: '12*24 Strategy B, stage-specific unused opponent seeds, keyed streams',
  checkpoints, evaluation: '72 disconnected frozen rounds, stage-specific unused opponent seeds',
  budget: { eligible: 276, downlinkCap: 69, perCuePriorRewardObservations: 23,
    upstreamRequestBitsPerAdaptiveArm: 276, decisionTime: 240 },
  controls: arms,
  sourceGate: 'trained >=70% winning preference per cue; all actions; frozen source weights',
  pilotGate: 'adaptive trained early mean 6/12/24/48 >=4/72 over each of five arms, first 58/72 <=48 earlier every control, final >=58/72 >=70% per cue and >=12 each action' };
const artifact = { protocol, status: 'started', teacher: null, receiver: null,
  synthetic: null, independent: null, source: null, schedule: null, pilot: null };
async function finish(status) {
  artifact.status = status;
  const file = await open(output, 'wx');
  try { await file.writeFile(JSON.stringify(artifact, null, 2) + '\n'); }
  finally { await file.close(); }
  console.log(JSON.stringify({ status, teacherValid: artifact.teacher?.teacherValid,
    receiver: artifact.receiver?.passed, synthetic: artifact.synthetic?.passed,
    independent: artifact.independent?.summary, source: artifact.source?.passed,
    schedule: artifact.schedule?.summary, pilot: artifact.pilot?.summary }));
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

function studentRound(s, game, key, target, learn) {
  const cue = game.observe().previousOpponent;
  const vpn = cue === null ? [] : Array.from(s.net.laneVpn[cue].slice(2));
  const net = s.net, agent = s.agent, dnCounts = [0, 0];
  let selectedTarget = target;
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
        laneCounts: [...agent.laneCounts], target: null, request: null,
        requestTime: null, priorCueCount: null, decoded: null, dnCounts: null,
        opponent: null, reward: null, rewardStep: 0, imitation: 0 };
      if (typeof target === 'function') {
        const choice = target(row);
        if (row.reward !== null || choice.requestTime !== 240)
          throw Error('Advice selection used invalid timing');
        selectedTarget = choice.target;
        row.request = choice.request;
        row.requestTime = choice.requestTime;
        row.priorCueCount = choice.priorCueCount;
      }
      row.target = selectedTarget;
      if (selectedTarget !== null) DN.forEach((id, i) => {
        if (codes[selectedTarget][i]) net.setDrive([id], 300);
      });
    }
    if (t === 280) {
      net.setDrive(DN, 0);
      row.decoded = decode(dnCounts);
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
      (selectedTarget === null ? row.decoded !== -1 : row.decoded !== selectedTarget) ||
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
  if (studentWeights(s.agent) !== original || rows.some(r => r.dnCounts.some(Boolean) ||
      r.imitation || r.request !== null))
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
      dnCounts: r.dnCounts, decoded: r.decoded, request: r.request,
      rewardStep: r.rewardStep,
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
function trainArm(seed, name, targets = null, adaptiveMask = null) {
  const s = student(seed), initial = studentWeights(s.agent), points = [];
  const training = [], initialFrozen = frozen(seed, s), priorCueCounts = [0, 0, 0];
  let sent = 0;
  points.push({ rounds: 0, ...initialFrozen });
  for (let episode = 0; episode < 12; episode++) {
    const game = opponent(trainingSeeds[episode]);
    for (let r = 0; r < 24; r++) {
      const index = episode * 24 + r;
      const selection = adaptiveMask ? decision => {
        const h = decision.routedCue;
        const request = h === 3 ? null : Number(priorCueCounts[h] < 23 && sent < 69);
        if (request !== adaptiveMask[index] || decision.reward !== null)
          throw Error('Adaptive request diverged or saw reward');
        const chosen = request === 1 ? targets[index] : null;
        if (request === 1 && chosen === null) throw Error('Requested advice missing');
        if (request === 1) sent++;
        return { target: chosen, request, requestTime: 240,
          priorCueCount: h === 3 ? null : priorCueCounts[h] };
      } : targets ? targets[index] : null;
      const row = studentRound(s, game, seed * 10000000 + episode * 1000 + r,
        selection, true);
      training.push(row);
      if (row.routedCue !== 3) priorCueCounts[row.routedCue]++;
      if (checkpoints.includes(index + 1)) points.push({ rounds: index + 1, ...frozen(seed, s) });
    }
  }
  return { name, seed, initial, finalWeights: studentWeights(s.agent),
    training, checkpoints: points, sent, priorCueCounts,
    frozenSynapses: s.net.m.every(x => x === 1) };
}


const trained = trainTeacher();
artifact.teacher = trained.result;
if (!trained.result.teacherValid || !trained.result.synapsesFrozen) {
  await finish('stopped_teacher'); process.exit(0);
}
artifact.receiver = receiverBatch(studentSeed);
if (!artifact.receiver.passed) {
  await finish('stopped_receiver'); process.exit(0);
}
const example = artifact.receiver.rows.find(r => r.message === 0 && r.counts[0] > 0);
const synthetic = student(studentSeed), a = synthetic.agent;
a.activeHead = 0; a.W = a.heads[0].W; a.b = a.heads[0].b; a.f = [1, 0, 0];
const decodedExample = decode(example.counts), before = a.logits(a.f);
const unchanged = studentWeights(a); a.imitate(decodedExample, false);
assert.equal(studentWeights(a), unchanged);
a.imitate(null, true);
assert.equal(studentWeights(a), unchanged, 'Skipped advice must not imitate');
a.imitate(decodedExample, true);
const after = a.logits(a.f);
artifact.synthetic = { decodedExample, before, after, disabledUnchanged: true,
  passed: after[decodedExample] - Math.max(...after.filter((_, i) => i !== decodedExample)) >
    before[decodedExample] - Math.max(...before.filter((_, i) => i !== decodedExample)) };
if (!artifact.synthetic.passed) { await finish('stopped_synthetic'); process.exit(0); }
const alone = trainArm(studentSeed, 'alone');
artifact.independent = { arm: alone, summary: { passed:
  alone.checkpoints[0].wins <= 50 && alone.checkpoints.at(-1).wins >= 58 &&
  alone.checkpoints.at(-1).byCue.every(x => x.wins / x.n >= .70) &&
  [0, 1, 2].every(a => alone.checkpoints.at(-1).byCue.reduce((sum, x) =>
    sum + x.actions[a], 0) >= 12) && alone.frozenSynapses } };
if (!artifact.independent.summary.passed) { await finish('stopped_independent'); process.exit(0); }
const correct = teacherTape(trained.agent), untrained = teacherTape(trained.agent, true);
const quality = (target, indices) => [0, 1, 2].map(cue => {
  const selected = indices.filter(i => correct.rows[i].cue === cue);
  return { cue, n: selected.length,
    correct: selected.filter(i => win(target[i], correct.rows[i].opponent)).length,
    actions: [0, 1, 2].map(a => selected.filter(i => target[i] === a).length) };
});
const trainedTarget = correct.rows.map(r => r.preference);
const present = target => [0, 1, 2].every(a => target.includes(a));
artifact.source = { trainedByCue: correct.byCue, frozenWeights: correct.frozenWeights && untrained.frozenWeights,
  actionTotals: [0, 1, 2].map(a => trainedTarget.filter(x => x === a).length),
  passed: correct.passed && present(trainedTarget) && correct.frozenWeights && untrained.frozenWeights };
if (!artifact.source.passed) { await finish('stopped_source'); process.exit(0); }

const observed = correct.rows.flatMap((row, i) => row.cue === null ? [] : [i]);
assert.equal(observed.length, 276);
const stageIndex = stage === 'initial' ? 0 : 1;
const requestMask = Array(288).fill(null), priorCueCounts = [0, 0, 0];
let requested = 0;
for (let index = 0; index < 288; index++) {
  const cue = correct.rows[index].cue;
  if (cue === null) continue;
  requestMask[index] = Number(priorCueCounts[cue] < 23 && requested < 69);
  if (requestMask[index]) requested++;
  priorCueCounts[cue]++;
}
const adaptiveSelected = observed.filter(i => requestMask[i] === 1);
const fixedSelected = observed.filter((_, j) => j % 4 === 0);
assert.ok(requested <= 69 && adaptiveSelected.length === 69 && fixedSelected.length === 69 &&
  priorCueCounts.every(n => n >= 23) &&
  [0, 1, 2].every(cue => adaptiveSelected.filter(i => correct.rows[i].cue === cue).length === 23));
const targets = Object.fromEntries(arms.filter(name => name !== 'alone')
  .map(name => [name, Array(288).fill(null)]));
const shuffledValues = adaptiveSelected.map(i => trainedTarget[i]);
const shuffleRandom = rng(2026110921 + stageIndex), genericRandom = rng(2026110923 + stageIndex);
for (let i = shuffledValues.length - 1; i > 0; i--) {
  const j = Math.floor(shuffleRandom() * (i + 1));
  [shuffledValues[i], shuffledValues[j]] = [shuffledValues[j], shuffledValues[i]];
}
adaptiveSelected.forEach((index, j) => {
  targets.adaptive_trained[index] = trainedTarget[index];
  targets.adaptive_untrained[index] = untrained.rows[index].preference;
  targets.adaptive_shuffled[index] = shuffledValues[j];
  targets.adaptive_generic[index] = Math.floor(3 * genericRandom());
});
fixedSelected.forEach(index => { targets.fixed_trained[index] = trainedTarget[index]; });
const histogram = (target, indices) => [0, 1, 2]
  .map(a => indices.filter(i => target[i] === a).length);
assert.ok(JSON.stringify(histogram(targets.adaptive_trained, adaptiveSelected)) ===
  JSON.stringify(histogram(targets.adaptive_shuffled, adaptiveSelected)) &&
  arms.filter(name => name.startsWith('adaptive_')).every(name =>
    targets[name].every((value, i) => (value !== null) === (requestMask[i] === 1))) &&
  targets.fixed_trained.every((value, i) => (value !== null) === fixedSelected.includes(i)),
  'Invalid matched message schedules');
artifact.schedule = { requestMask, adaptiveSelected, fixedSelected,
  summary: { eligible: observed.length, cap: 69, upstreamBitsPerAdaptiveArm: 276,
    adaptiveMessages: adaptiveSelected.length, fixedMessages: fixedSelected.length,
    adaptiveByCue: [0, 1, 2].map(cue => adaptiveSelected.filter(i => correct.rows[i].cue === cue).length),
    fixedByCue: [0, 1, 2].map(cue => fixedSelected.filter(i => correct.rows[i].cue === cue).length),
    shuffledMoved: adaptiveSelected.filter(i => targets.adaptive_trained[i] !==
      targets.adaptive_shuffled[i]).length,
    quality: Object.fromEntries(Object.entries(targets).map(([name, target]) =>
      [name, quality(target, name === 'fixed_trained' ? fixedSelected : adaptiveSelected)])),
    unusedBudget: 69 - adaptiveSelected.length } };

const byArm = { alone };
for (const name of arms.filter(name => name !== 'alone')) {
  const adaptive = name.startsWith('adaptive_');
  byArm[name] = trainArm(studentSeed, name, targets[name], adaptive ? requestMask : null);
  const arm = byArm[name], expected = 69;
  const requests = arm.training.filter(r => r.request !== null);
  if (arm.sent !== (adaptive ? expected : 0) ||
      requests.length !== (adaptive ? 276 : 0) ||
      arm.training.filter(r => r.target !== null).length !== expected ||
      arm.training.filter(r => r.imitation === 1).length !== expected ||
      arm.training.some((row, i) => row.target === null ?
        row.decoded !== -1 || row.imitation !== 0 || row.dnCounts.some(Boolean) :
        row.target !== row.decoded || row.imitation !== 1) ||
      (adaptive && arm.training.some((row, i) => row.request !== requestMask[i] ||
        (row.request !== null && row.requestTime !== 240))) ||
      !arm.frozenSynapses || arm.checkpoints.length !== 7 ||
      arm.checkpoints.some(point => !point.disconnected || !point.weightsUnchanged ||
        point.rows.some(row => row.decoded !== -1 || row.imitation !== 0 ||
          row.request !== null || row.dnCounts.some(Boolean)))) {
    artifact.pilot = { byArm, invalidArm: name };
    await finish('stopped_invalid_arm'); process.exit(0);
  }
}
const starting = byArm.alone.initial;
const schedule = JSON.stringify(byArm.alone.training.map(r => [r.cue, r.opponent]));
assert.ok(Object.values(byArm).every(arm => arm.initial === starting &&
  JSON.stringify(arm.training.map(r => [r.cue, r.opponent])) === schedule),
  'Student starts or opponent streams differ');
assert.ok(arms.filter(name => name.startsWith('adaptive_')).every(name =>
  JSON.stringify(byArm[name].training.map(r => r.request)) ===
  JSON.stringify(byArm.adaptive_trained.training.map(r => r.request))),
  'Adaptive controls diverged in request opportunities');
const first58 = Object.fromEntries(arms.map(name => [name,
  byArm[name].checkpoints.find(x => x.wins >= 58)?.rounds ?? null]));
const early = Object.fromEntries(arms.map(name => [name,
  [6, 12, 24, 48].reduce((sum, n) =>
    sum + byArm[name].checkpoints.find(x => x.rounds === n).wins, 0) / 4]));
const rivals = arms.filter(name => name !== 'adaptive_trained');
const gains = Object.fromEntries(rivals.map(name => [name,
  early.adaptive_trained - early[name]]));
const earlyAdvantage = rivals.every(name => gains[name] >= 4);
const proficiency = first58.adaptive_trained !== null &&
  first58.adaptive_trained <= 48 && rivals.every(name =>
    first58[name] === null || first58.adaptive_trained < first58[name]);
const final = byArm.adaptive_trained.checkpoints.at(-1);
const finalReliability = final.wins >= 58 && final.byCue.every(x => x.wins / x.n >= .70) &&
  [0, 1, 2].every(a => final.byCue.reduce((sum, x) => sum + x.actions[a], 0) >= 12);
const gates = { earlyAdvantage, proficiency, finalReliability,
  passed: earlyAdvantage && proficiency && finalReliability,
  earlyGains: gains, first58: first58.adaptive_trained, finalWins: final.wins,
  finalByCue: final.byCue, finalActions: final.actions };
artifact.pilot = { byArm, first58, early, gates,
  summary: { seed: studentSeed, gates, first58, early,
    cumulativeMessages: Object.fromEntries(arms.map(name => [name,
      checkpoints.map(rounds => ({ rounds, sent: byArm[name].training
        .slice(0, rounds).filter(r => r.target !== null).length }))])),
    final: Object.fromEntries(arms.map(name => [name, byArm[name].checkpoints.at(-1).wins])) } };
await finish(gates.passed ? 'passed' : 'failed');

