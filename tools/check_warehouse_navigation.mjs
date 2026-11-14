// Isolated toy warehouse route; see the prospectively locked protocol.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { access, open, readFile } from 'node:fs/promises';
import { Network, rng, wiring } from '../js/sim.js';
import { RpsContextualBandit } from '../js/rps.js';
import { ChallengeCueHeadBandit, studentWeights } from '../js/rps_challenge_students.js';

const stage = process.argv[2];
if (!['initial', 'confirmation'].includes(stage)) throw Error('Use initial or confirmation');
const root = new URL('../', import.meta.url);
const read = async path => JSON.parse(await readFile(new URL(path, root)));
const output = new URL(`data/warehouse_navigation_${stage}.json`, root);
try { await access(output); throw Error('Write-once warehouse result already exists'); }
catch (error) { if (error.code !== 'ENOENT') throw error; }
if (stage === 'confirmation' && (await read('data/warehouse_navigation_initial.json')).status !== 'passed')
  throw Error('Initial warehouse study did not pass');
const protocolSha256 = createHash('sha256').update(await readFile(new URL('docs/warehouse-navigation-protocol.md', root))).digest('hex');
const sub = await read('data/subgraph.json');
const { featureMean, featureStd } = (await read('data/rps_contextual_bandit.json')).design;
const conf = stage === 'initial' ? { teacherSeed: 13001, studentSeed: 13002,
  trainScheduleSeed: 2026110801, evalScheduleSeed: 2026110802, genericSeed: 2026110805 } :
  { teacherSeed: 13003, studentSeed: 13004,
    trainScheduleSeed: 2026110803, evalScheduleSeed: 2026110804, genericSeed: 2026110806 };
const DN = [3141, 3165], codes = [[1, 0], [0, 1], [1, 1]];
const cueForState = [2, 0, 1], optimalByCue = [1, 2, 0]; // analysis and gates only
const actions = [[1, 0], [0, 1], [-1, 0]]; // east, north, west
const cells = [[1, 1], [2, 1], [2, 2], [1, 2]]; // S, J, K, G
const checkpoints = [0, 3, 6, 12, 24, 36];
const arms = ['trained', 'alone', 'untrained', 'shuffled', 'generic'];
const artifact = { stage, protocolSha256, config: conf, status: 'started',
  environment: null, sensory: null, teacher: null, source: null,
  receiver: null, synthetic: null, independent: null, comparison: null };
async function finish(status) {
  artifact.status = status;
  const file = await open(output, 'wx');
  try { await file.writeFile(JSON.stringify(artifact, null, 2) + '\n'); }
  finally { await file.close(); }
  console.log(JSON.stringify({ status, environment: artifact.environment?.passed,
    sensory: artifact.sensory?.summary, teacher: artifact.teacher?.summary,
    source: artifact.source?.summary, receiver: artifact.receiver?.summary,
    independent: artifact.independent?.summary, comparison: artifact.comparison?.summary }));
}
const makeNet = seed => new Network(sub, wiring(sub, 'real', seed), { seed });
const teacherWeights = a => JSON.stringify({ W: a.W.map(w => [...w]), b: [...a.b], baseline: a.baseline });
function teacher(frozen = false, source = null) {
  const net = makeNet(conf.teacherSeed);
  const agent = new RpsContextualBandit(net, { seed: conf.teacherSeed, featureMean, featureStd,
    readoutLearn: !frozen });
  if (source) {
    source.agent.W.forEach((w, i) => agent.W[i].set(w));
    agent.b.set(source.agent.b); agent.baseline = source.agent.baseline;
  }
  agent.explore = !frozen;
  return { net, agent };
}
class PreferenceStudent extends ChallengeCueHeadBandit {
  imitate(decoded, enabled) {
    if (!enabled || decoded === null || decoded < 0) return 0;
    if (![0, 1, 2].includes(decoded)) throw Error('Invalid decoded DN preference');
    const head = this.heads[this.activeHead], q = this.logits(this.f);
    const mx = Math.max(...q), ex = q.map(x => Math.exp(x - mx));
    const z = ex.reduce((s, v) => s + v, 0);
    for (let a = 0; a < 3; a++) {
      const g = .03 * (Number(a === decoded) - ex[a] / z);
      head.b[a] += g;
      for (let j = 0; j < 3; j++) head.W[a][j] += g * this.f[j];
    }
    return 1;
  }
}
function student(frozen = false, source = null) {
  const net = makeNet(conf.studentSeed);
  const agent = new PreferenceStudent(net, conf.studentSeed, featureMean, featureStd);
  if (source) {
    source.agent.heads.forEach((head, h) => {
      head.W.forEach((w, a) => agent.heads[h].W[a].set(w));
      agent.heads[h].b.set(head.b);
    });
    agent.baseline = source.agent.baseline;
  }
  agent.explore = !frozen; agent.readoutLearn = !frozen;
  return { net, agent };
}
function schedule(seed, evaluation) {
  const rows = [];
  for (const offset of (evaluation ? [[0, 1]] : [[0, 0], [1, 0]]))
    for (let state = 0; state < 3; state++)
      for (let i = 0; i < (evaluation ? 4 : 6); i++) rows.push({ offset, start: state });
  const random = rng(seed);
  for (let i = rows.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [rows[i], rows[j]] = [rows[j], rows[i]];
  }
  return rows;
}
const trainingSchedule = schedule(conf.trainScheduleSeed, false);
const evaluationSchedule = schedule(conf.evalScheduleSeed, true);
function transition(state, action, spec, hadDelivery, finalStep) {
  const from = cells[state].map((v, i) => v + spec.offset[i]);
  const wanted = from.map((v, i) => v + actions[action][i]);
  const target = cells.findIndex(cell => cell.every((v, i) => v + spec.offset[i] === wanted[i]));
  const wall = state === 0 && target === 3;
  const collision = target < 0 || wall;
  const delivered = !collision && target === 3;
  const backward = state === 1 && target === 0;
  const forward = !collision && !delivered && !backward && target === state + 1;
  const next = delivered ? spec.start : collision ? state : target;
  let reward = delivered ? 1 : backward ? -.4 : forward ? .4 : -.8;
  if (finalStep && !hadDelivery && !delivered) reward -= .3;
  return { from, wanted, target, next, collision, delivered, backward, forward, reward };
}
function environmentCheck() {
  const routes = [0, 1, 2].map(start => {
    const spec = { start, offset: [0, 0] };
    let state = start, steps = 0, delivered = false;
    while (!delivered && steps < 4) {
      const cue = cueForState[state], action = optimalByCue[cue];
      const outcome = transition(state, action, spec, false, steps === 3);
      state = outcome.next; delivered = outcome.delivered; steps++;
    }
    return { start, steps, delivered };
  });
  const edges = [0, 1, 2].flatMap(state => [0, 1, 2].map(action => ({
    state, action, ...transition(state, action, { start: state, offset: [0, 0] }, false, false) })));
  const inBounds = [...trainingSchedule, ...evaluationSchedule].every(spec =>
    cells.every(cell => cell.every((v, i) => v + spec.offset[i] >= 0 && v + spec.offset[i] < 5)));
  const balance = trainingSchedule.length === 36 && evaluationSchedule.length === 12 &&
    [0, 1, 2].every(s => trainingSchedule.filter(x => x.start === s).length === 12 &&
      evaluationSchedule.filter(x => x.start === s).length === 4);
  const passed = routes.every(r => r.delivered && r.steps === 3 - r.start) &&
    inBounds && balance && edges.find(e => e.state === 0 && e.action === 1).collision &&
    edges.find(e => e.state === 1 && e.action === 2).backward &&
    edges.filter(e => e.collision).length === 5;
  return { routes, edges, trainingSchedule, evaluationSchedule, inBounds, balance, passed };
}
artifact.environment = environmentCheck();
if (!artifact.environment.passed) { await finish('stopped_environment'); process.exit(0); }

function sensoryProbe(net, cue, key) {
  net.resetState(); net.rand = rng(key + 2);
  const vpn = net.laneVpn[cue].slice(2);
  net.setDrive(vpn, 150);
  for (let t = 0; t < 240; t++) {
    if (t === 50) net.resetCounts();
    net.step();
  }
  const laneCounts = net.laneVpn.map(ids => Array.from(ids.slice(2))
    .reduce((s, id) => s + net.count[id], 0));
  const rates = net.groupMbon.map(ids => ids.reduce((s, id) => s + net.count[id], 0) / ids.length);
  const f = rates.map((rate, j) => (Math.log1p(rate) - featureMean[j]) / featureStd[j]);
  const maximum = Math.max(...laneCounts);
  return { cue, key, laneCounts, routed: maximum > 0 && laneCounts.filter(x => x === maximum).length === 1 ?
    laneCounts.indexOf(maximum) : -1, rates, f };
}
function sensoryBatch() {
  const net = makeNet(conf.studentSeed), calibration = [], heldout = [];
  for (let cue = 0; cue < 3; cue++) for (let repeat = 0; repeat < 12; repeat++) {
    const row = sensoryProbe(net, cue, conf.studentSeed * 10000000 + 500000 + cue * 100 + repeat);
    (repeat < 6 ? calibration : heldout).push({ ...row, repeat });
  }
  const centroids = [0, 1, 2].map(cue => [0, 1, 2].map(j =>
    calibration.filter(r => r.cue === cue).reduce((s, r) => s + r.f[j], 0) / 6));
  const predictions = heldout.map(row => {
    const distances = centroids.map(c => c.reduce((s, v, j) => s + (v - row.f[j]) ** 2, 0));
    return { ...row, distances, predicted: distances.indexOf(Math.min(...distances)) };
  });
  const byCue = [0, 1, 2].map(cue => ({ cue,
    routed: predictions.filter(r => r.cue === cue && r.routed === cue).length,
    recognized: predictions.filter(r => r.cue === cue && r.predicted === cue).length }));
  const correct = byCue.reduce((s, x) => s + x.recognized, 0);
  const distinct = new Set(centroids.map(x => JSON.stringify(x))).size === 3;
  const passed = correct >= 14 && byCue.every(x => x.routed === 6 && x.recognized >= 4) &&
    distinct && predictions.every(r => r.f.every(Number.isFinite)) && net.m.every(x => x === 1);
  return { calibration, heldout: predictions, centroids,
    summary: { correct, byCue, distinct, weightsUnchanged: true, passed } };
}
artifact.sensory = sensoryBatch();
if (!artifact.sensory.summary.passed) { await finish('stopped_sensory'); process.exit(0); }

// The later stages are deliberately separate below so an earlier failed gate cannot train either fly.

function episode(actor, spec, episodeIndex, kind, learn, targetFor = null, evaluation = false) {
  let state = spec.start, hadDelivery = false;
  const rows = [];
  for (let step = 0; step < 4; step++) {
    const cue = cueForState[state], index = episodeIndex * 4 + step;
    const key = (kind === 'teacher' ? conf.teacherSeed : conf.studentSeed) * 10000000 +
      (evaluation ? 900000 : 0) + episodeIndex * 1000 + step;
    const target = targetFor ? targetFor(index, cue) : null;
    const apply = action => transition(state, action, spec, hadDelivery, step === 3);
    const row = kind === 'teacher' ? teacherRound(actor, cue, key, learn, step === 0, apply) :
      studentRound(actor, cue, key, target, learn, apply);
    Object.assign(row, { episode: episodeIndex, step, layout: [...spec.offset], start: spec.start,
      state, cue, target });
    rows.push(row);
    hadDelivery ||= row.delivered;
    state = row.next;
  }
  return { layout: [...spec.offset], start: spec.start, success: hadDelivery,
    deliveries: rows.filter(r => r.delivered).length,
    collisions: rows.filter(r => r.collision).length, rows };
}
function teacherRound(s, cue, key, learn, episodeStart, apply) {
  const { net, agent } = s;
  if (episodeStart) net.resetState();
  net.rand = rng(key + 1); agent.rand = rng(key + 501);
  const vpn = net.laneVpn[cue].slice(2);
  net.setDrive(vpn, 150);
  let row;
  for (let t = 0; t < 400; t++) {
    if (t === 50) net.resetCounts();
    if (t === 240) {
      net.setDrive(vpn, 0);
      const action = agent.decide(net), outcome = apply(action);
      row = { action, rates: [...agent.rates], f: [...agent.f], scores: [...agent.scores],
        ...outcome, rewardStep: learn ? agent.alpha * (outcome.reward - agent.scores[action]) : 0 };
      agent.reward(net, outcome.reward);
      net.resetCounts();
    }
    if (t === 340) agent.afterDopamine(net, learn);
    net.step();
  }
  assert.ok(row && net.m.every(x => x === 1));
  return row;
}
function decode(counts) {
  const bits = counts.map(x => Number(x > 0));
  return codes.findIndex(code => code.every((v, i) => v === bits[i]));
}
function studentRound(s, cue, key, target, learn, apply) {
  const { net, agent } = s, dnCounts = [0, 0];
  net.resetState(); net.rand = rng(key + 2); agent.rand = rng(key + 501);
  const vpn = net.laneVpn[cue].slice(2);
  net.setDrive(vpn, 150);
  let row;
  for (let t = 0; t < 680; t++) {
    if (t === 50) net.resetCounts();
    if (t === 240) {
      net.setDrive(vpn, 0);
      const action = agent.decide(net);
      row = { action, rates: [...agent.rates], f: [...agent.f], scores: [...agent.scores],
        routedCue: agent.activeHead, laneCounts: [...agent.laneCounts],
        dnCounts: null, decoded: -1, rewardStep: 0, imitation: 0 };
      if (target !== null) DN.forEach((id, i) => {
        if (codes[target][i]) net.setDrive([id], 300);
      });
    }
    if (t === 280) {
      net.setDrive(DN, 0);
      row.dnCounts = [...dnCounts]; row.decoded = decode(dnCounts);
      const outcome = apply(row.action);
      Object.assign(row, outcome);
      agent.reward(net, outcome.reward);
      net.resetCounts();
    }
    if (t === 620) {
      row.rewardStep = learn ? agent.alpha * (row.reward - row.scores[row.action]) : 0;
      agent.afterDopamine(net, learn);
      row.imitation = agent.imitate(row.decoded, learn);
    }
    net.step();
    if (t >= 240 && t < 280) for (let i = 0; i < net.nSpikes; i++) {
      if (net.spikes[i] === DN[0]) dnCounts[0]++;
      else if (net.spikes[i] === DN[1]) dnCounts[1]++;
    }
  }
  assert.equal(row.routedCue, cue);
  assert.equal(row.decoded, target ?? -1);
  assert.ok(net.m.every(x => x === 1));
  return row;
}
function summarize(episodes) {
  const rows = episodes.flatMap(e => e.rows);
  return { episodes: episodes.length, decisions: rows.length,
    successes: episodes.filter(e => e.success).length,
    byStart: [0, 1, 2].map(start => ({ start,
      n: episodes.filter(e => e.start === start).length,
      successes: episodes.filter(e => e.start === start && e.success).length })),
    deliveries: episodes.reduce((s, e) => s + e.deliveries, 0),
    collisions: episodes.reduce((s, e) => s + e.collisions, 0),
    totalReward: rows.reduce((s, r) => s + r.reward, 0),
    actions: [0, 1, 2].map(action => rows.filter(r => r.action === action).length) };
}
function frozen(kind, source) {
  const actor = kind === 'teacher' ? teacher(true, source) : student(true, source);
  const before = kind === 'teacher' ? teacherWeights(actor.agent) : studentWeights(actor.agent);
  const episodes = evaluationSchedule.map((spec, i) => episode(actor, spec, i, kind, false, null, true));
  const after = kind === 'teacher' ? teacherWeights(actor.agent) : studentWeights(actor.agent);
  const rows = episodes.flatMap(e => e.rows);
  assert.equal(before, after);
  assert.ok(actor.net.m.every(x => x === 1));
  if (kind === 'student') assert.ok(rows.every(r => r.decoded === -1 &&
    r.dnCounts.every(x => x === 0) && r.imitation === 0 && r.rewardStep === 0));
  return { ...summarize(episodes), episodes,
    disconnected: true, explorationOff: true, weightsUnchanged: true };
}
function trainTeacher() {
  const actor = teacher(), before = teacherWeights(actor.agent);
  const episodes = trainingSchedule.map((spec, i) => episode(actor, spec, i, 'teacher', true));
  const evaluation = frozen('teacher', actor), after = teacherWeights(actor.agent);
  const passed = before !== after && evaluation.successes >= 9 &&
    evaluation.byStart.every(x => x.successes >= 2) &&
    evaluation.actions.every(n => n >= 4) && actor.net.m.every(x => x === 1);
  return { actor, result: { training: { ...summarize(episodes), episodes }, evaluation,
    initialWeights: before, finalWeights: after,
    summary: { passed, frozenSuccesses: evaluation.successes,
      byStart: evaluation.byStart, actions: evaluation.actions,
      weightsChanged: before !== after, neuralWeightsUnchanged: true } } };
}
const trainedTeacher = trainTeacher();
artifact.teacher = trainedTeacher.result;
if (!artifact.teacher.summary.passed) { await finish('stopped_teacher'); process.exit(0); }

function sourceTable(trained, untrained = false) {
  const actor = teacher(true, untrained ? null : trained), before = teacherWeights(actor.agent);
  const table = [];
  for (let index = 0; index < 144; index++) {
    const row = [];
    for (let cue = 0; cue < 3; cue++) {
      const key = conf.teacherSeed * 10000000 + 2000000 + index * 10 + cue;
      const probe = sensoryProbe(actor.net, cue, key);
      actor.agent.decide(actor.net);
      row.push({ index, cue, key, rates: [...actor.agent.rates],
        scores: [...actor.agent.scores], preference: actor.agent.scores.indexOf(Math.max(...actor.agent.scores)) });
    }
    table.push(row);
  }
  assert.equal(teacherWeights(actor.agent), before);
  assert.ok(actor.net.m.every(x => x === 1));
  return table;
}
const trainedTable = sourceTable(trainedTeacher), untrainedTable = sourceTable(trainedTeacher, true);
const byCue = [0, 1, 2].map(cue => ({ cue,
  correct: trainedTable.slice(0, 12).filter(row => row[cue].preference === optimalByCue[cue]).length,
  n: 12 }));
const sourceActions = [0, 1, 2].map(a => trainedTable.flat().filter(r => r.preference === a).length);
artifact.source = { trainedTable, untrainedTable,
  summary: { byCue, sourceActions, frozenWeights: true,
    passed: byCue.every(x => x.correct >= 9) && sourceActions.every(n => n > 0) } };
if (!artifact.source.summary.passed) { await finish('stopped_source'); process.exit(0); }

function receiverProbe(cue, repeat, message) {
  const net = makeNet(conf.studentSeed);
  const agent = new PreferenceStudent(net, conf.studentSeed, featureMean, featureStd);
  agent.explore = false; agent.readoutLearn = false;
  const before = studentWeights(agent), key = conf.studentSeed * 10000000 + 700000 + cue * 100 + repeat;
  net.resetState(); net.rand = rng(key + 2); agent.rand = rng(key + 501);
  const vpn = net.laneVpn[cue].slice(2), counts = [0, 0];
  net.setDrive(vpn, 150);
  let pre;
  for (let t = 0; t < 680; t++) {
    if (t === 50) net.resetCounts();
    if (t === 240) {
      net.setDrive(vpn, 0); agent.decide(net);
      pre = { rates: [...agent.rates], f: [...agent.f], routedCue: agent.activeHead,
        laneCounts: [...agent.laneCounts] };
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
  return { cue, repeat, message, key, ...pre, counts, decoded: decode(counts),
    weightsUnchanged: before === studentWeights(agent) && net.m.every(x => x === 1) };
}
function receiverBatch() {
  const rows = [];
  for (let cue = 0; cue < 3; cue++) for (let repeat = 0; repeat < 6; repeat++)
    for (const message of [null, 0, 1, 2]) rows.push(receiverProbe(cue, repeat, message));
  const conditions = [0, 1, 2].flatMap(cue => {
    const none = rows.filter(r => r.cue === cue && r.message === null);
    return [null, 0, 1, 2].map(message => {
      const x = rows.filter(r => r.cue === cue && r.message === message);
      const decoded = x.filter(r => r.decoded === (message ?? -1)).length;
      const routed = x.filter(r => r.routedCue === cue).length;
      const intended = x.every(r => message === null ? r.counts.every(n => n === 0) :
        codes[message].every((bit, i) => bit ? r.counts[i] > 0 : r.counts[i] === 0));
      const paired = message === null ? 6 : x.filter(r => {
        const n = none.find(v => v.repeat === r.repeat);
        return JSON.stringify([r.rates, r.f, r.routedCue]) === JSON.stringify([n.rates, n.f, n.routedCue]);
      }).length;
      return { cue, message, decoded, routed, intended, paired,
        passed: x.length === 6 && decoded === 6 && routed === 6 && intended && paired === 6 &&
          x.every(r => r.weightsUnchanged) };
    });
  });
  return { rows, conditions, summary: { passed: conditions.every(x => x.passed), conditions } };
}
artifact.receiver = receiverBatch();
if (!artifact.receiver.summary.passed) { await finish('stopped_receiver'); process.exit(0); }

const synthetic = student(), agent = synthetic.agent;
agent.activeHead = 0; agent.W = agent.heads[0].W; agent.b = agent.heads[0].b;
agent.f = [1, 0, 0];
const target = decode(artifact.receiver.rows.find(r => r.message === 0).counts);
const margin = () => { const q = agent.logits(agent.f); return q[target] - Math.max(...q.filter((_, i) => i !== target)); };
const initialMargin = margin(), before = studentWeights(agent);
agent.imitate(target, false);
const disabledUnchanged = studentWeights(agent) === before;
agent.imitate(target, true);
artifact.synthetic = { target, initialMargin, updatedMargin: margin(), disabledUnchanged,
  passed: disabledUnchanged && margin() > initialMargin };
if (!artifact.synthetic.passed) { await finish('stopped_synthetic'); process.exit(0); }

function trainArm(name, targetFor = null) {
  const actor = student(), initialWeights = studentWeights(actor.agent);
  const points = [{ episodes: 0, ...frozen('student', actor) }];
  const training = [];
  for (let i = 0; i < trainingSchedule.length; i++) {
    training.push(episode(actor, trainingSchedule[i], i, 'student', true, targetFor));
    if (checkpoints.includes(i + 1)) points.push({ episodes: i + 1, ...frozen('student', actor) });
  }
  return { name, initialWeights, finalWeights: studentWeights(actor.agent),
    training: { ...summarize(training), episodes: training }, checkpoints: points,
    neuralWeightsUnchanged: actor.net.m.every(x => x === 1) };
}
const alone = trainArm('alone');
artifact.independent = { arm: alone, summary: {
  passed: alone.checkpoints[0].successes <= 6 && alone.checkpoints.at(-1).successes >= 9 &&
    alone.checkpoints.at(-1).byStart.every(x => x.successes >= 2) &&
    alone.checkpoints.at(-1).actions.every(n => n > 0) && alone.neuralWeightsUnchanged,
  initial: alone.checkpoints[0].successes, final: alone.checkpoints.at(-1).successes,
  byStart: alone.checkpoints.at(-1).byStart } };
if (!artifact.independent.summary.passed) { await finish('stopped_independent'); process.exit(0); }

const genericRandom = rng(conf.genericSeed);
const genericTape = Array.from({ length: 144 }, () => Math.floor(3 * genericRandom()));
const targets = {
  untrained: (i, cue) => untrainedTable[i][cue].preference,
  shuffled: (i, cue) => trainedTable[i][(cue + 1) % 3].preference,
  generic: i => genericTape[i],
  trained: (i, cue) => trainedTable[i][cue].preference,
};
const byArm = { alone };
for (const name of ['untrained', 'shuffled', 'generic']) byArm[name] = trainArm(name, targets[name]);
function controlQuality(arm) {
  const rows = arm.training.episodes.flatMap(e => e.rows);
  return { n: rows.length, aligned: rows.filter(r => r.target === optimalByCue[r.cue]).length,
    actions: [0, 1, 2].map(a => rows.filter(r => r.target === a).length),
    byCue: [0, 1, 2].map(cue => ({ cue, n: rows.filter(r => r.cue === cue).length,
      aligned: rows.filter(r => r.cue === cue && r.target === optimalByCue[cue]).length })) };
}
const controls = { shuffled: controlQuality(byArm.shuffled), generic: controlQuality(byArm.generic) };
const controlPassed = Object.values(controls).every(q => q.n === 144 && q.aligned <= 72 &&
  q.actions.every(n => n > 0));
artifact.comparison = { controls, byArm, summary: { controlPassed, passed: false } };
if (!controlPassed) { await finish('stopped_control_validity'); process.exit(0); }
byArm.trained = trainArm('trained', targets.trained);
const trainingStart = byArm.alone.initialWeights;
assert.ok(arms.every(name => byArm[name].initialWeights === trainingStart &&
  byArm[name].training.decisions === 144 && byArm[name].checkpoints.length === checkpoints.length &&
  byArm[name].neuralWeightsUnchanged));
for (const name of arms) for (const point of byArm[name].checkpoints)
  assert.ok(point.disconnected && point.explorationOff && point.weightsUnchanged);
for (const name of arms.filter(x => x !== 'alone')) assert.ok(byArm[name].training.episodes
  .flatMap(e => e.rows).every(r => r.decoded === r.target && r.imitation === 1));
assert.ok(byArm.alone.training.episodes.flatMap(e => e.rows)
  .every(r => r.decoded === -1 && r.imitation === 0));
const first9 = Object.fromEntries(arms.map(name => [name,
  byArm[name].checkpoints.find(p => p.successes >= 9)?.episodes ?? null]));
const early = Object.fromEntries(arms.map(name => [name,
  [3, 6, 12, 24].reduce((s, n) => s + byArm[name].checkpoints.find(p => p.episodes === n).successes, 0) / 4]));
const trainedFinal = byArm.trained.checkpoints.at(-1);
const passed = first9.trained !== null && first9.trained <= 12 &&
  arms.filter(x => x !== 'trained').every(name => early.trained - early[name] >= 2 &&
    (first9[name] === null || first9.trained < first9[name])) &&
  trainedFinal.successes >= 9 && trainedFinal.byStart.every(x => x.successes >= 2);
artifact.comparison.summary = { controlPassed, passed, first9, early,
  final: Object.fromEntries(arms.map(name => [name, byArm[name].checkpoints.at(-1).successes])),
  trainingDecisions: Object.fromEntries(arms.map(name => [name, byArm[name].training.decisions])) };
await finish(passed ? 'passed' : 'failed');
