// Opt-in post-sensory action-preference teaching; locked in docs/lab-notebook.md.
import assert from 'node:assert/strict';
import { readFile, open, access } from 'node:fs/promises';
import { Network, rng, wiring } from '../js/sim.js';
import { Opponent, RpsContextualBandit, RpsEpisode, result } from '../js/rps.js';
import { ChallengeCueHeadBandit, studentWeights } from '../js/rps_challenge_students.js';

const read = async name => JSON.parse(await readFile(new URL('../data/' + name, import.meta.url)));
const output = new URL('../data/rps_action_preference_teaching_v3.json', import.meta.url);
try { await access(output); throw Error('Write-once teaching result already exists'); }
catch (error) { if (error.code !== 'ENOENT') throw error; }
const sub = await read('subgraph.json');
const receiver = await read('rps_action_preference_receiver_v3.json');
assert.equal(receiver.status, 'passed');
assert.deepEqual(receiver.batches.map(b => b.seed), [11006, 11007, 11008]);
const { featureMean, featureStd } = (await read('rps_contextual_bandit.json')).design;
const DN = [3141, 3165], codes = [[1, 0], [0, 1], [1, 1]];
const seeds = [11006, 11007, 11008], teacherSeed = 830;
const arms = ['trained', 'alone', 'untrained', 'shuffled', 'generic'];
const checkpoints = [0, 6, 12, 24, 48, 96, 288];
const protocol = { teacherSeed, studentSeeds: seeds.slice(1), receiverArtifact: 'rps_action_preference_receiver_v3.json',
  source: 'frozen reward-trained Strategy A teacher 830, continuous neural state and MBON-driven Q argmax',
  code: codes, dn: DN, dnHz: 300, cueWindow: [0, 240], dnWindow: [240, 280],
  rewardTime: 280, updateTime: 620, roundEnd: 680,
  opponent: 'Strategy A, 41000..41011 training, 51000..51002 held-out',
  studentKey: 'seed*10000000+episode*1000+round; rng(key+2), rng(key+501)',
  teacherKey: '830*10000000+episode*1000+round; rng(key+1)',
  rewardAlpha: .03, imitationAlpha: .03, epsilon: .20, budget: 288,
  checkpoints, arms, shuffledSeed: 2026110602, genericSeed: 2026110603,
  sourceGate: 'trained preference >=70% win per observed cue; shuffled/generic <=60% per cue, all actions present, shuffle >=50% changed',
  aloneGate: 'initial <=50/72, final >=58/72 and >=70% each observed cue for both',
  pilotGate: 'both seeds: trained early mean at 6/12/24/48 >=4/72 over each control; first 58/72 <=48 and earlier than every control; final >=58/72 and >=70% each cue' };
const artifact = { protocol, status: 'started', teacher: null, synthetic: null,
  independent: null, source: null, pilot: null };
async function finish(status) {
  artifact.status = status;
  const file = await open(output, 'wx');
  try { await file.writeFile(JSON.stringify(artifact, null, 2) + '\n'); }
  finally { await file.close(); }
  console.log(JSON.stringify({ status, teacher: artifact.teacher, synthetic: artifact.synthetic,
    independent: artifact.independent?.summary, source: artifact.source,
    pilot: artifact.pilot?.summary }));
}
function decode(counts) {
  const bits = counts.map(x => Number(x > 0));
  return codes.findIndex(code => code.every((value, i) => value === bits[i]));
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
        laneCounts: [...agent.laneCounts], target, decoded: null, dnCounts: null,
        opponent: null, reward: null, rewardStep: 0, imitation: 0 };
      if (target !== null) DN.forEach((id, i) => {
        if (codes[target][i]) net.setDrive([id], 300);
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
      (target === null ? row.decoded !== -1 : row.decoded !== target) ||
      row.reward !== result(row.action, row.opponent))
    throw Error('Student round validity failed');
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
const example = receiver.batches[0].rows.find(r => r.message === 0 && r.counts[0] > 0);
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
const shuffle = shuffled(correct), randomGeneric = rng(2026110603);
const generic = correct.rows.map(row => row.cue === null ? null : Math.floor(3 * randomGeneric()));
const quality = target => [0, 1, 2].map(cue => {
  const indices = correct.rows.flatMap((row, i) => row.cue === cue ? [i] : []);
  return { cue, n: indices.length,
    correct: indices.filter(i => win(target[i], correct.rows[i].opponent)).length,
    actions: [0, 1, 2].map(a => indices.filter(i => target[i] === a).length) };
});
const trainedTarget = correct.rows.map(r => r.preference);
const genericQuality = quality(generic), shuffledQuality = quality(shuffle.target);
const present = target => [0, 1, 2].every(a => target.includes(a));
artifact.source = { trainedByCue: correct.byCue, genericQuality, shuffledQuality,
  shuffledMoved: shuffle.moved, actionTotals: {
    trained: [0, 1, 2].map(a => trainedTarget.filter(x => x === a).length),
    shuffled: [0, 1, 2].map(a => shuffle.target.filter(x => x === a).length),
    generic: [0, 1, 2].map(a => generic.filter(x => x === a).length) },
  frozenWeights: correct.frozenWeights && untrained.frozenWeights,
  passed: correct.passed && correct.frozenWeights && untrained.frozenWeights &&
    shuffle.moved >= 138 && present(shuffle.target) && present(generic) &&
    genericQuality.every(x => x.correct / x.n <= .60) &&
    shuffledQuality.every(x => x.correct / x.n <= .60) };
if (!artifact.source.passed) { await finish('stopped_source_or_controls'); process.exit(0); }
const targets = { trained: trainedTarget, untrained: untrained.rows.map(r => r.preference),
  shuffled: shuffle.target, generic };
const pilot = seeds.slice(1).map((seed, i) => {
  const byArm = { alone: artifact.independent.arms[i] };
  for (const name of arms.filter(x => x !== 'alone'))
    byArm[name] = trainArm(seed, name, targets[name]);
  const initial = byArm.alone.initial;
  const opponentSchedule = byArm.alone.training.map(r => [r.cue, r.opponent]);
  assert.ok(arms.every(name => byArm[name].initial === initial &&
    JSON.stringify(byArm[name].training.map(r => [r.cue, r.opponent])) ===
      JSON.stringify(opponentSchedule)), 'Student starts or opponent streams differ');
  const first58 = Object.fromEntries(arms.map(name => [name,
    byArm[name].checkpoints.find(x => x.wins >= 58)?.rounds ?? null]));
  const early = Object.fromEntries(arms.map(name => [name,
    [6, 12, 24, 48].reduce((sum, n) =>
      sum + byArm[name].checkpoints.find(x => x.rounds === n).wins, 0) / 4]));
  const controls = arms.filter(x => x !== 'trained');
  const passed = controls.every(name => early.trained - early[name] >= 4 &&
      (first58[name] === null || first58.trained < first58[name])) &&
    first58.trained !== null && first58.trained <= 48 &&
    byArm.trained.checkpoints.at(-1).wins >= 58 &&
    byArm.trained.checkpoints.at(-1).byCue.every(x => x.wins / x.n >= .70) &&
    Object.values(byArm).every(x => x.frozenSynapses && x.checkpoints.length === 7 &&
      x.training.every(row => row.target === null ?
        row.decoded === -1 && row.imitation === 0 :
        row.target === row.decoded && row.imitation === 1));
  return { seed, byArm, first58, early, passed };
});
artifact.pilot = { pairs: pilot, summary: { passed: pilot.every(x => x.passed),
  seeds: pilot.map(x => ({ seed: x.seed, passed: x.passed, first58: x.first58,
    early: x.early, final: Object.fromEntries(arms.map(name =>
      [name, x.byArm[name].checkpoints.at(-1).wins])) })) } };
await finish(artifact.pilot.summary.passed ? 'pilot_passed_unconfirmed' : 'pilot_failed');
