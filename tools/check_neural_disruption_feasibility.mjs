// One locked, frozen KC->MBON group-0 attenuation. No learning or teacher messages.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, open, access } from 'node:fs/promises';
import { Network, rng, wiring } from '../js/sim.js';
import { Opponent, result } from '../js/rps.js';
import { ChallengeCueHeadBandit, studentWeights } from '../js/rps_challenge_students.js';

const root = new URL('../', import.meta.url);
const protocolSha256 = createHash('sha256').update(await readFile(
  new URL('docs/neural-disruption-feasibility-protocol.md', root))).digest('hex');
const read = async name => JSON.parse(await readFile(new URL(`data/${name}`, root)));
const sub = await read('subgraph.json');
const { featureMean, featureStd } = (await read('rps_contextual_bandit.json')).design;
const archives = [
  { seed: 11015, name: 'rps_action_preference_strategy_b_initial.json',
    evaluationSeeds: [91000, 91001, 91003] },
  { seed: 11016, name: 'rps_action_preference_strategy_b_confirmation.json',
    evaluationSeeds: [91004, 91006, 91009] },
];
const output = new URL('data/neural_disruption_feasibility.json', root);
const preflight = process.argv[2] === 'preflight';
if (process.argv[2] && !preflight) throw Error('Use no argument or preflight');
const saved = await Promise.all(archives.map(async c => {
  const artifact = await read(c.name);
  assert.equal(artifact.protocol.studentSeed, c.seed);
  assert.deepEqual(artifact.protocol.evaluationSeeds, c.evaluationSeeds);
  const trained = artifact.pilot.byArm.trained;
  assert.equal(trained.checkpoints[6].rounds, 288);
  assert.equal(trained.checkpoints[6].rows.length, 72);
  return { ...c, weights: JSON.parse(trained.finalWeights),
    archived: trained.checkpoints[6] };
}));
const makeNet = seed => new Network(sub, wiring(sub, 'real', seed), { seed });
const checkNet = makeNet(saved[0].seed);
assert.equal(checkNet.plastic[0].length, 9210);
assert.equal(checkNet.plastic.reduce((s, x) => s + x.length, 0), 28238);
assert.equal(checkNet.groupMbon[0].length, 14);
assert.ok(checkNet.plastic[0].every(k =>
  sub.neurons.role[checkNet.preOfEdge(k)] === 'kc' &&
  sub.neurons.role[checkNet.col[k]] === 'mbon' &&
  checkNet.groupOf[checkNet.col[k]] === 0));
if (preflight) {
  console.log(JSON.stringify({ protocolSha256, students: saved.map(x =>
    ({ seed: x.seed, archivedWins: x.archived.wins, evaluationSeeds: x.evaluationSeeds })),
    targetEdges: checkNet.plastic[0].length, targetMbons: checkNet.groupMbon[0].length,
    multiplier: .5, roundsPerCondition: 72 }));
  process.exit(0);
}
try { await access(output); throw Error('Write-once result already exists'); }
catch (error) { if (error.code !== 'ENOENT') throw error; }

function makeStudent(savedStudent, condition) {
  const net = makeNet(savedStudent.seed);
  const agent = new ChallengeCueHeadBandit(net, savedStudent.seed, featureMean, featureStd);
  savedStudent.weights.heads.forEach((head, i) => {
    head.W.forEach((w, a) => agent.heads[i].W[a].set(w));
    agent.heads[i].b.set(head.b);
  });
  agent.baseline = savedStudent.weights.baseline;
  agent.explore = false; agent.readoutLearn = false;
  const target = [...net.plastic[0]];
  if (condition === 'sham') for (const k of target) net.m[k] = 1;
  if (condition === 'disrupted') for (const k of target) net.m[k] = .5;
  const targetSet = new Set(target);
  assert.ok(net.m.every((value, k) => value === (condition === 'disrupted' && targetSet.has(k) ? .5 : 1)));
  assert.equal(studentWeights(agent), JSON.stringify(savedStudent.weights));
  return { net, agent };
}
function round(net, agent, game, seed, episode, index) {
  const cue = game.observe().previousOpponent;
  const vpn = cue === null ? [] : [...net.laneVpn[cue].slice(2)];
  const key = seed * 10000000 + (100 + episode) * 1000 + index;
  net.resetState(); net.rand = rng(key + 2); agent.rand = rng(key + 501);
  if (vpn.length) net.setDrive(vpn, 150);
  let row;
  for (let t = 0; t < 680; t++) {
    if (t === 50) net.resetCounts();
    if (t === 240) {
      if (vpn.length) net.setDrive(vpn, 0);
      const action = agent.decide(net);
      row = { episode, round: index, key, cue, action,
        laneCounts: [...agent.laneCounts], routedCue: agent.activeHead,
        kcActive: [...net.idx.kc].reduce((n, id) => n + Number(net.count[id] > 0), 0),
        mbonCounts: [...net.idx.mbon].map(id => net.count[id]),
        rates: [...agent.rates], f: [...agent.f], scores: [...agent.scores],
        correctAction: cue === null ? null : cue,
        margin: cue === null ? null : agent.scores[cue] -
          Math.max(...agent.scores.filter((_, a) => a !== cue)),
        dnCounts: [0, 0] };
    }
    if (t === 280) {
      const played = game.play(row.action);
      row.opponent = played.opponent; row.reward = played.reward;
      agent.reward(net, played.reward); net.resetCounts();
    }
    if (t === 620) agent.afterDopamine(net, false);
    net.step();
  }
  assert.equal(row.reward, result(row.action, row.opponent));
  assert.equal(row.routedCue, cue === null ? 3 : cue);
  return row;
}
function run(savedStudent, condition) {
  const { net, agent } = makeStudent(savedStudent, condition);
  const before = studentWeights(agent), rows = [];
  for (let episode = 0; episode < 3; episode++) {
    const game = new Opponent({ strategy: 'previous_opponent', offset: 2,
      seed: savedStudent.evaluationSeeds[episode] });
    for (let index = 0; index < 24; index++)
      rows.push(round(net, agent, game, savedStudent.seed, episode, index));
  }
  assert.equal(studentWeights(agent), before, 'frozen controller changed');
  const targets = new Set(net.plastic[0]);
  assert.ok(net.m.every((v, k) => v === (condition === 'disrupted' && targets.has(k) ? .5 : 1)));
  return { condition, rows, weightsUnchanged: true,
    wins: rows.filter(r => r.reward === 1).length,
    byCue: [0, 1, 2].map(cue => {
      const x = rows.filter(r => r.cue === cue);
      return { cue, n: x.length, wins: x.filter(r => r.reward === 1).length,
        actions: [0, 1, 2].map(a => x.filter(r => r.action === a).length) };
    }) };
}
const pairs = [];
for (const s of saved) {
  const intact = run(s, 'intact');
  assert.equal(intact.wins, s.archived.wins, 'archived wins not reproduced');
  for (let i = 0; i < 72; i++) {
    const a = intact.rows[i], b = s.archived.rows[i];
    for (const field of ['cue', 'action', 'opponent', 'reward', 'routedCue'])
      assert.equal(a[field], b[field], `archived ${field} mismatch ${s.seed}/${i}`);
    for (const field of ['rates', 'f', 'scores'])
      assert.deepEqual(a[field], b[field], `archived ${field} mismatch ${s.seed}/${i}`);
  }
  const sham = run(s, 'sham');
  assert.deepEqual(sham, { ...intact, condition: 'sham' }, 'sham not exact');
  const disrupted = run(s, 'disrupted');
  const observed = condition => condition.rows.filter(r => r.cue !== null);
  const base = observed(sham), damaged = observed(disrupted);
  const mean = (rows, fn) => rows.reduce((sum, row) => sum + fn(row), 0) / rows.length;
  const group0Sham = mean(base, r => r.rates[0]);
  const group0Disrupted = mean(damaged, r => r.rates[0]);
  const actionFlips = damaged.filter((r, i) => r.action !== base[i].action).length;
  const routeCorrect = damaged.filter(r => r.routedCue === r.cue).length;
  const otherRates = [1, 2].map(j => ({ sham: mean(base, r => r.rates[j]),
    disrupted: mean(damaged, r => r.rates[j]) }));
  const meanKc = { sham: mean(base, r => r.kcActive),
    disrupted: mean(damaged, r => r.kcActive) };
  const metrics = { group0Sham, group0Disrupted,
    group0Reduction: group0Sham > 0 ? 1 - group0Disrupted / group0Sham : null,
    otherRates, meanKc, routeCorrect, actionFlips,
    winLoss: sham.wins - disrupted.wins,
    marginSham: mean(base, r => r.margin),
    marginDisrupted: mean(damaged, r => r.margin) };
  const gates = { exactReplay: true, shamExact: true, weightsUnchanged: true,
    neuralEffect: metrics.group0Reduction >= .2 && group0Disrupted > .05,
    sensoryPreserved: routeCorrect === 69 && meanKc.disrupted >= .5 * meanKc.sham,
    behavior: actionFlips >= 4 && metrics.winLoss >= 4 };
  gates.passed = Object.values(gates).every(Boolean);
  pairs.push({ seed: s.seed, evaluationSeeds: s.evaluationSeeds,
    targetEdgeCount: 9210, intact, sham, disrupted, metrics, gates });
  console.log(JSON.stringify({ seed: s.seed, wins: [intact.wins, sham.wins, disrupted.wins],
    metrics, gates }));
}
const artifact = { protocolSha256, status: pairs.every(p => p.gates.passed) ? 'passed' : 'failed',
  circuit: { target: 'KC->MBON group 0', mbons: 14, edges: 9210,
    shamMultiplier: 1, disruptedMultiplier: .5 }, pairs,
  allPassed: pairs.every(p => p.gates.passed) };
const file = await open(output, 'wx');
try { await file.writeFile(JSON.stringify(artifact) + '\n'); }
finally { await file.close(); }
console.log(JSON.stringify({ status: artifact.status, protocolSha256, output: output.pathname }));
