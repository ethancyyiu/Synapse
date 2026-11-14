// Frozen, no-message population diagnostic; protocol locked in docs/lab-notebook.md.
// node tools/check_rps_noisy_cue_population.mjs
import { open, readFile } from 'node:fs/promises';
import { Network, rng, wiring } from '../js/sim.js';
import { ChallengeCueHeadBandit, studentWeights } from '../js/rps_challenge_students.js';

const read = async name => JSON.parse(await readFile(new URL(`../data/${name}`, import.meta.url), 'utf8'));
const archived = await read('rps_noisy_guided_receiver.json');
const sub = await read('subgraph.json');
const { featureMean, featureStd } = (await read('rps_contextual_bandit.json')).design;
const centroids = (await read('rps_readout_information.json')).centroids;
const output = new URL('../data/rps_noisy_cue_population.json', import.meta.url);
const targets = [3141, 3165];
const conditions = [
  { seed: 10917, firstCue: 1, targetPhase: 'second', role: 'failure' },
  { seed: 10918, firstCue: 0, targetPhase: 'first', role: 'failure' },
  { seed: 10917, firstCue: 0, targetPhase: 'second', role: 'cue2_control' },
  { seed: 10918, firstCue: 2, targetPhase: 'first', role: 'cue2_control' },
];
const wires = new Map();
const counts = (net, ids) => Array.from(ids, i => net.count[i]);
const kcActive = net => Array.from(net.idx.kc).flatMap(i => net.count[i] ? [[i, net.count[i]]] : []);
const nonzero = xs => xs.reduce((n, x) => n + Number(x !== 0), 0);
const state = (net, baselineV) => ({
  t: net.t, vOffBaseline: net.v.reduce((n, x) => n + Number(Math.abs(x - baselineV) > 1e-12), 0),
  gNonzero: nonzero(net.g), refrNonzero: nonzero(net.refr),
  traceNonzero: nonzero(net.trace), countNonzero: nonzero(net.count),
  lastSpikeNotReset: net.lastSpike.reduce((n, x) => n + Number(x !== -1e9), 0),
  ringNonzero: net.ring.map(nonzero), driveNonzero: nonzero(net.drive),
});
const windowCounts = net => ({
  vpnLanes: net.laneVpn.map(ids => counts(net, ids.slice(2)).reduce((s, x) => s + x, 0)),
  kcActive: kcActive(net),
  individualMbon: net.groupMbon.map(ids => counts(net, ids)),
});
function decision(net, agent, cue, early) {
  const rates = [...agent.rates], features = rates.map(Math.log1p);
  const distances = centroids.map(c => c.reduce((s, x, i) => s + (features[i] - x) ** 2, 0));
  const ordered = [...distances].sort((a, b) => a - b);
  const decoderChoice = distances.indexOf(ordered[0]);
  const lanes = [...agent.laneCounts];
  return { cue, estimatedCue: agent.activeHead, rates, features,
    vpnLaneCounts: lanes, vpnMargin: lanes[cue] - Math.max(...lanes.filter((_, i) => i !== cue)),
    early, late: windowCounts(net), centroidDistances: distances, decoderChoice,
    decisionMargin: ordered[1] - ordered[0], recognized: decoderChoice === cue,
    dnTargetDriveHz: targets.map(i => net.drive[i]),
  };
}
function receiverRound(net, agent, cue, key) {
  const vpn = Array.from(net.laneVpn[cue].slice(2));
  net.rand = rng(key + 2); agent.rand = rng(key + 501);
  net.setDrive(vpn, 150);
  let early, row, dnEverDriven = false;
  for (let t = 0; t < 680; t++) {
    if (t === 50) { early = windowCounts(net); net.resetCounts(); }
    if (t === 240) {
      net.setDrive(vpn, 0);
      agent.decide(net);
      row = decision(net, agent, cue, early);
    }
    if (t === 280) net.resetCounts();
    if (targets.some(i => net.drive[i] !== 0)) dnEverDriven = true;
    net.step();
  }
  row.dnEverDriven = dnEverDriven;
  return row;
}
function runPair(condition, repeat) {
  const { seed, firstCue } = condition;
  if (!wires.has(seed)) wires.set(seed, wiring(sub, 'real', seed));
  const net = new Network(sub, wires.get(seed), { seed });
  const agent = new ChallengeCueHeadBandit(net, seed, featureMean, featureStd);
  agent.explore = false; agent.readoutLearn = false;
  const baselineV = net.v[0], beforeWeights = studentWeights(agent);
  const key = seed * 10000000 + (300 + 12 * firstCue + repeat) * 1000;
  const initialBefore = state(net, baselineV);
  net.resetState();
  const initialAfter = state(net, baselineV);
  const first = receiverRound(net, agent, firstCue, key);
  const betweenBefore = state(net, baselineV);
  net.resetState();
  const betweenAfter = state(net, baselineV);
  const second = receiverRound(net, agent, (firstCue + 2) % 3, key + 1);
  return { ...condition, repeat, key, first, second,
    reset: { initialBefore, initialAfter, betweenBefore, betweenAfter },
    weightsUnchanged: studentWeights(agent) === beforeWeights && net.m.every(x => x === 1) };
}
function reproduce(row) {
  const old = archived.probes.find(p => p.seed === row.seed && p.cue === row.firstCue &&
    p.repeat === row.repeat && p.bit === null);
  if (!old) return { seed: row.seed, firstCue: row.firstCue, repeat: row.repeat, error: 'missing archive row' };
  const phases = ['first', 'second'].map(phase => {
    const actual = row[phase], expected = old[phase];
    const maxRateDiff = Math.max(...actual.rates.map((x, i) => Math.abs(x - expected.rates[i])));
    const ok = actual.cue === expected.cue && maxRateDiff <= 1e-12 &&
      actual.estimatedCue === expected.estimatedCue && actual.recognized === expected.recognized;
    return { phase, maxRateDiff, ok };
  });
  return { seed: row.seed, firstCue: row.firstCue, repeat: row.repeat, phases,
    ok: phases.every(p => p.ok) && row.weightsUnchanged &&
      !row.first.dnEverDriven && !row.second.dnEverDriven };
}
async function save(value) {
  const file = await open(output, 'wx');
  try { await file.writeFile(`${JSON.stringify(value, null, 2)}\n`); }
  finally { await file.close(); }
}
const protocol = { source: 'docs/lab-notebook.md locked no-message RPS population diagnostic',
  seeds: [10917, 10918], repeats: 12, conditions, branches: ['no_message'],
  cueWindowMs: [0, 240], countResetMs: 50, decisionMs: 240,
  pairReset: true, key: 's*10000000+(300+12*firstCue+repeat)*1000; second key +1',
  rateTolerance: 1e-12, fixedDecoder: true, rewards: false, learning: false };
const rows = [], reproduction = [];
for (const condition of conditions) {
  if (condition.role !== 'failure') continue;
  for (let repeat = 0; repeat < 12; repeat++) {
    const row = runPair(condition, repeat), check = reproduce(row);
    rows.push(row); reproduction.push(check);
  }
}
if (reproduction.some(x => !x.ok)) {
  await save({ protocol, status: 'stopped_reproduction', reproduction });
  console.log(JSON.stringify({ status: 'stopped_reproduction', mismatches: reproduction.filter(x => !x.ok) }));
  process.exit(0);
}
for (const condition of conditions) {
  if (condition.role !== 'cue2_control') continue;
  for (let repeat = 0; repeat < 12; repeat++) {
    const row = runPair(condition, repeat), check = reproduce(row);
    rows.push(row); reproduction.push(check);
  }
}
if (reproduction.some(x => !x.ok)) {
  await save({ protocol, status: 'stopped_reproduction', reproduction });
  console.log(JSON.stringify({ status: 'stopped_reproduction', mismatches: reproduction.filter(x => !x.ok) }));
  process.exit(0);
}
await save({ protocol, status: 'population_recorded', reproduction, pairs: rows });
console.log(JSON.stringify({ status: 'population_recorded', pairs: rows.length,
  reproduced: reproduction.length, cue0Correct: conditions.slice(0, 2).map(c =>
    rows.filter(r => r.seed === c.seed && r.firstCue === c.firstCue && r.role === c.role)
      .filter(r => r[c.targetPhase].recognized).length) }));
