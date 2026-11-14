// One write-once, reward-free validation of the existing pooled-MBON decoder recipe.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { access, open, readFile } from 'node:fs/promises';
import { Network, P, rng, wiring } from '../js/sim.js';
import { nearestCue } from './practical_transfer_controller.mjs';

const root = new URL('../', import.meta.url);
const read = async path => JSON.parse(await readFile(new URL(path, root)));
const output = new URL('data/package_recovery_sensory_validation.json', root);
try { await access(output); throw Error('Write-once sensory-validation artifact already exists'); }
catch (error) { if (error.code !== 'ENOENT') throw error; }
const protocolSha256 = createHash('sha256').update(
  await readFile(new URL('docs/package-recovery-sensory-validation-protocol.md', root))).digest('hex');
const old = await read('data/practical_transfer_initial.json');
assert.equal(old.status, 'stopped_teacher');
const archivedCentroids = old.sensory.teacher.centroids;
const sub = await read('data/subgraph.json');
const { featureMean, featureStd } = (await read('data/rps_contextual_bandit.json')).design;
const CAL = 17101, VALID = [{ seed: 17102, orderSeed: 2026110961 },
  { seed: 17103, orderSeed: 2026110962 }];
const key = (seed, offset, index) => seed * 10000000 + (offset + index) * 1000 + 1;
const uniqueKeys = [];
for (let cue = 0; cue < 3; cue++) for (let repeat = 0; repeat < 12; repeat++)
  uniqueKeys.push(key(CAL, 2700, cue * 12 + repeat) >>> 0);
for (const item of VALID) for (let index = 0; index < 72; index++)
  uniqueKeys.push(key(item.seed, 0, index) >>> 0);
assert.equal(new Set(uniqueKeys).size, 180, 'Effective RNG stream keys overlap');

function state(net) {
  const nonzero = a => a.reduce((s, v) => s + Number(v !== 0), 0);
  return { t: net.t, meanV: net.v.reduce((s, v) => s + v, 0) / net.N,
    nonzeroG: nonzero(net.g), nonzeroTrace: nonzero(net.trace),
    nonzeroRefr: nonzero(net.refr), nonzeroCount: nonzero(net.count),
    totalCount: net.count.reduce((s, v) => s + v, 0),
    nonzeroRing: net.ring.reduce((s, row) => s + nonzero(row), 0),
    nonzeroDrive: nonzero(net.drive) };
}
function probe(net, seed, cue, streamKey, index, stream) {
  net.resetState(); net.rand = rng(streamKey);
  const beforeDrive = state(net);
  const vpn = net.laneVpn[cue].slice(2);
  net.setDrive(vpn, 150);
  for (let t = 0; t < 240; t++) { if (t === 50) net.resetCounts(); net.step(); }
  net.setDrive(vpn, 0);
  const atDecision = state(net);
  const vpnLaneCounts = net.laneVpn.map(ids =>
    [...ids.slice(2)].reduce((s, id) => s + net.count[id], 0));
  const kcActive = [...net.idx.kc].filter(id => net.count[id] > 0)
    .map(id => [id, net.count[id]]);
  const mbon = [...net.idx.mbon].map(id =>
    ({ id, group: net.groupOf[id], count: net.count[id] }));
  const pooled = net.groupMbon.map(ids =>
    ids.reduce((s, id) => s + net.count[id], 0) / ids.length);
  const f = pooled.map((rate, j) => (Math.log1p(rate) - featureMean[j]) / featureStd[j]);
  const maxVpn = Math.max(...vpnLaneCounts);
  const vpnRoute = maxVpn > 0 && vpnLaneCounts.filter(n => n === maxVpn).length === 1 ?
    vpnLaneCounts.indexOf(maxVpn) : -1;
  return { seed, stream, index, cue, key: streamKey, beforeDrive, atDecision,
    vpnLaneCounts, vpnRoute, kcActive, mbon, pooled, f };
}
const makeNet = seed => new Network(sub, wiring(sub, 'real', seed), { seed });
const calibrationNet = makeNet(CAL), calibration = [], internalCheck = [];
for (let cue = 0; cue < 3; cue++) for (let repeat = 0; repeat < 12; repeat++) {
  const row = probe(calibrationNet, CAL, cue, key(CAL, 2700, cue * 12 + repeat),
    repeat, repeat < 6 ? 'calibration' : 'internal_check');
  (repeat < 6 ? calibration : internalCheck).push(row);
}
const centroids = [0, 1, 2].map(cue => [0, 1, 2].map(j =>
  calibration.filter(r => r.cue === cue).reduce((s, r) => s + r.f[j], 0) / 6));
const distance = (f, c) => f.reduce((s, v, j) => s + (v - c[j]) ** 2, 0);
function score(row, centers, field) {
  const d = centers.map(c => distance(row.f, c));
  const sorted = [...d].sort((a, b) => a - b);
  row[field] = { distances: d, prediction: nearestCue(row.f, centers),
    margin: sorted[1] - sorted[0] };
}
for (const row of [...calibration, ...internalCheck]) score(row, centroids, 'fixedDecoder');
function shuffledCues(orderSeed) {
  const cues = [0, 1, 2].flatMap(cue => Array(24).fill(cue));
  const random = rng(orderSeed);
  for (let i = cues.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [cues[i], cues[j]] = [cues[j], cues[i]];
  }
  return cues;
}
const validation = VALID.map(({ seed, orderSeed }) => {
  const net = makeNet(seed), order = shuffledCues(orderSeed), rows = [];
  for (let i = 0; i < order.length; i++) {
    const row = probe(net, seed, order[i], key(seed, 0, i), i, 'training_style');
    score(row, centroids, 'fixedDecoder');
    score(row, archivedCentroids, 'archivedDecoder');
    rows.push(row);
  }
  return { seed, orderSeed, order, rows, neuralWeightsUnchanged: net.m.every(v => v === 1) };
});
const calibrationWeightsUnchanged = calibrationNet.m.every(v => v === 1);
function matrix(rows, field) {
  return [0, 1, 2].map(cue => [0, 1, 2].map(prediction =>
    rows.filter(r => r.cue === cue && r[field].prediction === prediction).length));
}
function scored(rows, field) {
  const confusion = matrix(rows, field), perCue = confusion.map((row, cue) => row[cue]);
  return { n: rows.length, confusion, perCue,
    correct: perCue.reduce((s, x) => s + x, 0), c0ToC2: confusion[0][2] };
}
const initialReport = scored(internalCheck, 'fixedDecoder');
for (const item of validation) {
  item.fixed = scored(item.rows, 'fixedDecoder');
  item.archived = scored(item.rows, 'archivedDecoder');
  item.vpnByCue = [0, 1, 2].map(cue => item.rows.filter(r =>
    r.cue === cue && r.vpnRoute === cue).length);
}
const allRows = [...calibration, ...internalCheck, ...validation.flatMap(x => x.rows)];
const resetValid = allRows.every(r => r.beforeDrive.t === 0 &&
  Math.abs(r.beforeDrive.meanV - P.v0) < 1e-10 &&
  ['nonzeroG', 'nonzeroTrace', 'nonzeroRefr', 'nonzeroCount', 'totalCount',
    'nonzeroRing', 'nonzeroDrive'].every(name => r.beforeDrive[name] === 0) &&
  r.atDecision.t === 240);
const recordingValid = allRows.length === 180 && calibration.length === 18 &&
  internalCheck.length === 18 && validation.every(x =>
    x.rows.length === 72 && x.neuralWeightsUnchanged &&
    [0, 1, 2].every(cue => x.order.filter(v => v === cue).length === 24)) &&
  calibrationWeightsUnchanged && resetValid &&
  allRows.every(r => r.kcActive.every(([id, count]) =>
    Number.isInteger(id) && count > 0) && r.mbon.length === 42 &&
    r.f.every(Number.isFinite) && r.fixedDecoder.distances.every(Number.isFinite));
const vpnPassed = validation.every(x => x.vpnByCue.every(n => n === 24));
const decoderPassed = validation.every(x => x.fixed.correct >= 63 &&
  x.fixed.perCue.every(n => n >= 20) && x.fixed.c0ToC2 <= 4);
const gate = { recordingValid, vpnPassed, decoderPassed,
  passed: recordingValid && vpnPassed && decoderPassed };
const artifact = { protocolSha256, sourceArtifact: 'data/practical_transfer_initial.json',
  sourceStatus: old.status, status: gate.passed ? 'passed' : 'failed',
  config: { calibrationSeed: CAL, validation: VALID, cueHz: 150,
    cueWindow: [0, 240], countReset: 50, calibrationPerCue: 6,
    internalCheckPerCue: 6, validationPerCuePerSeed: 24,
    featureMean, featureStd },
  centroids, archivedCentroids, calibration, internalCheck, initialReport,
  calibrationWeightsUnchanged, validation, gate };
const file = await open(output, 'wx');
try { await file.writeFile(JSON.stringify(artifact, null, 2) + '\n'); }
finally { await file.close(); }
console.log(JSON.stringify({ status: artifact.status, gate, initialReport,
  validation: validation.map(x => ({ seed: x.seed, vpn: x.vpnByCue,
    fixed: x.fixed, archived: x.archived })) }));
