// Isolated synthetic sensor benchmark; protocol is locked in docs/sensor-classification-protocol.md.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, open, access } from 'node:fs/promises';
import { Network, rng, wiring } from '../js/sim.js';

const stage = process.argv[2];
if (!['initial', 'confirmation'].includes(stage)) throw Error('Use initial or confirmation');
const root = new URL('../', import.meta.url);
const read = async path => JSON.parse(await readFile(new URL(path, root)));
const protocolSha256 = createHash('sha256').update(await readFile(new URL('docs/sensor-classification-protocol.md', root))).digest('hex');
const output = new URL(`data/sensor_classification_${stage}.json`, root);
try { await access(output); throw Error('Write-once sensor result already exists'); }
catch (error) { if (error.code !== 'ENOENT') throw error; }
if (stage === 'confirmation') {
  const initial = await read('data/sensor_classification_initial.json');
  if (initial.status !== 'passed') throw Error('Initial sensor benchmark did not pass');
}
const sub = await read('data/subgraph.json');
const classNames = ['normal', 'overheating', 'excess_vibration'];
const means = [[50, 2.2], [71, 2.5], [51, 7]];
const sd = [8, 1.6];
const config = stage === 'initial' ? {
  teacher: 12031, student: 12032, teacherTrain: 610003, teacherEval: 610004,
  studentTrain: 610005, studentEval: 610006,
} : {
  teacher: 12033, student: 12034, teacherTrain: 610007, teacherEval: 610008,
  studentTrain: 610009, studentEval: 610010,
};
const artifact = { protocolSha256, stage, config, status: 'started', sensory: null };
async function finish(status) {
  artifact.status = status;
  const file = await open(output, 'wx');
  try { await file.writeFile(JSON.stringify(artifact, null, 2) + '\n'); }
  finally { await file.close(); }
  console.log(JSON.stringify({ status, sensory: artifact.sensory?.summary }));
}
function normal(random) {
  const u = Math.max(random(), Number.MIN_VALUE), v = random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function dataset(seed, perClass) {
  const random = rng(seed), rows = [];
  for (let label = 0; label < 3; label++) for (let i = 0; i < perClass; i++) {
    let temperature, vibration;
    do { temperature = means[label][0] + sd[0] * normal(random); }
    while (temperature < 35 || temperature > 85);
    do { vibration = means[label][1] + sd[1] * normal(random); }
    while (vibration < 0 || vibration > 10);
    rows.push({ label, temperature, vibration });
  }
  for (let i = rows.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [rows[i], rows[j]] = [rows[j], rows[i]];
  }
  return rows;
}
const splits = {
  calibration: dataset(610001, 24), sensoryHeldout: dataset(610002, 12),
  teacherTrain: dataset(config.teacherTrain, 32), teacherEval: dataset(config.teacherEval, 12),
  studentTrain: dataset(config.studentTrain, 32), studentEval: dataset(config.studentEval, 12),
};
const seen = new Set();
for (const [split, rows] of Object.entries(splits)) for (const row of rows) {
  const key = `${row.temperature},${row.vibration}`;
  assert.ok(!seen.has(key), `duplicate sensor pair in ${split}`);
  seen.add(key);
}
const ratesFor = row => [40 + 3 * (row.temperature - 35), 40 + 15 * row.vibration];
const makeNet = seed => new Network(sub, wiring(sub, 'real', seed), { seed });
function sense(net, row, key) {
  const hz = ratesFor(row);
  net.resetState(); net.rand = rng(key);
  net.setDrive(net.laneVpn[0].slice(2), hz[0]);
  net.setDrive(net.laneVpn[1].slice(2), hz[1]);
  for (let t = 0; t < 240; t++) {
    if (t === 50) net.resetCounts();
    net.step();
  }
  return { hz, rates: net.groupMbon.map(ids => ids.reduce((sum, i) => sum + net.count[i], 0) / ids.length) };
}
const calibrationNet = makeNet(12030), studentNet = makeNet(config.student);
const calibration = splits.calibration.map((row, i) => ({ ...row,
  ...sense(calibrationNet, row, 610001000 + i) }));
const heldout = splits.sensoryHeldout.map((row, i) => ({ ...row,
  ...sense(studentNet, row, 610002000 + i) }));
const replay = sense(studentNet, splits.sensoryHeldout[0], 610002000);
const exactReplay = JSON.stringify(replay) === JSON.stringify({ hz: heldout[0].hz, rates: heldout[0].rates });
function scaledVectors(train, test, transform) {
  const a = train.map(transform), b = test.map(transform);
  const mean = [0, 1, 2].slice(0, a[0].length).map(j => a.reduce((s, x) => s + x[j], 0) / a.length);
  const std = mean.map((mu, j) => Math.max(transform === neuralVector ? .05 : 1e-9,
    Math.sqrt(a.reduce((s, x) => s + (x[j] - mu) ** 2, 0) / a.length)));
  const scale = x => x.map((v, j) => (v - mean[j]) / std[j]);
  return { mean, std, train: a.map(scale), test: b.map(scale) };
}
const neuralVector = row => row.rates.map(Math.log1p);
const rawVector = row => [row.temperature, row.vibration];
function classify(train, held, features) {
  const vectors = scaledVectors(train, held, features);
  const centroids = [0, 1, 2].map(label => vectors.train.reduce((sum, v, i) => {
    if (train[i].label === label) v.forEach((x, j) => { sum[j] += x / 24; });
    return sum;
  }, Array(vectors.mean.length).fill(0)));
  const rows = vectors.test.map((v, i) => {
    const distances = centroids.map(c => c.reduce((s, x, j) => s + (x - v[j]) ** 2, 0));
    const predicted = distances.indexOf(Math.min(...distances));
    return { label: held[i].label, predicted, distances, features: v };
  });
  const byClass = [0, 1, 2].map(label => rows.filter(r => r.label === label && r.predicted === label).length);
  return { mean: vectors.mean, std: vectors.std, centroids, rows,
    correct: byClass.reduce((a, b) => a + b, 0), byClass };
}
const neural = classify(calibration, heldout, neuralVector);
const raw = classify(calibration, heldout, rawVector);
const varying = [0, 1, 2].filter(j => new Set(heldout.map(r => r.rates[j])).size > 1).length;
const passed = neural.correct >= 26 && neural.byClass.every(n => n >= 7) &&
  raw.correct >= 26 && raw.byClass.every(n => n >= 7) &&
  varying >= 2 && exactReplay &&
  [...calibration, ...heldout].every(r => r.rates.every(Number.isFinite)) &&
  calibrationNet.m.every(x => x === 1) && studentNet.m.every(x => x === 1);
artifact.sensory = { classNames, splits, calibration, heldout, neural, raw,
  summary: { neuralCorrect: neural.correct, neuralByClass: neural.byClass,
    rawCorrect: raw.correct, rawByClass: raw.byClass, varyingComponents: varying,
    exactReplay, weightsUnchanged: true, passed } };
if (!passed) await finish('stopped_sensory');
else throw Error('Sensory gate passed: complete the locked downstream stages before interpreting success');
