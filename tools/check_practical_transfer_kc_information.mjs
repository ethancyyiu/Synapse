// Locked reward-free population-information comparison. Does not train a controller.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, open } from 'node:fs/promises';
import { Network, rng, wiring } from '../js/sim.js';

const root = new URL('../', import.meta.url);
const protocol = await readFile(new URL('docs/practical-transfer-kc-information-protocol.md', root));
const protocolSha256 = createHash('sha256').update(protocol).digest('hex');
const archive = JSON.parse(await readFile(new URL('data/practical_transfer_redesign_initial_sensory.json', root)));
const sub = JSON.parse(await readFile(new URL('data/subgraph.json', root)));
const output = new URL('data/practical_transfer_kc_information.json', root);
const configs = [
  { neuralSeed: 19202, caseSeed: 20261109202 },
  { neuralSeed: 19203, caseSeed: 20261109203 },
];
const kcIds = sub.neurons.role.flatMap((role, id) => role === 'kc' ? [id] : []);
const mbonIds = sub.neurons.role.flatMap((role, id) => role === 'mbon' ? [id] : []);
assert.equal(kcIds.length, 2045); assert.equal(mbonIds.length, 42);
assert.equal(archive.calibrationRows.length, 96);
assert.equal(archive.validationRows.length, 96);
const kcPosition = new Map(kcIds.map((id, i) => [id, i]));
const mean = x => x.reduce((a, b) => a + b, 0) / x.length;
const dot = (x, y) => x.reduce((s, v, i) => s + v * y[i], 0);
const sd = x => Math.max(Math.sqrt(x.reduce((s, v) => s + (v - mean(x)) ** 2, 0) / (x.length - 1)), .1);
const rawKc = row => {
  const x = new Float64Array(kcIds.length);
  for (const [id, count] of row.kcActive) {
    const j = kcPosition.get(id);
    assert.ok(j !== undefined && Number.isInteger(count) && count > 0);
    x[j] = Math.log1p(count);
  }
  return x;
};
const rawVpn = row => row.vpn.map(v => Math.log1p(v));
const rawMbon = row => row.mbon.map((m, j) => {
  assert.equal(m.id, mbonIds[j]); return Math.log1p(m.count);
});
const rawBucket = row => {
  const x = new Float64Array(42);
  for (const [id, count] of row.kcActive) x[kcPosition.get(id) % 42] += count;
  return Array.from(x, v => Math.log1p(v));
};
const cal = archive.calibrationRows;
const standardize = (raw, d) => {
  const values = cal.map(raw);
  const center = Array.from({ length: d }, (_, j) => mean(values.map(v => v[j])));
  const scale = Array.from({ length: d }, (_, j) => sd(values.map(v => v[j])));
  return row => raw(row).map((v, j) => (v - center[j]) / scale[j] / Math.sqrt(d));
};
const vpnInput = standardize(rawVpn, 3);
const mbonInput = row => rawMbon(row).map((v, j) =>
  (v - archive.scaler[j].mean) / archive.scaler[j].sd / Math.sqrt(42));
const bucketInput = standardize(rawBucket, 42);
const calKc = cal.map(rawKc);
const kcCenter = new Float64Array(kcIds.length);
for (const row of calKc) for (let j = 0; j < kcCenter.length; j++) kcCenter[j] += row[j] / calKc.length;
const kcRadius = Math.max(Math.sqrt(mean(calKc.map(x =>
  x.reduce((s, v, j) => s + (v - kcCenter[j]) ** 2, 0)))), .1);
const kcInput = row => Array.from(rawKc(row), (v, j) => (v - kcCenter[j]) / kcRadius);
const representations = { vpn: vpnInput, kc: kcInput, mbon: mbonInput, kc42: bucketInput };

function solve(matrix, rhs) {
  const n = rhs.length, a = matrix.map((row, i) => [...row, rhs[i]]);
  for (let k = 0; k < n; k++) {
    let p = k;
    for (let i = k + 1; i < n; i++) if (Math.abs(a[i][k]) > Math.abs(a[p][k])) p = i;
    assert.ok(Math.abs(a[p][k]) > 1e-12, 'singular ridge system');
    [a[k], a[p]] = [a[p], a[k]];
    const pivot = a[k][k]; for (let j = k; j <= n; j++) a[k][j] /= pivot;
    for (let i = 0; i < n; i++) if (i !== k) {
      const factor = a[i][k]; for (let j = k; j <= n; j++) a[i][j] -= factor * a[k][j];
    }
  }
  return a.map(row => row[n]);
}
function fit(input) {
  const X = cal.map(input), n = X.length;
  for (const x of X) assert.equal(x.length, X[0].length);
  const gram = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) =>
    dot(X[i], X[j]) + (i === j ? 4 : 0)));
  const targetMean = [0, 1, 2].map(j => mean(cal.map(r => r.telemetry[j])));
  const alpha = targetMean.map((m, j) => solve(gram, cal.map(r => r.telemetry[j] - m)));
  return row => {
    const x = input(row), similarity = X.map(v => dot(v, x));
    return alpha.map((a, j) => targetMean[j] + dot(a, similarity));
  };
}
const predict = Object.fromEntries(Object.entries(representations).map(([name, input]) => [name, fit(input)]));
let maxArchiveError = 0;
for (const row of archive.validationRows) for (let j = 0; j < 3; j++)
  maxArchiveError = Math.max(maxArchiveError, Math.abs(predict.mbon(row)[j] - row.reconstructed[j]));
assert.ok(maxArchiveError < 1e-9, `archived MBON ridge mismatch: ${maxArchiveError}`);
if (process.argv[2] === 'preflight') {
  console.log(JSON.stringify({ protocolSha256, archiveRows: [cal.length, archive.validationRows.length],
    kcCount: kcIds.length, mbonCount: mbonIds.length, kcRadius, maxArchiveError, streams: configs }));
  process.exit(0);
}
if (process.argv[2]) throw Error('Use no argument or preflight');

function cases(seed) {
  const random = rng(seed), octants = Array.from({ length: 8 }, (_, o) => Array(12).fill(o)).flat();
  for (let i = octants.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1)); [octants[i], octants[j]] = [octants[j], octants[i]];
  }
  return octants.map((octant, index) => ({ index, octant,
    telemetry: [2, 1, 0].map(bit => ((octant >> bit) & 1) / 2 + random() / 2) }));
}
function probe(net, spec, config) {
  const streamKey = config.neuralSeed * 1e7 + 1e6 + spec.index * 1000 + 1;
  net.resetState(); net.rand = rng(streamKey);
  const drives = spec.telemetry.map(v => 40 + 110 * v);
  for (let j = 0; j < 3; j++) net.setDrive(net.laneVpn[j].slice(2), drives[j]);
  for (let t = 0; t < 240; t++) { if (t === 50) net.resetCounts(); net.step(); }
  for (let j = 0; j < 3; j++) net.setDrive(net.laneVpn[j].slice(2), 0);
  return { ...spec, split: 'validation', neuralSeed: config.neuralSeed,
    caseSeed: config.caseSeed, streamKey, drives, windowMs: [50, 240],
    reset: 'resetState before drive; resetCounts at 50 ms',
    vpn: net.laneVpn.map(ids => ids.slice(2).reduce((s, id) => s + net.count[id], 0)),
    kcActive: [...net.idx.kc].filter(id => net.count[id] > 0).map(id => [id, net.count[id]]),
    mbon: [...net.idx.mbon].map(id => ({ id, group: net.groupOf[id], count: net.count[id] })),
    pooled: net.groupMbon.map(ids => ids.reduce((s, id) => s + net.count[id], 0) / ids.length) };
}
const allKeys = configs.flatMap(c => cases(c.caseSeed).map(r =>
  (c.neuralSeed * 1e7 + 1e6 + r.index * 1000 + 1) >>> 0));
assert.equal(new Set(allKeys).size, 192, 'new effective RNG key collision');
const streams = [];
for (const config of configs) {
  const net = new Network(sub, wiring(sub, 'real', config.neuralSeed), { seed: config.neuralSeed });
  const specs = cases(config.caseSeed);
  const rows = specs.map(spec => probe(net, spec, config));
  assert.deepEqual(probe(net, specs[0], config), rows[0], 'keyed probe mismatch');
  assert.ok(net.m.every(v => v === 1), 'neural weights changed');
  streams.push({ config, rows, keyedRepeatPassed: true, weightsUnchanged: true });
  console.log(JSON.stringify({ completedNeuralSeed: config.neuralSeed, probes: rows.length }));
}
function pearson(a, b) {
  const am = mean(a), bm = mean(b);
  const cov = a.reduce((s, v, i) => s + (v - am) * (b[i] - bm), 0);
  const aa = a.reduce((s, v) => s + (v - am) ** 2, 0);
  const bb = b.reduce((s, v) => s + (v - bm) ** 2, 0);
  return cov / Math.sqrt(aa * bb);
}
const metrics = streams.map(stream => {
  for (let o = 0; o < 8; o++) assert.equal(stream.rows.filter(r => r.octant === o).length, 12);
  for (const row of stream.rows) {
    assert.equal(row.kcActive.length, row.kcActive.filter(([id, count]) =>
      kcPosition.has(id) && Number.isInteger(count) && count > 0).length);
    assert.equal(row.mbon.length, 42);
    assert.ok(row.vpn.every(v => Number.isInteger(v) && v >= 0));
    row.predictions = Object.fromEntries(Object.entries(predict).map(([name, fn]) => [name, fn(row)]));
  }
  const byModel = Object.fromEntries(Object.keys(predict).map(name => [name, [0, 1, 2].map(j => {
    const y = stream.rows.map(r => r.telemetry[j]), p = stream.rows.map(r => r.predictions[name][j]);
    const yMean = mean(y), sst = y.reduce((s, v) => s + (v - yMean) ** 2, 0);
    const sse = y.reduce((s, v, i) => s + (v - p[i]) ** 2, 0);
    return { r2: 1 - sse / sst, nmae: mean(y.map((v, i) => Math.abs(v - p[i]))),
      bias: mean(y.map((v, i) => p[i] - v)) };
  })]));
  const vpnDriveCorrelation = [0, 1, 2].map(j => pearson(
    stream.rows.map(r => r.drives[j]), stream.rows.map(r => r.vpn[j])));
  const active = stream.rows.map(r => r.kcActive.length);
  const sorted = [...active].sort((a, b) => a - b);
  const kcMbonCorrelation = pearson(active,
    stream.rows.map(r => r.mbon.reduce((s, m) => s + m.count, 0)));
  return { neuralSeed: stream.config.neuralSeed, byModel, vpnDriveCorrelation,
    kcActiveMean: mean(active), kcActiveMedian: (sorted[47] + sorted[48]) / 2,
    kcActiveNonzero: active.filter(x => x > 0).length,
    kcMbonCountCorrelation: kcMbonCorrelation };
});
const recording = streams.every(s => s.keyedRepeatPassed && s.weightsUnchanged &&
  s.rows.length === 96 && s.rows.every(r => r.windowMs[0] === 50 && r.windowMs[1] === 240));
const upstream = metrics.every(s => s.vpnDriveCorrelation.every(x => x >= .8) &&
  s.byModel.vpn.every(x => x.r2 >= .7));
const kcAbsolute = metrics.every(s => s.byModel.kc.every(x => x.r2 >= .4 && x.nmae <= .2));
const kcOverMbon = metrics.every(s => [1, 2].every(j =>
  s.byModel.kc[j].r2 - s.byModel.mbon[j].r2 >= .1) &&
  s.byModel.kc[0].r2 - s.byModel.mbon[0].r2 >= -.05);
const gates = { recording, upstream, kcAbsolute, kcOverMbon,
  allPassed: recording && upstream && kcAbsolute && kcOverMbon };
const artifact = { protocolSha256, previousArtifact: 'practical_transfer_redesign_initial_sensory.json',
  calibration: { neuralSeed: 19101, caseSeed: 20261109115, rows: cal },
  scalers: { kcCenter: Array.from(kcCenter), kcRadius, mbon: archive.scaler },
  ridgePenalty: 4, maxArchiveError, streams, metrics, gates };
const file = await open(output, 'wx');
try { await file.writeFile(JSON.stringify(artifact) + '\n'); } finally { await file.close(); }
console.log(JSON.stringify({ output: output.pathname, metrics, gates }));
