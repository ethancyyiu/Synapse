// Read-only, predeclared supervised information probe on archived neural recordings.
// node tools/analyze_rps_hidden_strategy_offline_information.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';

const output = new URL('../data/rps_hidden_strategy_offline_information.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite offline information result');
const read = name => JSON.parse(readFileSync(new URL(`../data/${name}.json`, import.meta.url)));
const archive = read('rps_hidden_strategy_continuous_receiver');
const pilot = read('rps_hidden_strategy_teaching_pilot');
const dnUse = read('rps_hidden_strategy_dn_use');
const forced = read('rps_hidden_strategy_frozen_forced_bits');
if (!archive.passed || pilot.passed || !pilot.allValid || forced.passed ||
    dnUse.originalPilotPassed || archive.receiverRows.length !== 522 ||
    archive.sourceRows.length !== 621)
  throw Error('Archived study status or row count changed');

const dnScale = 7.388888888888889;
const calibrationTeachers = [781, 784, 785, 791];
const heldoutTeachers = [782, 783, 786, 792, 793];
const receiverSeeds = [811, 812];
const l2 = 1;
const approx = (a, b) => Math.abs(a - b) < 1e-10;
const receiver = new Map();
for (const row of archive.receiverRows) {
  const key = `${row.receiverSeed}|${row.cue}|${row.repeat}|${row.condition}`;
  if (receiver.has(key)) throw Error(`Repeated receiver key ${key}`);
  receiver.set(key, row);
}
const source = new Map();
for (const teacher of [...calibrationTeachers, ...heldoutTeachers])
  for (const cue of [0, 1, 2]) {
    const rows = archive.sourceRows.filter(x => x.teacherSeed === teacher && x.cue === cue)
      .sort((a, b) => a.sequenceIndex - b.sequenceIndex || a.round - b.round);
    if (rows.length !== 23 || rows.some(x => x.bit !== Number(x.scores[x.cue] >
        Math.max(...x.scores.filter((_, j) => j !== x.cue))) ||
        x.offset !== (x.mapping === 'A' ? 1 : 2)))
      throw Error(`Incomplete or inconsistent teacher source ${teacher}/${cue}`);
    source.set(`${teacher}|${cue}`, rows);
  }

function mulberry32(seed) {
  let x = seed >>> 0;
  return () => {
    x = (x + 0x6D2B79F5) >>> 0;
    let t = Math.imul(x ^ (x >>> 15), 1 | x);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function events(teachers, receiverSeed, fold, partition) {
  const rows = [];
  for (const teacherSeed of teachers)
    for (const cue of [0, 1, 2])
      source.get(`${teacherSeed}|${cue}`).forEach((s, ordinal) => rows.push({
        teacherSeed, receiverSeed, cue, repeat: 6 + ordinal,
        sourceSequenceSeed: s.sequenceSeed, sourceRound: s.round,
        sourceBit: s.bit, strategy: s.mapping === 'A' ? 0 : 1,
        correctAction: (cue + s.offset + 1) % 3
      }));
  // One fixed, cue-stratified permutation within this fold's calibration/test partition.
  const random = mulberry32(2026110505 + 100 * fold + partition);
  for (const cue of [0, 1, 2]) {
    const indexes = rows.map((r, i) => r.cue === cue ? i : -1).filter(i => i >= 0);
    const bits = indexes.map(i => rows[i].sourceBit);
    for (let j = bits.length - 1; j > 0; j--) {
      const k = Math.floor(random() * (j + 1));
      [bits[j], bits[k]] = [bits[k], bits[j]];
    }
    indexes.forEach((i, j) => { rows[i].shuffledBit = bits[j]; });
  }
  return rows;
}
function withFeatures(rows, mode) {
  return rows.map(e => {
    const bit = mode === 'actual' ? e.sourceBit : mode === 'shuffled' ? e.shuffledBit : 0;
    const physical = receiver.get(`${e.receiverSeed}|${e.cue}|${e.repeat}|bit${bit}`);
    if (!physical || physical.features.length !== 5 || physical.dnCounts.length !== 2 ||
        !physical.features.slice(3).every((v, j) => approx(v, physical.dnCounts[j] / dnScale)) ||
        physical.dnCounts[bit] <= physical.dnCounts[1 - bit])
      throw Error(`Missing or malformed physical receiver row ${e.receiverSeed}/${e.cue}/${e.repeat}/${bit}`);
    return { ...e, deliveredBit: bit, dnCounts: physical.dnCounts,
      features: [...physical.features] };
  });
}
function solve(matrix, rhs) {
  const n = rhs.length;
  const a = matrix.map((row, i) => [...row, rhs[i]]);
  for (let k = 0; k < n; k++) {
    let pivot = k;
    for (let i = k + 1; i < n; i++) if (Math.abs(a[i][k]) > Math.abs(a[pivot][k])) pivot = i;
    if (Math.abs(a[pivot][k]) < 1e-12) throw Error('Singular fixed ridge system');
    [a[k], a[pivot]] = [a[pivot], a[k]];
    const scale = a[k][k];
    for (let j = k; j <= n; j++) a[k][j] /= scale;
    for (let i = 0; i < n; i++) if (i !== k) {
      const factor = a[i][k];
      for (let j = k; j <= n; j++) a[i][j] -= factor * a[k][j];
    }
  }
  return a.map(row => row[n]);
}
function fit(rows, dimension, outcome, classes) {
  const d = dimension + 1;
  const gram = Array.from({ length: d }, () => Array(d).fill(0));
  const rhs = Array.from({ length: classes }, () => Array(d).fill(0));
  const byStrategy = [0, 1].map(s => rows.filter(r => r.strategy === s).length);
  if (byStrategy.some(n => !n)) throw Error('Both mappings required in calibration');
  for (const row of rows) {
    const x = [1, ...row.features.slice(0, dimension)];
    const weight = rows.length / (2 * byStrategy[row.strategy]);
    for (let i = 0; i < d; i++) {
      for (let j = 0; j < d; j++) gram[i][j] += weight * x[i] * x[j];
      rhs[row[outcome]][i] += weight * x[i];
    }
  }
  for (let i = 1; i < d; i++) gram[i][i] += l2;
  return rhs.map(target => solve(gram, target));
}
function classify(model, row, dimension) {
  const x = [1, ...row.features.slice(0, dimension)];
  const scores = model.map(w => w.reduce((sum, v, j) => sum + v * x[j], 0));
  return scores.indexOf(Math.max(...scores));
}
const mean = x => x.reduce((a, b) => a + b, 0) / x.length;
function metrics(rows, predictions, label) {
  const indexed = rows.map((row, i) => ({ row, ok: Number(predictions[i] === row[label]) }));
  const accuracy = items => mean(items.map(x => x.ok));
  const strategyBalanced = items => mean([0, 1].map(s => accuracy(items.filter(x => x.row.strategy === s))));
  return { count: rows.length, raw: accuracy(indexed), balanced: strategyBalanced(indexed),
    byCue: [0, 1, 2].map(cue => {
      const xs = indexed.filter(x => x.row.cue === cue);
      return { cue, raw: accuracy(xs), balanced: strategyBalanced(xs) };
    }),
    byTeacher: [...new Set(rows.map(r => r.teacherSeed))].map(teacherSeed => {
      const xs = indexed.filter(x => x.row.teacherSeed === teacherSeed);
      return { teacherSeed, mapping: xs[0].row.strategy === 0 ? 'A' : 'B',
        accuracy: accuracy(xs), count: xs.length };
    }) };
}
function evaluate(train, test, dimension) {
  const strategyModel = fit(train, dimension, 'strategy', 2);
  const actionModel = fit(train, dimension, 'correctAction', 3);
  const strategyPredictions = test.map(r => classify(strategyModel, r, dimension));
  const actionPredictions = test.map(r => classify(actionModel, r, dimension));
  return { models: { strategy: strategyModel, action: actionModel },
    strategy: metrics(test, strategyPredictions, 'strategy'),
    action: metrics(test, actionPredictions, 'correctAction'),
    predictions: test.map((row, i) => ({ teacherSeed: row.teacherSeed,
      receiverSeed: row.receiverSeed, cue: row.cue, repeat: row.repeat,
      sourceSequenceSeed: row.sourceSequenceSeed, sourceRound: row.sourceRound,
      sourceBit: row.sourceBit, deliveredBit: row.deliveredBit,
      strategy: row.strategy, correctAction: row.correctAction,
      features: row.features, dnCounts: row.dnCounts,
      predictedStrategy: strategyPredictions[i], predictedAction: actionPredictions[i] })) };
}

const folds = receiverSeeds.map((calibrationReceiver, foldIndex) => {
  const heldoutReceiver = receiverSeeds[1 - foldIndex];
  const calibration = events(calibrationTeachers, calibrationReceiver, foldIndex, 0);
  const heldout = events(heldoutTeachers, heldoutReceiver, foldIndex, 1);
  if (calibration.length !== 276 || heldout.length !== 345) throw Error('Partition size mismatch');
  const actualTrain = withFeatures(calibration, 'actual');
  const actualTest = withFeatures(heldout, 'actual');
  const shuffledTrain = withFeatures(calibration, 'shuffled');
  const shuffledTest = withFeatures(heldout, 'shuffled');
  const untrainedTrain = withFeatures(calibration, 'untrained');
  const untrainedTest = withFeatures(heldout, 'untrained');
  const mbon = evaluate(actualTrain, actualTest, 3);
  const five = evaluate(actualTrain, actualTest, 5);
  const shuffled = evaluate(shuffledTrain, shuffledTest, 5);
  const untrained = evaluate(untrainedTrain, untrainedTest, 5);
  const cueOnly = { strategy: metrics(actualTest, actualTest.map(() => 1), 'strategy'),
    action: metrics(actualTest, actualTest.map(r => r.cue), 'correctAction') };
  const thresholds = { strategy: five.strategy.balanced >= 0.8 &&
      five.strategy.balanced - mbon.strategy.balanced >= 0.2 &&
      five.strategy.balanced - shuffled.strategy.balanced >= 0.2 &&
      five.strategy.balanced - untrained.strategy.balanced >= 0.2 &&
      five.strategy.byCue.every(c => c.balanced >= 0.75),
    action: five.action.balanced >= 0.6 &&
      five.action.balanced - mbon.action.balanced >= 0.1 &&
      five.action.balanced - shuffled.action.balanced >= 0.1 &&
      five.action.balanced - untrained.action.balanced >= 0.1 &&
      five.action.byCue.every(c => c.balanced >= 0.5) };
  return { calibrationReceiver, heldoutReceiver, calibrationTeachers, heldoutTeachers,
    calibrationEvents: calibration.length, heldoutEvents: heldout.length,
    bitCounts: { calibration: [0, 1].map(bit => calibration.filter(x => x.sourceBit === bit).length),
      heldout: [0, 1].map(bit => heldout.filter(x => x.sourceBit === bit).length) },
    cueOnly, mbon, five, shuffled, untrained, gates: thresholds };
});
const gates = { strategy: folds.every(f => f.gates.strategy),
  directAction: folds.every(f => f.gates.action) };
const result = { command: 'node tools/analyze_rps_hidden_strategy_offline_information.mjs',
  status: 'complete', sources: [
    'data/rps_hidden_strategy_continuous_receiver.json',
    'data/rps_hidden_strategy_teaching_pilot.json',
    'data/rps_hidden_strategy_dn_use.json',
    'data/rps_hidden_strategy_frozen_forced_bits.json' ],
  design: { primaryArchive: 'continuous_receiver only', calibrationTeachers,
    heldoutTeachers, receiverFolds: [[811, 812], [812, 811]],
    pairing: 'teacher rows sorted within cue by sequenceIndex,round to physical receiver repeats 6-28 in ordinal order',
    featureNormalization: 'three archived fixed standardized MBON features; measured DN counts / 7.388888888888889',
    classifier: 'class-balanced weighted ridge one-hot linear scores, intercept unpenalized, L2=1, argmax lowest-index tie',
    controls: 'cue-only majority B; zero-Q forced bit0; fixed cue-stratified within-partition bit permutation',
    gates: 'strategy balanced >=80%, +20pp over MBON/shuffled/untrained, each cue >=75%; direct action balanced >=60%, +10pp over MBON/shuffled/untrained, each cue >=50%; both receiver folds required' },
  priorTeachingPilotPassed: pilot.passed, priorForcedBitPassed: forced.passed,
  folds, gates, limitations: [
    'Source and receiver recordings are paired offline, not simultaneous teacher-student trials.',
    'Source teachers and receiver probes were previously characterized; this is not an untouched independent confirmation.',
    'Receiver branches are reused across teachers; fold-level receiver seed and teacher groups are disjoint, but effective physical sample size is limited.',
    'The two receiver folds share teacher partitions and are not statistically independent.',
    'Direct linear action scores have no cue-by-message interaction, which can limit representation of the cyclic RPS rule.',
    'Supervised offline labels never enter the simulated student and cannot establish reward-based teaching.' ] };
writeFileSync(output, `${JSON.stringify(result)}\n`);
console.log(JSON.stringify({ gates, folds: folds.map(f => ({
  calibrationReceiver: f.calibrationReceiver, heldoutReceiver: f.heldoutReceiver,
  metrics: Object.fromEntries(['cueOnly', 'mbon', 'five', 'shuffled', 'untrained'].map(k =>
    [k, { strategy: f[k].strategy.balanced, action: f[k].action.balanced,
      strategyByCue: f[k].strategy.byCue.map(c => c.balanced),
      actionByCue: f[k].action.byCue.map(c => c.balanced) }])), gates: f.gates })) }, null, 2));
