// One locked, read-only MBON-by-DN interaction capacity test on archived recordings.
// node tools/analyze_rps_hidden_strategy_interaction.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';

const output = new URL('../data/rps_hidden_strategy_interaction.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite interaction result');
const read = name => JSON.parse(readFileSync(new URL(`../data/${name}.json`, import.meta.url)));
const previous = read('rps_hidden_strategy_offline_information');
const archive = read('rps_hidden_strategy_continuous_receiver');
const pilot = read('rps_hidden_strategy_teaching_pilot');
const dnUse = read('rps_hidden_strategy_dn_use');
const forced = read('rps_hidden_strategy_frozen_forced_bits');
if (!archive.passed || previous.gates.strategy !== true || previous.gates.directAction !== false ||
    pilot.passed || !pilot.allValid || forced.passed ||
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
    const bit = mode === 'actual' ? e.sourceBit :
      mode === 'shuffled' ? e.shuffledBit : mode === 'force1' ? 1 : 0;
    const physical = receiver.get(`${e.receiverSeed}|${e.cue}|${e.repeat}|bit${bit}`);
    if (!physical || physical.features.length !== 5 || physical.dnCounts.length !== 2 ||
        !physical.features.slice(3).every((v, j) => approx(v, physical.dnCounts[j] / dnScale)) ||
        physical.dnCounts[bit] <= physical.dnCounts[1 - bit])
      throw Error(`Missing or malformed physical receiver row ${e.receiverSeed}/${e.cue}/${e.repeat}/${bit}`);
    return { ...e, deliveredBit: bit, dnCounts: physical.dnCounts,
      features: [...physical.features] };
  });
}
function vector(row, dimension) {
  const base = row.features.slice(0, Math.min(dimension, 5));
  if (dimension !== 11) return base;
  const m = base.slice(0, 3), d = base.slice(3, 5);
  return [...base, ...d.flatMap(value => m.map(rate => rate * value))];
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
    const x = [1, ...vector(row, dimension)];
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
  const x = [1, ...vector(row, dimension)];
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
function evaluateAction(train, test, dimension) {
  const model = fit(train, dimension, 'correctAction', 3);
  const predicted = test.map(row => classify(model, row, dimension));
  const confusion = Array.from({ length: 3 }, () => [0, 0, 0]);
  test.forEach((row, i) => { confusion[row.correctAction][predicted[i]]++; });
  return { model, action: metrics(test, predicted, 'correctAction'), confusion,
    predicted };
}
function same(a, b) { return JSON.stringify(a) === JSON.stringify(b); }
function bitSwap(heldout, model) {
  const zero = withFeatures(heldout, 'force0').map(row => classify(model, row, 11));
  const one = withFeatures(heldout, 'force1').map(row => classify(model, row, 11));
  const summarize = subset => {
    let switches = 0, teacherCorrect = 0, oppositeCorrect = 0, helpful = 0, harmful = 0;
    for (const i of subset) {
      const e = heldout[i];
      const teacher = e.sourceBit ? one[i] : zero[i];
      const opposite = e.sourceBit ? zero[i] : one[i];
      switches += Number(zero[i] !== one[i]);
      teacherCorrect += Number(teacher === e.correctAction);
      oppositeCorrect += Number(opposite === e.correctAction);
      helpful += Number(teacher === e.correctAction && opposite !== e.correctAction);
      harmful += Number(teacher !== e.correctAction && opposite === e.correctAction);
    }
    return { count: subset.length, switches, teacherCorrect, oppositeCorrect, helpful, harmful };
  };
  const all = heldout.map((_, i) => i);
  return { overall: summarize(all), byCue: [0, 1, 2].map(cue =>
    ({ cue, ...summarize(all.filter(i => heldout[i].cue === cue)) })),
    byStrategy: [0, 1].map(strategy => ({ strategy,
      ...summarize(all.filter(i => heldout[i].strategy === strategy)) })),
    byCueStrategy: [0, 1, 2].flatMap(cue => [0, 1].map(strategy => ({ cue, strategy,
      ...summarize(all.filter(i => heldout[i].cue === cue && heldout[i].strategy === strategy)) }))) };
}

const folds = receiverSeeds.map((calibrationReceiver, foldIndex) => {
  const heldoutReceiver = receiverSeeds[1 - foldIndex];
  const calibration = events(calibrationTeachers, calibrationReceiver, foldIndex, 0);
  const heldout = events(heldoutTeachers, heldoutReceiver, foldIndex, 1);
  if (calibration.length !== 276 || heldout.length !== 345 || calibrationReceiver === heldoutReceiver ||
      calibrationTeachers.some(t => heldoutTeachers.includes(t))) throw Error('Partition mismatch');
  const actualTrain = withFeatures(calibration, 'actual');
  const actualTest = withFeatures(heldout, 'actual');
  const mbon = evaluateAction(actualTrain, actualTest, 3);
  const linear = evaluateAction(actualTrain, actualTest, 5);
  const prior = previous.folds[foldIndex];
  if (prior.calibrationReceiver !== calibrationReceiver || prior.heldoutReceiver !== heldoutReceiver ||
      !same(mbon.model, prior.mbon.models.action) ||
      !same(linear.model, prior.five.models.action) ||
      !same(mbon.predicted, prior.mbon.predictions.map(x => x.predictedAction)) ||
      !same(linear.predicted, prior.five.predictions.map(x => x.predictedAction)) ||
      !same(mbon.action, prior.mbon.action) || !same(linear.action, prior.five.action))
    throw Error(`Prior linear baseline failed exact reconstruction in fold ${foldIndex}`);
  // The interaction readout is fitted only after both saved baselines reproduce exactly.
  const interaction = evaluateAction(actualTrain, actualTest, 11);
  const shuffled = evaluateAction(withFeatures(calibration, 'shuffled'),
    withFeatures(heldout, 'shuffled'), 11);
  const untrained = evaluateAction(withFeatures(calibration, 'untrained'),
    withFeatures(heldout, 'untrained'), 11);
  const x = interaction.action, metric = model => model.action.balanced;
  const gate = x.raw >= 0.6 && x.balanced >= 0.6 &&
    [linear, mbon, shuffled, untrained].every(other => x.balanced - metric(other) >= 0.1) &&
    x.byCue.every(c => c.balanced >= 0.5);
  return { calibrationReceiver, heldoutReceiver, calibrationTeachers, heldoutTeachers,
    calibrationEvents: calibration.length, heldoutEvents: heldout.length,
    previousBaselineReproduced: true, mbon, linear, interaction, shuffled, untrained,
    bitSwap: bitSwap(heldout, interaction.model), gate,
    heldoutRows: actualTest.map((row, i) => ({ teacherSeed: row.teacherSeed,
      receiverSeed: row.receiverSeed, sourceSequenceSeed: row.sourceSequenceSeed,
      sourceRound: row.sourceRound, cue: row.cue, repeat: row.repeat,
      sourceBit: row.sourceBit, strategy: row.strategy, correctAction: row.correctAction,
      features: row.features, dnCounts: row.dnCounts,
      predictions: { mbon: mbon.predicted[i], linear: linear.predicted[i],
        interaction: interaction.predicted[i], shuffled: shuffled.predicted[i],
        untrained: untrained.predicted[i] } })) };
});
const gates = { receiver811To812: folds[0].gate, receiver812To811: folds[1].gate,
  all: folds.every(f => f.gate) };
const result = { command: 'node tools/analyze_rps_hidden_strategy_interaction.mjs',
  status: 'complete', priorTeachingPilotPassed: pilot.passed,
  priorForcedBitPassed: forced.passed, priorDirectLinearGatePassed: previous.gates.directAction,
  design: { sources: ['rps_hidden_strategy_continuous_receiver',
      'rps_hidden_strategy_offline_information'], calibrationTeachers,
    heldoutTeachers, receiverFolds: [[811, 812], [812, 811]],
    pairing: 'teacher rows within cue sorted by sequenceIndex,round; receiver repeats 6-28 ordinally',
    features: 'm0,m1,m2,d0,d1,m0*d0,m1*d0,m2*d0,m0*d1,m1*d1,m2*d1; d=measured counts/7.388888888888889',
    classifier: 'three-class mapping-balanced weighted one-hot ridge, L2=1 on coefficients, intercept unpenalized, exact normal equations, lowest-index argmax tie',
    controls: 'same interaction on prior cue-stratified shuffled bits and zero-Q bit0 physical branches',
    gates: 'both folds: raw and mapping-balanced action >=60%; >=10pp balanced over original linear, MBON-only, shuffled and untrained interaction; every cue balanced >=50%' },
  folds, gates, limitations: [
    'Supervised offline decoder is not installed in Fly B and cannot establish reward-based teaching.',
    'Previously characterized teacher and receiver recordings are paired offline, not jointly simulated.',
    'Physical receiver branches repeat across teacher events; effective neural sample size is limited.',
    'Receiver folds share teacher partitions; they are not independent confirmation experiments.',
    'Pre-existing MBON cue decoder recognized only 17/23 no-message cue-0 probes for receiver 811.' ] };
writeFileSync(output, `${JSON.stringify(result)}\n`);
console.log(JSON.stringify({ gates, folds: folds.map(f => ({
  calibrationReceiver: f.calibrationReceiver, heldoutReceiver: f.heldoutReceiver,
  accuracy: Object.fromEntries(['mbon','linear','interaction','shuffled','untrained'].map(k =>
    [k, { raw: f[k].action.raw, balanced: f[k].action.balanced,
      byCue: f[k].action.byCue.map(c => c.balanced) }])),
  bitSwap: f.bitSwap.overall })) }, null, 2));
