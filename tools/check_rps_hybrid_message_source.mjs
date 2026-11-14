// Locked source-only one-bit hybrid RPS message check; never stimulates or trains Fly B.
// node tools/check_rps_hybrid_message_source.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { rng } from '../js/sim.js';

const output = new URL('../data/rps_hybrid_message_source.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite hybrid message-source result');
const read = path => JSON.parse(readFileSync(new URL(path, import.meta.url)));
const saved = read('../data/rps_contextual_bandit.json');
const source = read('../data/rps_teacher_source_information.json');
if (!saved.allValid || saved.passed || source.passed ||
  JSON.stringify(saved.design.seeds) !== JSON.stringify([711, 712, 713, 714]) ||
  JSON.stringify(source.design.heldoutTeacherSeeds) !== JSON.stringify([712, 713, 714]) ||
  source.probes.length !== 108) throw Error('Saved source or controller protocol differs');

const { featureMean, featureStd } = saved.design;
const teacher = seed => {
  const student = saved.students.find(x => x.seed === seed);
  if (!student || !Object.values(student.validity).every(Boolean)) throw Error(`Invalid teacher ${seed}`);
  const trained = student.runs.find(x => x.arm === 'contextual_bandit');
  const untrained = student.runs.find(x => x.arm === 'contextual_bandit_no_learning');
  if (!trained || !untrained || trained.kcMbonChangedWeights || untrained.kcMbonChangedWeights ||
    untrained.finalController.W.flat().some(x => x !== 0) ||
    untrained.finalController.b.some(x => x !== 0)) throw Error(`Invalid controller states ${seed}`);
  return { trained, untrained };
};
const scores = (rates, params) => {
  const f = rates.map((x, j) => (Math.log1p(x) - featureMean[j]) / featureStd[j]);
  return params.W.map((row, a) => row.reduce((sum, w, j) => sum + w * f[j], params.b[a]));
};
const spread = q => Math.max(...q) - Math.min(...q);
const counts = rows => [0, 1].map(bit => rows.filter(x => x.bit === bit).length);
const contingency = (rows, key) => [0, 1].map(bit =>
  [0, 1, 2].map(action => rows.filter(x => x.bit === bit && x[key] === action).length));
const purityPass = rows => ['teacherAction', 'winningAction'].every(key =>
  contingency(rows, key).every(group => Math.max(...group) / group.reduce((a, b) => a + b, 0) <= 0.75));
const information = rows => {
  const n = rows.length;
  let bits = 0;
  for (const state of [0, 1, 2]) for (const bit of [0, 1]) {
    const joint = rows.filter(x => x.state === state && x.bit === bit).length;
    if (!joint) continue;
    const stateN = rows.filter(x => x.state === state).length;
    const bitN = rows.filter(x => x.bit === bit).length;
    bits += joint / n * Math.log2(joint * n / (stateN * bitN));
  }
  return bits;
};
const calibrationSeed = 711, heldoutSeeds = [712, 713, 714];
const calibrationRun = teacher(calibrationSeed).trained;
const calibrationEvents = calibrationRun.after.events.filter(e => e.previous !== null);
if (calibrationEvents.length !== 69) throw Error('Invalid calibration trial count');
const calibrationRows = calibrationEvents.map(e => {
  const q = scores(e.rates, calibrationRun.finalController);
  if (q.some((x, a) => Math.abs(x - e.scores[a]) > 1e-12) ||
    q.indexOf(Math.max(...q)) !== e.action || e.opponent !== (e.previous + 1) % 3)
    throw Error('Calibration score, action, or opponent mismatch');
  return { seed: calibrationSeed, state: e.previous, rates: e.rates,
    q, spread: spread(q), teacherAction: e.action, winningAction: (e.opponent + 1) % 3 };
});
const ordered = calibrationRows.map(x => x.spread).toSorted((a, b) => a - b);
const threshold = ordered[34];
calibrationRows.forEach(x => { x.bit = Number(x.spread > threshold); });
const calibrationBitCounts = counts(calibrationRows);
const calibrationGates = { nonSaturated: calibrationBitCounts.every(n => n / 69 >= 0.2),
  notEncodedAction: purityPass(calibrationRows) };
const result = { command: 'node tools/check_rps_hybrid_message_source.mjs',
  design: { message: 'one bit: max(Q_rock,Q_paper,Q_scissors)-min(Q_rock,Q_paper,Q_scissors) > calibration median',
    source: 'saved trained contextual-bandit response to real 240ms action-group MBON rates',
    calibrationSeed, heldoutSeeds, probesPerTeacher: 36,
    firstRoundExcluded: true, shuffledPermutations: 1000, shuffleSeed: 20261104,
    calibrationGate: 'each bit >=20% and no bit bin >75% one selected or winning action',
    heldoutGates: 'each trained bit 20-80%; untrained bit1 <=5%; no bin >75% selected or winning action; pooled I(bit;state) >=0.10 bit and >shuffle95; each teacher I>=0.03 bit',
    conditionalNextMoveInformationBits: 0 },
  threshold, calibration: { counts: calibrationBitCounts,
    stateByBit: [0, 1, 2].map(state => counts(calibrationRows.filter(x => x.state === state))),
    selectedActionByBit: contingency(calibrationRows, 'teacherAction'),
    winningActionByBit: contingency(calibrationRows, 'winningAction'),
    gates: calibrationGates, rows: calibrationRows },
  heldout: null, passed: false };

if (Object.values(calibrationGates).every(Boolean)) {
  const heldoutRows = [];
  for (const seed of heldoutSeeds) {
    const { trained, untrained } = teacher(seed);
    const probes = source.probes.filter(x => x.seed === seed);
    if (probes.length !== 36 || [0, 1, 2].some(s => probes.filter(x => x.state === s).length !== 12))
      throw Error(`Invalid held-out probe schedule ${seed}`);
    for (const probe of probes) {
      if (JSON.stringify(probe.trained.rates) !== JSON.stringify(probe.untrained.rates))
        throw Error('Matched neural source changed');
      const trainedQ = scores(probe.trained.rates, trained.finalController);
      const untrainedQ = scores(probe.untrained.rates, untrained.finalController);
      const teacherAction = trainedQ.indexOf(Math.max(...trainedQ));
      if (teacherAction !== probe.trained.action || untrainedQ.some(x => x !== 0))
        throw Error('Saved frozen action or untrained control mismatch');
      const trainedSpread = spread(trainedQ), untrainedSpread = spread(untrainedQ);
      heldoutRows.push({ seed, state: probe.state, repeat: probe.repeat,
        rates: probe.trained.rates, q: trainedQ, spread: trainedSpread,
        bit: Number(trainedSpread > threshold), untrainedBit: Number(untrainedSpread > threshold),
        teacherAction, winningAction: (probe.state + 2) % 3 });
    }
  }
  const perSeed = heldoutSeeds.map(seed => {
    const rows = heldoutRows.filter(x => x.seed === seed);
    return { seed, bitCounts: counts(rows), untrainedBit1: rows.filter(x => x.untrainedBit === 1).length,
      stateByBit: [0, 1, 2].map(state => counts(rows.filter(x => x.state === state))),
      selectedActionByBit: contingency(rows, 'teacherAction'),
      winningActionByBit: contingency(rows, 'winningAction'),
      stateInformationBits: information(rows), noEncodedAction: purityPass(rows) };
  });
  const shuffled = [], random = rng(20261104);
  for (let rep = 0; rep < 1000; rep++) {
    const permuted = [];
    for (const seed of heldoutSeeds) {
      const rows = heldoutRows.filter(x => x.seed === seed);
      const bits = rows.map(x => x.bit);
      for (let i = bits.length - 1; i > 0; i--) {
        const j = Math.floor(random() * (i + 1));
        [bits[i], bits[j]] = [bits[j], bits[i]];
      }
      rows.forEach((row, i) => permuted.push({ state: row.state, bit: bits[i] }));
    }
    shuffled.push(information(permuted));
  }
  const shuffled95 = shuffled.toSorted((a, b) => a - b)[949];
  const pooledStateInformationBits = information(heldoutRows);
  const gates = { trainingDependent: perSeed.every(x =>
      x.bitCounts[1] / 36 >= 0.2 && x.untrainedBit1 / 36 <= 0.05),
    nonSaturated: perSeed.every(x => x.bitCounts.every(n => n / 36 >= 0.2)),
    notEncodedAction: perSeed.every(x => x.noEncodedAction),
    stateInformation: pooledStateInformationBits >= 0.10 &&
      pooledStateInformationBits > shuffled95 &&
      perSeed.every(x => x.stateInformationBits >= 0.03) };
  result.heldout = { perSeed, pooledStateInformationBits,
    untrainedStateInformationBits: information(heldoutRows.map(x => ({ state: x.state, bit: x.untrainedBit }))),
    shuffledMean: shuffled.reduce((a, b) => a + b, 0) / shuffled.length,
    shuffled95, gates, rows: heldoutRows };
  result.passed = Object.values(gates).every(Boolean);
}
writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({ threshold, calibration: {
  counts: result.calibration.counts, stateByBit: result.calibration.stateByBit,
  selectedActionByBit: result.calibration.selectedActionByBit,
  winningActionByBit: result.calibration.winningActionByBit, gates: calibrationGates },
  heldout: result.heldout && { perSeed: result.heldout.perSeed,
    pooledStateInformationBits: result.heldout.pooledStateInformationBits,
    shuffledMean: result.heldout.shuffledMean, shuffled95: result.heldout.shuffled95,
    gates: result.heldout.gates }, passed: result.passed }, null, 2));
