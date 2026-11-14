// One locked, saved-recording-only RPS hidden-strategy bit source test.
// node tools/check_rps_hidden_strategy_source.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { rng } from '../js/sim.js';
import { result } from '../js/rps.js';

const output = new URL('../data/rps_hidden_strategy_source.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite hidden-strategy source result');
const read = path => JSON.parse(readFileSync(new URL(path, import.meta.url)));
const original = read('../data/rps_independent_teachers.json');
const replication = read('../data/rps_strategy_b_replication.json');
const calibrationIds = [781, 784, 785, 791], heldoutIds = [782, 783, 786, 792, 793];
const all = [...original.teachers, ...replication.teachers];
const ids = [...calibrationIds, ...heldoutIds];
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const approx = (a, b) => Math.abs(a - b) <= 1e-10;
const bitFrom = (scores, cue) => Number(scores[cue] >
  Math.max(...scores.filter((_, action) => action !== cue)));
const counts = rows => [0, 1].map(bit => rows.filter(r => r.bit === bit).length);
if (original.passed || !original.allValid || !replication.passed || !replication.allValid ||
    !same(original.design.featureMean, replication.design.featureMean) ||
    !same(original.design.featureStd, replication.design.featureStd) ||
    !same(ids.slice().sort((a, b) => a - b), all.map(t => t.seed).sort((a, b) => a - b)))
  throw Error('Archived teacher split, original failure, or feature calibration changed');
const { featureMean, featureStd } = original.design;

const teacherRows = all.map(t => {
  if (!Object.values(t.validity).every(Boolean) || !t.frozen.controllerUnchanged ||
      !t.frozen.synapsesUnchanged || t.frozen.rows.length !== 72 ||
      t.frozen.byCue.some(x => x.trials !== 23) ||
      !t.initial.W.flat().every(x => x === 0) || !t.initial.b.every(x => x === 0))
    throw Error(`Archived teacher validity failed: ${t.seed}`);
  const rows = t.frozen.rows.filter(r => r.cue !== null).map(r => {
    const f = r.rates.map((v, j) => (Math.log1p(v) - featureMean[j]) / featureStd[j]);
    const q = t.final.W.map((w, a) => w.reduce((s, v, j) => s + v * f[j], t.final.b[a]));
    const greedy = q.indexOf(Math.max(...q));
    if (r.cue < 0 || r.cue > 2 || f.length !== 3 || q.length !== 3 ||
        !f.every((v, j) => approx(v, r.features[j])) ||
        !q.every((v, j) => approx(v, r.scores[j])) ||
        greedy !== r.action || r.reward !== result(r.action, r.opponent) ||
        r.opponent !== (r.cue + t.offset) % 3)
      throw Error(`Archived score, action, or reward mismatch: ${t.seed}`);
    return { seed: t.seed, mapping: t.mapping, sequenceSeed: r.sequenceSeed,
      round: r.round, cue: r.cue, rates: r.rates, features: f, scores: q,
      teacherAction: greedy, bit: bitFrom(q, r.cue), untrainedBit: bitFrom([0, 0, 0], r.cue) };
  });
  if (rows.length !== 69 || [0, 1, 2].some(cue => rows.filter(r => r.cue === cue).length !== 23))
    throw Error(`Archived cue coverage changed: ${t.seed}`);
  return { seed: t.seed, mapping: t.mapping, individualReliabilityPassed: t.passed, rows };
});
const calibration = teacherRows.filter(t => calibrationIds.includes(t.seed));
const heldout = teacherRows.filter(t => heldoutIds.includes(t.seed));
if (calibration.length !== 4 || heldout.length !== 5 ||
    calibration.find(t => t.seed === 784)?.individualReliabilityPassed !== false)
  throw Error('Calibration includes failed teacher 784 or split differs');
const byTeacher = t => ({ seed: t.seed, mapping: t.mapping,
  individualReliabilityPassed: t.individualReliabilityPassed,
  byCue: [0, 1, 2].map(cue => ({ cue,
    bits: counts(t.rows.filter(r => r.cue === cue)),
    untrainedBits: counts(t.rows.filter(r => r.cue === cue).map(r => ({ ...r, bit: r.untrainedBit }))) })) });

// Equal mapping priors; B calibration teachers are pooled, not given triple prior weight.
const lookup = [0, 1, 2].map(cue => [0, 1].map(bit => {
  const frequency = mapping => {
    const xs = calibration.flatMap(t => t.mapping === mapping ? t.rows.filter(r => r.cue === cue) : []);
    return xs.filter(r => r.bit === bit).length / xs.length;
  };
  return frequency('B') > frequency('A') ? 'B' : 'A';
}));
const rows = heldout.flatMap(t => t.rows);
const balanced = xs => {
  const byMapping = ['A', 'B'].map(mapping => {
    const subset = xs.filter(r => r.mapping === mapping);
    return { mapping, n: subset.length,
      correct: subset.filter(r => lookup[r.cue][r.bit] === mapping).length };
  });
  return { byMapping, accuracy: byMapping.reduce((s, x) => s + x.correct / x.n, 0) / 2 };
};
const shuffledRows = rows.map(r => ({ ...r })), random = rng(2026110402);
for (const cue of [0, 1, 2]) {
  const indices = shuffledRows.map((r, i) => r.cue === cue ? i : -1).filter(i => i >= 0);
  const bits = indices.map(i => shuffledRows[i].bit);
  for (let i = bits.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [bits[i], bits[j]] = [bits[j], bits[i]];
  }
  indices.forEach((index, j) => { shuffledRows[index].bit = bits[j]; });
}
const byCue = [0, 1, 2].map(cue => {
  const x = rows.filter(r => r.cue === cue);
  const actual = balanced(x), shuffled = balanced(shuffledRows.filter(r => r.cue === cue));
  const untrained = balanced(x.map(r => ({ ...r, bit: r.untrainedBit })));
  return { cue, trainedBits: counts(x), actual, shuffled, untrained,
    cueOnlyChance: 0.5, aboveShuffle: actual.accuracy - shuffled.accuracy,
    aboveCueOnly: actual.accuracy - 0.5 };
});
const perTeacher = heldout.map(t => ({ ...byTeacher(t),
  decodedByCue: [0, 1, 2].map(cue => ({ cue,
    correct: t.rows.filter(r => r.cue === cue && lookup[cue][r.bit] === t.mapping).length,
    trials: 23 })) }));
const gates = {
  archivedValidity: true,
  trainedVsUntrained: perTeacher.filter(t => t.mapping === 'B').every(t =>
    t.byCue.every(x => x.bits[1] >= 18 && x.untrainedBits[1] === 0)),
  withinCueMapping: byCue.every(x => x.actual.accuracy >= 0.80 &&
    x.aboveCueOnly >= 0.10 && x.aboveShuffle >= 0.10),
  everyHeldoutTeacher: perTeacher.every(t => t.decodedByCue.every(x => x.correct >= 18)),
  nonsaturatedPooled: byCue.every(x => x.trainedBits.every(n => n / 115 >= 0.20)),
  notCueRecoding: byCue.every(x => x.trainedBits[0] > 0 && x.trainedBits[1] > 0)
};
const outputData = { command: 'node tools/check_rps_hidden_strategy_source.mjs',
  design: { calibrationIds, heldoutIds,
    message: 'bit=1 iff Q[observed cue] strictly exceeds both other Q scores; ties 0',
    source: 'saved final controllers and real-MBON frozen features only; no neural replay',
    decoder: 'per-cue maximum conditional bit likelihood with equal A/B priors; calibration only; ties A',
    shuffle: 'one Fisher-Yates permutation of held-out trained bits pooled within cue; rng(2026110402)',
    gates: 'B trained bit1>=18/23 each cue vs untrained 0; balanced accuracy>=80% each cue and >=10 points over chance and shuffle; every held-out teacher>=18/23 each cue; both bits>=20% pooled within each cue; exact archived validity' },
  originalSixTeacherGatePassed: original.passed, strategyBReplicationPassed: replication.passed,
  calibration: calibration.map(byTeacher), lookup,
  heldout: perTeacher, byCue,
  shuffledRows: shuffledRows.map(r => ({ seed: r.seed, mapping: r.mapping, cue: r.cue,
    sequenceSeed: r.sequenceSeed, round: r.round, bit: r.bit })),
  sourceRows: teacherRows, gates, passed: Object.values(gates).every(Boolean),
  receivingStatus: 'not_run', studentTrainingStatus: 'not_run' };
writeFileSync(output, `${JSON.stringify(outputData, null, 2)}\n`);
console.log(JSON.stringify({ lookup, calibration: outputData.calibration,
  heldout: perTeacher, byCue, gates, passed: outputData.passed }, null, 2));
