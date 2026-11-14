// Read-only check of a previously calibrated two-MBON representation.
import { existsSync, readFileSync, writeFileSync } from 'node:fs';

const output = new URL('../data/two_cue_2146_2134_source_check.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite the source result');
const calibration = JSON.parse(readFileSync(new URL('../data/two_cue_encoding_check.json', import.meta.url)));
const ids = [2146, 2134], thresholds = calibration.thresholds.slice(0, 2);
if (calibration.selected.slice(0, 2).some((id, i) => id !== ids[i]) ||
    calibration.calibrationTeachers.join() !== '501,502') throw Error('Unexpected calibration lock');
const evaluationTeachers = [603, 606, 609, 612, 615, 618, 621, 624];

function load(teacher, trained, test = null) {
  const source = JSON.parse(readFileSync(new URL(`../data/two_cue_source_02_${teacher}_lanes02.json`, import.meta.url))).source;
  if (source.sampleMs !== 120) throw Error(`Unexpected sample time: ${teacher}`);
  const indices = ids.map(id => source.sourceIds.indexOf(id));
  if (indices.some(i => i < 0)) throw Error(`Missing MBON: ${teacher}`);
  return [0, 1].flatMap(mapping => {
    const run = source.runs.find(r => r.mapping === mapping && r.trained === trained);
    if (!run) throw Error(`Missing source run: ${teacher}/${mapping}/${trained}`);
    return run.samples.filter(s => test === null || s.test === test).map(s => ({
      teacher, mapping, cue: s.cue, round: s.round,
      bits: indices.map((i, j) => Number(s.trace[i] > thresholds[j]))
    }));
  });
}

const fit = [501, 502].flatMap(teacher => load(teacher, true));
const probabilities = [0, 1].map(cue => [0, 1].map(mapping => {
  const rows = fit.filter(r => r.cue === cue && r.mapping === mapping);
  return ids.map((_, i) => (1 + rows.filter(r => r.bits[i]).length) / (2 + rows.length));
}));
const predict = row => {
  const score = mapping => row.bits.reduce((sum, bit, i) => {
    const p = probabilities[row.cue][mapping][i];
    return sum + Math.log(bit ? p : 1 - p);
  }, 0);
  return score(0) >= score(1) ? 0 : 1;
};
for (const teacher of [503, 504]) {
  const rows = load(teacher, true), expected = calibration.approaches['2bits'].heldoutByTeacher[teacher];
  for (const cue of [0, 1]) {
    const correct = rows.filter(r => r.cue === cue && predict(r) === r.mapping).length;
    if (correct !== expected[cue].correct) throw Error(`Archived decoder mismatch: ${teacher}/${cue}`);
  }
}

const byCue = [0, 1].map(cue => {
  const perTeacher = evaluationTeachers.map(teacher => {
    const trained = load(teacher, true, 2).filter(r => r.cue === cue);
    const untrained = load(teacher, false, 2).filter(r => r.cue === cue);
    const rounds = [...new Set(trained.map(r => r.round))].sort((a, b) => a - b);
    if (rounds.length !== 12 || trained.length !== 24 || untrained.length !== 24 ||
        [0, 1].some(mapping => trained.filter(r => r.mapping === mapping).length !== 12))
      throw Error(`Unbalanced held-out sample: ${teacher}/${cue}`);
    const score = (rows, shuffled = false) => rows.filter(r => {
      const label = shuffled && rounds.indexOf(r.round) % 2 === 1 ? 1 - r.mapping : r.mapping;
      return predict(r) === label;
    }).length;
    const patterns = ['00', '01', '10', '11'];
    return { teacher, trainedCorrect: score(trained), shuffledLabelCorrect: score(trained, true),
      untrainedCorrect: score(untrained),
      messageDistribution: [0, 1].map(mapping => Object.fromEntries(patterns.map(pattern =>
        [pattern, trained.filter(r => r.mapping === mapping && r.bits.join('') === pattern).length]))) };
  });
  const mean = field => perTeacher.reduce((sum, row) => sum + row[field], 0) / perTeacher.length;
  const trained = mean('trainedCorrect'), shuffled = mean('shuffledLabelCorrect'), untrained = mean('untrainedCorrect');
  const gate = { meanAtLeast18: trained >= 18,
    sixTeachersAtLeast16: perTeacher.filter(r => r.trainedCorrect >= 16).length >= 6,
    shuffledMarginAtLeast4: trained - shuffled >= 4,
    untrainedMarginAtLeast4: trained - untrained >= 4 };
  return { cue, perTeacher, meanCorrectOutOf24: { trained, shuffled, untrained, disconnected: 12 },
    pooledMessageDistribution: [0, 1].map(mapping => Object.fromEntries(['00', '01', '10', '11'].map(pattern =>
      [pattern, perTeacher.reduce((sum, r) => sum + r.messageDistribution[mapping][pattern], 0)]))),
    gate, pass: Object.values(gate).every(Boolean) };
});
const result = { command: 'node tools/check_two_cue_2146_2134_source.mjs',
  representation: { ids, thresholds, bit: 'trace > threshold', method: calibration.method,
    calibrationTeachers: [501, 502], archivedVerificationTeachers: [503, 504] },
  evaluationTeachers, heldOutTest: 2,
  shuffle: 'Swap mapping labels on alternating source rounds within teacher and cue',
  byCue, receivingDiagnosticJustified: byCue.every(r => r.pass) };
writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({ byCue: byCue.map(r => ({ cue: r.cue, meanCorrectOutOf24: r.meanCorrectOutOf24,
  individualTrained: r.perTeacher.map(p => p.trainedCorrect),
  pooledMessageDistribution: r.pooledMessageDistribution, gate: r.gate, pass: r.pass })),
  receivingDiagnosticJustified: result.receivingDiagnosticJustified }, null, 2));
