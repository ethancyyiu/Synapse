// Locked source-preservation gate for the existing cue-gated two-bit receiver.
import { existsSync, readFileSync, writeFileSync } from 'node:fs';

const output = new URL('../data/two_cue_receiver_gate.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite receiver gate result');
const ids = [2146, 2134], thresholds = [0.000368, 0.232596];
function rows(seed, test = null) {
  const source = JSON.parse(readFileSync(new URL(`../data/two_cue_source_02_${seed}_lanes02.json`, import.meta.url))).source;
  const index = ids.map(id => source.sourceIds.indexOf(id));
  if (source.sampleMs !== 120 || index.some(i => i < 0)) throw Error(`Invalid source ${seed}`);
  return [0, 1].flatMap(mapping => {
    const run = source.runs.find(r => r.trained && r.mapping === mapping);
    if (!run) throw Error(`Missing trained source ${seed}/${mapping}`);
    return run.samples.filter(s => test === null || s.test === test).map(s => ({ seed, mapping,
      cue: s.cue, bits: index.map((i, k) => Number(s.trace[i] < thresholds[k])) }));
  });
}
const calibration = [501, 502].flatMap(seed => rows(seed));
const probabilities = [0, 1].map(cue => [0, 1].map(mapping => {
  const group = calibration.filter(r => r.cue === cue && r.mapping === mapping);
  return ids.map((_, i) => (1 + group.filter(r => i === 0 && cue === 1 ? false : r.bits[i]).length) / (2 + group.length));
}));
function predict(row) {
  const bits = row.cue === 1 ? [0, row.bits[1]] : row.bits;
  const score = mapping => bits.reduce((sum, bit, i) => {
    const p = probabilities[row.cue][mapping][i];
    return sum + Math.log(bit ? p : 1 - p);
  }, 0);
  return score(0) >= score(1) ? 0 : 1;
}
const byTeacher = [603, 606].map(seed => {
  const heldout = rows(seed, 2);
  return { seed, byCue: [0, 1].map(cue => {
    const group = heldout.filter(r => r.cue === cue);
    if (group.length !== 24 || [0, 1].some(m => group.filter(r => r.mapping === m).length !== 12))
      throw Error(`Unbalanced ${seed}/${cue}`);
    return { cue, correct: group.filter(r => predict(r) === r.mapping).length, total: 24 };
  }) };
});
const meanByCue = [0, 1].map(cue => byTeacher.reduce((n, t) => n + t.byCue[cue].correct, 0) / 2);
const pass = meanByCue.every(n => n >= 18) && byTeacher.every(t => t.byCue.every(c => c.correct >= 16));
const result = { command: 'node tools/check_two_cue_receiver_gate.mjs', ids, thresholds,
  method: 'Cue-gated projection of existing receiver bits; cue-aware Laplace-smoothed Bernoulli decoder fitted on teachers 501/502 only',
  byTeacher, meanByCue, gate: 'mean >=18/24 per cue; each teacher >=16/24 per cue', pass };
writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({ byTeacher, meanByCue, pass }, null, 2));
