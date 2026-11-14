// Saved-trace encoding check only; no message or student simulation.
// node tools/check_two_cue_encoding.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';

const output = new URL('../data/two_cue_encoding_check.json', import.meta.url);
if (existsSync(output)) throw Error(`Refusing to overwrite ${output.pathname}`);
const selected = [2146, 2134, 2124, 2122]; // locked by the prior population analysis
const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;
const teachers = Object.fromEntries([501, 502, 503, 504].map(seed => {
  const source = JSON.parse(readFileSync(new URL(`../data/two_cue_source_02_${seed}_lanes02.json`, import.meta.url))).source;
  const indices = selected.map(id => source.sourceIds.indexOf(id));
  if (indices.includes(-1)) throw Error(`Missing MBON in teacher ${seed}`);
  const runs = [0, 1].map(mapping => source.runs.find(r => r.mapping === mapping && r.trained));
  if (runs.some(r => !r || mean(r.frozenAccuracy) < .75)) throw Error(`Teacher ${seed} failed frozen gate`);
  return [seed, runs.flatMap((run, mapping) => run.samples.map(s =>
    ({ seed, cue: s.cue, mapping, values: indices.map(i => s.trace[i]) })))];
}));
const calibration = [...teachers[501], ...teachers[502]];
const heldout = [...teachers[503], ...teachers[504]];
const thresholds = selected.map((_, i) => {
  const sorted = calibration.map(r => r.values[i]).sort((a, b) => a - b);
  return (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2;
});
const bits = (r, n) => r.values.slice(0, n).map((value, i) => Number(value > thresholds[i]));

function fit(n) {
  const probabilities = [0, 1].map(cue => [0, 1].map(mapping => {
    const rows = calibration.filter(r => r.cue === cue && r.mapping === mapping);
    return selected.slice(0, n).map((_, i) => (1 + rows.filter(r => bits(r, n)[i]).length) / (2 + rows.length));
  }));
  return r => {
    const x = bits(r, n);
    const score = mapping => x.reduce((sum, bit, i) => {
      const p = probabilities[r.cue][mapping][i];
      return sum + Math.log(bit ? p : 1 - p);
    }, 0);
    return score(0) >= score(1) ? 0 : 1;
  };
}
function measure(rows, n, predict) {
  return [0, 1].map(cue => {
    const part = rows.filter(r => r.cue === cue);
    return { cue, correct: part.filter(r => predict(r) === r.mapping).length, n: part.length,
      byMapping: [0, 1].map(mapping => {
        const group = part.filter(r => r.mapping === mapping);
        return { correct: group.filter(r => predict(r) === mapping).length, n: group.length,
          bit1Frequencies: selected.slice(0, n).map((_, i) => mean(group.map(r => bits(r, n)[i]))) };
      }) };
  });
}
const approaches = Object.fromEntries([2, 4].map(n => {
  const predict = fit(n);
  return [n + 'bits', { neurons: selected.slice(0, n),
    calibration: measure(calibration, n, predict), heldout: measure(heldout, n, predict),
    heldoutByTeacher: Object.fromEntries([503, 504].map(seed =>
      [seed, measure(teachers[seed], n, predict)])) }];
}));
const result = { command: 'node tools/check_two_cue_encoding.mjs',
  method: 'Per-neuron pooled calibration median; bit = trace > median. Cue-aware Bernoulli naive Bayes fitted on 501-502 only, for analysis only; Laplace smoothing 1.',
  calibrationTeachers: [501, 502], evaluationTeachers: [503, 504],
  selected, thresholds, approaches };
writeFileSync(output, JSON.stringify(result, null, 2));
for (const [name, a] of Object.entries(approaches))
  console.log(name, 'calibration', a.calibration.map(c => `${c.correct}/${c.n}`),
    'heldout', a.heldout.map(c => `${c.correct}/${c.n}`),
    'by teacher', Object.fromEntries(Object.entries(a.heldoutByTeacher).map(([seed, rows]) =>
      [seed, rows.map(c => `${c.correct}/${c.n}`)])));
console.log('thresholds', thresholds, 'saved', output.pathname);
