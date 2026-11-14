// Archived source-only gate for the predeclared fixed-budget four-MBON pattern.
// node tools/check_two_cue_fixed_budget_source.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';

const output = new URL('../data/two_cue_fixed_budget_source.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite source result');
const ids = [2146, 2134, 2124, 2122];
const archived = JSON.parse(readFileSync(new URL('../data/two_cue_graded_population_decode_02_203_lanes02.json', import.meta.url))).gradedPopulationDecode;
const ceilings = archived.calibrationCeilings;
if (ceilings.length !== 4 || ceilings.some(x => !(x > 0))) throw Error('Invalid archived calibration');

function rows(teacher, trained, test = null) {
  const source = JSON.parse(readFileSync(new URL(`../data/two_cue_source_02_${teacher}_lanes02.json`, import.meta.url))).source;
  if (source.sampleMs !== 120) throw Error(`Unexpected source sample: ${teacher}`);
  const indices = ids.map(id => source.sourceIds.indexOf(id));
  if (indices.some(i => i < 0)) throw Error(`Missing source MBON: ${teacher}`);
  return [0, 1].flatMap(mapping => {
    const run = source.runs.find(r => r.mapping === mapping && r.trained === trained);
    if (!run) throw Error(`Missing run: ${teacher}/${mapping}/${trained}`);
    return run.samples.filter(s => test === null || s.test === test).map(s => ({
      teacher, mapping, cue: s.cue, test: s.test, round: s.round,
      trace: indices.map(i => s.trace[i])
    }));
  });
}
function pattern(trace) {
  const q = trace.map((x, i) => Math.min(1, Math.max(0, x / ceilings[i])));
  const total = q.reduce((sum, x) => sum + x, 0);
  return total ? q.map(x => 300 * x / total) : q;
}
const calibration = [501, 502].flatMap(teacher => rows(teacher, true));
const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;
const centroids = [0, 1].map(cue => [0, 1].map(mapping => {
  const sample = calibration.filter(r => r.cue === cue && r.mapping === mapping);
  if (sample.length !== 72) throw Error(`Invalid calibration cell: ${cue}/${mapping}`);
  return ids.map((_, i) => mean(sample.map(r => pattern(r.trace)[i])));
}));
function predict(row) {
  const x = pattern(row.trace);
  const d = mapping => x.reduce((sum, value, i) => sum + (value - centroids[row.cue][mapping][i]) ** 2, 0);
  return d(0) <= d(1) ? 0 : 1;
}
function assess(sample, label = row => row.mapping) {
  return { correct: sample.filter(r => predict(r) === label(r)).length,
    byMapping: [0, 1].map(m => sample.filter(r => r.mapping === m && predict(r) === label(r)).length) };
}
const evaluation = Object.fromEntries([503, 504].map(teacher => {
  const trained = rows(teacher, true, 2), untrained = rows(teacher, false, 2);
  const byCue = [0, 1].map(cue => {
    const t = trained.filter(r => r.cue === cue), u = untrained.filter(r => r.cue === cue);
    const rounds = [...new Set(t.map(r => r.round))].sort((a, b) => a - b);
    if (t.length !== 24 || u.length !== 24 || rounds.length !== 12 ||
        [0, 1].some(mapping => t.filter(r => r.mapping === mapping).length !== 12 ||
          u.filter(r => r.mapping === mapping).length !== 12))
      throw Error(`Unbalanced evaluation: ${teacher}/${cue}`);
    const correct = assess(t), untrainedControl = assess(u);
    const shuffled = assess(t, r => r.mapping ^ (rounds.indexOf(r.round) % 2));
    const gate = { accuracy: correct.correct >= 18,
      eachMapping: correct.byMapping.every(n => n >= 8),
      untrainedMargin: correct.correct - untrainedControl.correct >= 4,
      shuffledMargin: correct.correct - shuffled.correct >= 4 };
    return { cue, correct, untrained: untrainedControl, shuffledLabels: shuffled,
      zeroPatterns: t.filter(r => pattern(r.trace).every(x => x === 0)).length,
      gate, passed: Object.values(gate).every(Boolean),
      trials: t.map(r => ({ mapping: r.mapping, round: r.round, pattern: pattern(r.trace), prediction: predict(r) })) };
  });
  return [teacher, byCue];
}));
const passed = Object.values(evaluation).every(byCue => byCue.every(x => x.passed));
const result = { command: 'node tools/check_two_cue_fixed_budget_source.mjs',
  hypothesis: 'A fixed total KC drive allocated by the relative four-MBON pattern retains mapping information across both cues',
  calibrationTeachers: [501, 502], evaluationTeachers: [503, 504], test: 2,
  ids, ceilings, centroids, normalization: 'q_i = clamp(trace_i / ceiling_i, 0, 1); rate_i = 300*q_i/sum(q), or all zero',
  gate: 'Each teacher and cue: >=18/24, >=8/12 per mapping, and >=4/24 over untrained and balanced shuffled-label controls',
  evaluation, passed };
writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({ passed, byTeacher: Object.fromEntries(Object.entries(evaluation).map(([teacher, cues]) =>
  [teacher, cues.map(({ cue, correct, untrained, shuffledLabels, zeroPatterns, gate, passed }) =>
    ({ cue, correct, untrained, shuffledLabels, zeroPatterns, gate, passed }))])) }, null, 2));
