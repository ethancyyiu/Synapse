// Archive-only comparison against the locked fixed-budget source result.
// node tools/check_two_cue_raw_amplitude_source.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';

const output = new URL('../data/two_cue_raw_amplitude_source.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite source result');
const previous = JSON.parse(readFileSync(new URL('../data/two_cue_fixed_budget_source.json', import.meta.url)));
const { ids, ceilings } = previous;
if (ids.join() !== '2146,2134,2124,2122' || ceilings.length !== 4) throw Error('Unexpected archived calibration');

function rows(teacher, trained, test = null) {
  const source = JSON.parse(readFileSync(new URL(`../data/two_cue_source_02_${teacher}_lanes02.json`, import.meta.url))).source;
  if (source.sampleMs !== 120) throw Error(`Unexpected sample time: ${teacher}`);
  const indices = ids.map(id => source.sourceIds.indexOf(id));
  if (indices.some(i => i < 0)) throw Error(`Missing MBON: ${teacher}`);
  return [0, 1].flatMap(mapping => {
    const run = source.runs.find(r => r.mapping === mapping && r.trained === trained);
    if (!run) throw Error(`Missing run: ${teacher}/${mapping}/${trained}`);
    return run.samples.filter(s => test === null || s.test === test).map(s => ({
      mapping, cue: s.cue, round: s.round, trace: indices.map(i => s.trace[i])
    }));
  });
}
const pattern = trace => trace.map((x, i) => Math.min(1, Math.max(0, x / ceilings[i])));
const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;
const calibration = [501, 502].flatMap(teacher => rows(teacher, true));
const centroids = [0, 1].map(cue => [0, 1].map(mapping => {
  const sample = calibration.filter(r => r.cue === cue && r.mapping === mapping);
  if (sample.length !== 72) throw Error(`Invalid calibration cell: ${cue}/${mapping}`);
  return ids.map((_, i) => mean(sample.map(r => pattern(r.trace)[i])));
}));
function predict(row) {
  const x = pattern(row.trace);
  const distance = mapping => x.reduce((sum, value, i) => sum + (value - centroids[row.cue][mapping][i]) ** 2, 0);
  return distance(0) <= distance(1) ? 0 : 1;
}
function assess(sample, label = row => row.mapping) {
  return { correct: sample.filter(r => predict(r) === label(r)).length,
    byMapping: [0, 1].map(m => sample.filter(r => r.mapping === m && predict(r) === label(r)).length) };
}
const evaluation = Object.fromEntries([503, 504].map(teacher => [teacher, [0, 1].map(cue => {
  const trained = rows(teacher, true, 2).filter(r => r.cue === cue);
  const untrained = rows(teacher, false, 2).filter(r => r.cue === cue);
  const rounds = [...new Set(trained.map(r => r.round))].sort((a, b) => a - b);
  if (trained.length !== 24 || untrained.length !== 24 || rounds.length !== 12 ||
      [0, 1].some(m => trained.filter(r => r.mapping === m).length !== 12 ||
        untrained.filter(r => r.mapping === m).length !== 12)) throw Error(`Unbalanced cell: ${teacher}/${cue}`);
  const correct = assess(trained);
  const shuffled = assess(trained, r => r.mapping ^ (rounds.indexOf(r.round) % 2));
  const untrainedControl = assess(untrained);
  const normalized = previous.evaluation[teacher][cue].correct;
  const improvement = correct.correct - normalized.correct;
  const gate = { improvement: improvement >= 4, accuracy: correct.correct >= 18,
    eachMapping: correct.byMapping.every(n => n >= 8),
    untrainedMargin: correct.correct - untrainedControl.correct >= 4,
    shuffledMargin: correct.correct - shuffled.correct >= 4 };
  return { cue, normalized, correct, untrained: untrainedControl, shuffledLabels: shuffled,
    disconnectedChance: 12, improvement, gate, passed: Object.values(gate).every(Boolean),
    trials: trained.map(r => ({ mapping: r.mapping, round: r.round,
      pattern: pattern(r.trace), prediction: predict(r) })) };
})]));
const passed = evaluation[503][0].passed && evaluation[504][1].passed;
const result = { command: 'node tools/check_two_cue_raw_amplitude_source.mjs',
  hypothesis: 'Removing total calibrated activity caused both failed fixed-budget source cells',
  calibrationTeachers: [501, 502], evaluationTeachers: [503, 504], test: 2,
  ids, ceilings, centroids, representation: 'q_i = clamp(trace_i / ceiling_i, 0, 1)',
  gate: 'Both previously failed cells: >=4/24 gain, >=18/24, >=8/12 per mapping, >=4/24 above untrained and shuffled',
  evaluation, passed };
writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({ passed, evaluation: Object.fromEntries(Object.entries(evaluation).map(([teacher, cues]) =>
  [teacher, cues.map(({ trials, ...summary }) => summary)])) }, null, 2));
