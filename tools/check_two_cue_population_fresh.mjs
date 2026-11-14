// Fixed four-MBON source-information check; no Fly B simulation.
// node tools/check_two_cue_population_fresh.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';

const output = new URL('../data/two_cue_population_fresh.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite fresh population result');
const archived = JSON.parse(readFileSync(new URL('../data/two_cue_population_check.json', import.meta.url)));
const ids = archived.selected.map(row => row.id);
if (JSON.stringify(ids) !== JSON.stringify([2146, 2134, 2124, 2122]))
  throw Error('Archived source selection changed');
const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;
const sources = Object.fromEntries([501, 502, 503, 504, 505, 506].map(seed => [seed,
  JSON.parse(readFileSync(new URL(`../data/two_cue_source_02_${seed}_lanes02.json`, import.meta.url))).source]));
const rows = (seed, trained, test = null, cue = 1) => {
  const source = sources[seed], indices = ids.map(id => source.sourceIds.indexOf(id));
  if (indices.includes(-1)) throw Error(`Missing source neuron in teacher ${seed}`);
  return [0, 1].flatMap(mapping => {
    const run = source.runs.find(row => row.mapping === mapping && row.trained === trained);
    if (!run || (trained && mean(run.frozenAccuracy) < .75))
      throw Error(`Teacher ${seed} failed frozen source gate`);
    return run.samples.filter(sample => (cue === null || sample.cue === cue) && (test === null || sample.test === test))
      .map(sample => ({ mapping, cue: sample.cue, test: sample.test, sourceRound: sample.round,
        trace: indices.map(index => sample.trace[index]) }));
  });
};
const calibration = [501, 502].flatMap(seed => rows(seed, true, null, null));
const scale = ids.map((_, i) => {
  const xs = calibration.map(row => row.trace[i]), center = mean(xs);
  return { mean: center, sd: Math.sqrt(mean(xs.map(x => (x - center) ** 2))) || 1 };
});
const vector = row => row.trace.map((x, i) => (x - scale[i].mean) / scale[i].sd);
const centroids = [0, 1].map(mapping => ids.map((_, i) =>
  mean(calibration.filter(row => row.cue === 1 && row.mapping === mapping).map(row => vector(row)[i]))));
const predict = row => {
  const x = vector(row);
  const distance = mapping => x.reduce((sum, value, i) => sum + (value - centroids[mapping][i]) ** 2, 0);
  return distance(0) <= distance(1) ? 0 : 1;
};
// Check exact reconstruction of the archived cue-1 analytical decoder.
for (const seed of [503, 504]) {
  const check = rows(seed, true);
  const expected = archived.evaluationByTeacher[seed].population[1];
  if (check.length !== expected.n || [0, 1].some(mapping =>
    check.filter(row => row.mapping === mapping && predict(row) === mapping).length !==
      expected.byMapping[mapping].correct)) throw Error(`Archived decoder mismatch: ${seed}`);
}
const assess = sample => ({ score: sample.filter(row => predict(row) === row.mapping).length,
  byMapping: [0, 1].map(mapping => sample.filter(row => row.mapping === mapping && predict(row) === mapping).length) });
const shuffle = (labels, seed) => {
  let state = seed >>> 0;
  for (let i = labels.length - 1; i > 0; i--) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    const j = Math.floor((state / 4294967296) * (i + 1));
    [labels[i], labels[j]] = [labels[j], labels[i]];
  }
  return labels;
};
const evaluation = Object.fromEntries([505, 506].map(seed => {
  const trained = rows(seed, true, 2), untrained = rows(seed, false, 2);
  if (trained.length !== 24 || untrained.length !== 24 ||
      [trained, untrained].some(sample => [0, 1].some(mapping => sample.filter(row => row.mapping === mapping).length !== 12)))
    throw Error(`Unbalanced source test: ${seed}`);
  const labels = shuffle(trained.map(row => row.mapping), 20261102 + seed);
  const shuffled = trained.map((row, i) => ({ ...row, mapping: labels[i] }));
  return [seed, { trained: assess(trained), untrained: assess(untrained),
    shuffledLabels: assess(shuffled), disconnected: { score: 12, byMapping: [12, 0] },
    trials: trained.map((row, i) => ({ mapping: row.mapping, shuffledLabel: labels[i],
      test: row.test, sourceRound: row.sourceRound, trace: row.trace, prediction: predict(row) })) }];
}));
const passed = [505, 506].every(seed => {
  const r = evaluation[seed];
  return r.trained.score >= 18 && r.trained.byMapping.every(n => n >= 8) && r.shuffledLabels.score <= 14;
});
const result = { command: 'node tools/check_two_cue_population_fresh.mjs',
  hypothesis: 'The fixed four-MBON analytical source readout generalizes to fresh trained teachers',
  sourceNeurons: ids, calibrationTeachers: [501, 502], evaluationTeachers: [505, 506],
  evaluationTest: 2, shuffleSeedRule: '20261102 + teacher seed',
  gate: 'each teacher: trained >= 18/24 and >= 8/12 per mapping; shuffled labels <= 14/24',
  archivedDecoderReproduced: true, evaluation, passed };
writeFileSync(output, JSON.stringify(result, null, 2));
console.log(JSON.stringify({ passed, evaluation: Object.fromEntries([505, 506].map(seed => {
  const { trained, untrained, shuffledLabels, disconnected } = evaluation[seed];
  return [seed, { trained, untrained, shuffledLabels, disconnected }];
})) }, null, 2));
