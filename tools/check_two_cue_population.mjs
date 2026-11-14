// One calibration-only four-MBON selection and held-out analytical readout.
// node tools/check_two_cue_population.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';

const output = new URL('../data/two_cue_population_check.json', import.meta.url);
if (existsSync(output)) throw Error(`Refusing to overwrite ${output.pathname}`);
const avg = xs => xs.reduce((a, b) => a + b, 0) / xs.length;
const entropy = p => p === 0 || p === 1 ? 0 : -p * Math.log2(p) - (1 - p) * Math.log2(1 - p);
const groups = {};
let ids;
for (const seed of [501, 502, 503, 504]) {
  const source = JSON.parse(readFileSync(new URL(`../data/two_cue_source_02_${seed}_lanes02.json`, import.meta.url))).source;
  if (ids && JSON.stringify(ids) !== JSON.stringify(source.sourceIds)) throw Error('Different MBON lists');
  ids = source.sourceIds;
  const runs = [0, 1].map(mapping => source.runs.find(r => r.trained && r.mapping === mapping));
  if (runs.some(r => !r || !r.frozenAccuracy || avg(r.frozenAccuracy) < .75))
    throw Error(`Teacher ${seed} did not pass the frozen accuracy gate`);
  groups[seed] = runs.flatMap((run, mapping) => run.samples.map((s, i) => {
    const paired = runs[1 - mapping].samples[i];
    if (s.test !== paired.test || s.round !== paired.round || s.cue !== paired.cue)
      throw Error(`Unpaired teacher ${seed} recording`);
    return { seed, mapping, cue: s.cue, trace: s.trace };
  }));
}
const calibration = [...groups[501], ...groups[502]];
const heldout = [...groups[503], ...groups[504]];
const subset = (rows, cue, mapping) => rows.filter(r => r.cue === cue && r.mapping === mapping);
const contrast = (rows, cue, index) => {
  const a = subset(rows, cue, 0).map(r => r.trace[index]);
  const b = subset(rows, cue, 1).map(r => r.trace[index]);
  const ma = avg(a), mb = avg(b);
  const variance = (xs, m) => avg(xs.map(x => (x - m) ** 2));
  return (ma - mb) / Math.sqrt((variance(a, ma) + variance(b, mb)) / 2 + 1e-9);
};
// Rank once: a feature must separate both mappings in the same direction across
// both calibration teachers, for each cue. No held-out trace enters this ranking.
const ranked = ids.map((id, index) => {
  const d = [501, 502].map(seed => [0, 1].map(cue => contrast(groups[seed], cue, index)));
  const stable = [0, 1].every(cue => d[0][cue] * d[1][cue] > 0);
  return { id, index, score: stable ? Math.min(...d.flat().map(Math.abs)) : 0, contrasts: d };
}).sort((a, b) => b.score - a.score || a.id - b.id);
const selected = ranked.slice(0, 4);
const indices = selected.map(r => r.index);

// One cue-aware, Euclidean nearest-centroid decoder. Standardization and class
// centroids are fitted only on teachers 501-502, then frozen for 503-504.
const scale = indices.map(index => {
  const xs = calibration.map(r => r.trace[index]), mean = avg(xs);
  return { mean, sd: Math.sqrt(avg(xs.map(x => (x - mean) ** 2))) || 1 };
});
const vector = r => indices.map((index, i) => (r.trace[index] - scale[i].mean) / scale[i].sd);
const centroids = [0, 1].map(cue => [0, 1].map(mapping =>
  indices.map((_, i) => avg(subset(calibration, cue, mapping).map(r => vector(r)[i])))));
const predictPopulation = r => {
  const x = vector(r);
  const distance = mapping => x.reduce((sum, value, i) => sum + (value - centroids[r.cue][mapping][i]) ** 2, 0);
  return distance(0) <= distance(1) ? 0 : 1;
};
const bit = (r, id, threshold) => Number(r.trace[ids.indexOf(id)] < threshold);
const twoPattern = r => `${bit(r, 2132, 1.915)}${bit(r, 2134, .2760)}`;
const patternKeys = ['00', '01', '10', '11'];
const cueAwareRule = [0, 1].map(cue => Object.fromEntries(patternKeys.map(pattern => {
  const matching = calibration.filter(r => r.cue === cue && twoPattern(r) === pattern);
  const a = matching.filter(r => r.mapping === 0).length;
  return [pattern, a >= matching.length - a ? 0 : 1];
})));
const singleRule = [0, 1].map(cue => [0, 1].map(value => {
  const matching = calibration.filter(r => r.cue === cue && bit(r, 2132, 1.915) === value);
  const a = matching.filter(r => r.mapping === 0).length;
  return a >= matching.length - a ? 0 : 1;
}));
const predictors = {
  population: predictPopulation,
  neuron2132Locked: r => bit(r, 2132, 1.915) ? 0 : 1,
  neuron2132CueAware: r => singleRule[r.cue][bit(r, 2132, 1.915)],
  twoBitGlobal: r => ({ '10': 0, '11': 1, '00': 1, '01': 1 })[twoPattern(r)],
  twoBitCueAware: r => cueAwareRule[r.cue][twoPattern(r)]
};
function assess(rows) {
  return Object.fromEntries(Object.entries(predictors).map(([name, predict]) => [name, [0, 1].map(cue => {
    const part = rows.filter(r => r.cue === cue), correct = part.filter(r => predict(r) === r.mapping).length;
    const byMapping = [0, 1].map(mapping => {
      const group = part.filter(r => r.mapping === mapping);
      return { correct: group.filter(r => predict(r) === mapping).length, n: group.length };
    });
    const p0A = byMapping[0].correct / byMapping[0].n;
    const p0B = 1 - byMapping[1].correct / byMapping[1].n;
    return { cue, correct, n: part.length, byMapping,
      decodedInformationBits: entropy((p0A + p0B) / 2) - (entropy(p0A) + entropy(p0B)) / 2 };
  })]));
}
const result = {
  command: 'node tools/check_two_cue_population.mjs',
  method: 'Top four MBONs by minimum stable standardized mapping separation across calibration teachers and cues; per-cue nearest standardized class centroid.',
  calibrationTeachers: [501, 502], evaluationTeachers: [503, 504],
  selected: selected.map(({ id, score, contrasts }) => ({ id, score, contrasts })),
  singleRule, cueAwareRule,
  calibration: assess(calibration), evaluation: assess(heldout),
  evaluationByTeacher: Object.fromEntries([503, 504].map(seed => [seed, assess(groups[seed])]))
};
writeFileSync(output, JSON.stringify(result, null, 2));
console.log(JSON.stringify({ selected: result.selected.map(r => [r.id, r.score]),
  calibration: result.calibration, evaluation: result.evaluation,
  byTeacher: result.evaluationByTeacher }, null, 2));
console.log(`saved ${output.pathname}`);
