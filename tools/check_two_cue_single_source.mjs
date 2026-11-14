// One locked, three-neuron source check using saved frozen Fly A traces.
// node tools/check_two_cue_single_source.mjs calibration|evaluation|evaluation504
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { rng } from '../js/sim.js';

const mode = process.argv[2];
if (!['calibration', 'evaluation', 'evaluation504'].includes(mode)) throw Error('Use calibration or evaluation');
const output = new URL(`../data/two_cue_single_source_${mode}.json`, import.meta.url);
if (existsSync(output)) throw Error(`Refusing to overwrite ${output.pathname}`);
const read = name => JSON.parse(readFileSync(new URL(`../data/${name}`, import.meta.url)));
const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;
const entropy = p => p <= 0 || p >= 1 ? 0 : -p * Math.log2(p) - (1 - p) * Math.log2(1 - p);
const candidates = [2132, 2135, 2134]; // current source, another action-0 MBON, an action-2 MBON

function pairs(seed, neuron) {
  const source = read(`two_cue_source_02_${seed}_lanes02.json`).source;
  const index = source.sourceIds.indexOf(neuron);
  if (index < 0) throw Error(`Missing neuron ${neuron}`);
  const trained = [0, 1].map(mapping => source.runs.find(row => row.mapping === mapping && row.trained));
  if (trained.some(row => row.frozenAccuracy && mean(row.frozenAccuracy) < .75))
    throw Error(`Teacher ${seed} failed mapping training`);
  return trained[0].samples.map((row, i) => {
    const other = trained[1].samples[i];
    if (row.test !== other.test || row.round !== other.round || row.cue !== other.cue)
      throw Error('Teacher samples are not paired');
    return { cue: row.cue, a: row.trace[index], b: other.trace[index] };
  });
}
function assess(bits, bit1MeansA) {
  const byCue = [0, 1].map(cue => {
    const rows = bits.filter(row => row.cue === cue);
    const a = mean(rows.map(row => row.a)), b = mean(rows.map(row => row.b));
    return { cue, nPerMapping: rows.length, bit1ByMapping: [a, b],
      mappingAccuracy: bit1MeansA ? (a + 1 - b) / 2 : (1 - a + b) / 2,
      informationBits: entropy((a + b) / 2) - (entropy(a) + entropy(b)) / 2 };
  });
  return { byCue, weakestCueAccuracy: Math.min(...byCue.map(row => row.mappingAccuracy)),
    meanAccuracy: mean(byCue.map(row => row.mappingAccuracy)),
    conditionalInformationBits: mean(byCue.map(row => row.informationBits)) };
}
const encode = (rows, threshold) => rows.map(row => ({ cue: row.cue,
  a: Number(row.a < threshold), b: Number(row.b < threshold) }));
function shuffled(bits, seed) {
  const random = rng(seed * 7919 + 2179), out = bits.map(row => ({ ...row }));
  const shuffle = values => {
    for (let i = values.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [values[i], values[j]] = [values[j], values[i]];
    }
    return values;
  };
  for (const cue of [0, 1]) {
    const positions = out.flatMap((row, i) => row.cue === cue ? [i] : []);
    const ones = positions.reduce((sum, i) => sum + out[i].a + out[i].b, 0);
    const aOnes = Math.floor(ones / 2) + (ones % 2 && random() < .5 ? 1 : 0);
    const a = shuffle([...Array(aOnes).fill(1), ...Array(positions.length - aOnes).fill(0)]);
    const b = shuffle([...Array(ones - aOnes).fill(1),
      ...Array(positions.length - ones + aOnes).fill(0)]);
    positions.forEach((position, i) => { out[position].a = a[i]; out[position].b = b[i]; });
  }
  return out;
}

if (mode === 'calibration') {
  const seeds = [501, 502], results = candidates.map(neuron => {
    const bySeed = seeds.map(seed => pairs(seed, neuron));
    const values = [...new Set(bySeed.flatMap(rows => rows.flatMap(row => [row.a, row.b])))].sort((a, b) => a - b);
    const thresholds = neuron === 2132 ? [1.915] :
      values.slice(0, -1).map((value, i) => (value + values[i + 1]) / 2);
    const ranked = thresholds.flatMap(threshold => [true, false].map(bit1MeansA => {
      const scores = bySeed.map(rows => assess(encode(rows, threshold), bit1MeansA));
      const pooled = assess(bySeed.flatMap(rows => encode(rows, threshold)), bit1MeansA);
      return { neuron, threshold, bit1MeansA, byTeacher: scores, pooled,
        weakestTeacherCueAccuracy: Math.min(...scores.flatMap(score =>
          score.byCue.map(row => row.mappingAccuracy))) };
    })).sort((a, b) => b.weakestTeacherCueAccuracy - a.weakestTeacherCueAccuracy ||
      b.pooled.weakestCueAccuracy - a.pooled.weakestCueAccuracy ||
      b.pooled.meanAccuracy - a.pooled.meanAccuracy || a.threshold - b.threshold);
    return ranked[0];
  });
  const selected = [...results].sort((a, b) => b.weakestTeacherCueAccuracy - a.weakestTeacherCueAccuracy ||
    b.pooled.weakestCueAccuracy - a.pooled.weakestCueAccuracy ||
    b.pooled.meanAccuracy - a.pooled.meanAccuracy)[0];
  const result = { command: 'node tools/check_two_cue_single_source.mjs calibration',
    calibrationTeachers: seeds, evaluationTeacher: 503, candidateNeurons: candidates,
    rule: 'bit = trace < threshold, with one global bit-to-mapping polarity',
    selection: 'maximize weakest teacher/cue mapping accuracy, then pooled weakest cue',
    minimumReliableCueAccuracy: .7, candidates: results,
    locked: { neuron: selected.neuron, threshold: selected.threshold,
      bit1MeansA: selected.bit1MeansA } };
  writeFileSync(output, JSON.stringify(result, null, 2));
  console.log(`locked neuron ${selected.neuron} threshold ${selected.threshold}; ` +
    `weakest calibration cue ${selected.weakestTeacherCueAccuracy.toFixed(3)}`);
} else {
  const calibration = read('two_cue_single_source_calibration.json');
  const evaluationTeacher = mode === 'evaluation504' ? 504 : 503;
  const compare = [calibration.candidates[0], calibration.locked].map(lock => {
    const rows = pairs(evaluationTeacher, lock.neuron), bits = encode(rows, lock.threshold);
    return { neuron: lock.neuron, threshold: lock.threshold, bit1MeansA: lock.bit1MeansA,
      measured: assess(bits, lock.bit1MeansA), shuffled: assess(shuffled(bits, evaluationTeacher), lock.bit1MeansA) };
  });
  const chosen = compare[1];
  const suitable = calibration.candidates.find(row => row.neuron === chosen.neuron)
    .weakestTeacherCueAccuracy >= calibration.minimumReliableCueAccuracy &&
    chosen.measured.weakestCueAccuracy >= calibration.minimumReliableCueAccuracy &&
    chosen.measured.byCue.every(row => row.informationBits > .05);
  const result = { command: `node tools/check_two_cue_single_source.mjs ${mode}`,
    calibrationFile: 'data/two_cue_single_source_calibration.json',
    evaluationTeacher, locked: calibration.locked, comparisons: compare, suitable };
  writeFileSync(output, JSON.stringify(result, null, 2));
  console.log(`held-out neuron ${chosen.neuron}: ${chosen.measured.byCue.map(row =>
    row.mappingAccuracy.toFixed(3)).join(', ')} by cue; suitable ${suitable}`);
}
console.log(`saved ${output.pathname}`);
