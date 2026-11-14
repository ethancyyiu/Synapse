// Analyze saved paired Fly A source runs; does not simulate or change weights.
// node tools/check_two_cue_source_2132.mjs [threshold]
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { rng } from '../js/sim.js';

const thresholdMode = process.argv[2] === 'threshold';
if (process.argv[2] && !thresholdMode) throw Error('Use no argument or threshold');
const output = new URL(thresholdMode
  ? '../data/two_cue_source_2132_threshold.json'
  : '../data/two_cue_source_2132_mapping_info.json', import.meta.url);
if (existsSync(output)) throw Error(`Refusing to overwrite ${output.pathname}`);
const read = name => JSON.parse(readFileSync(new URL(`../data/${name}`, import.meta.url)));
const bit = trace => Number(trace < 0.5);
const mean = xs => xs.reduce((sum, x) => sum + x, 0) / xs.length;
const entropy = p => p === 0 || p === 1 ? 0 : -p * Math.log2(p) - (1 - p) * Math.log2(1 - p);
const information = pairs => {
  const byCue = [0, 1].map(cue => {
    const rows = pairs.filter(row => row.cue === cue);
    const pa = mean(rows.map(row => row.a)), pb = mean(rows.map(row => row.b));
    return { cue, bitInformation: entropy((pa + pb) / 2) - (entropy(pa) + entropy(pb)) / 2,
      bestBitMappingAccuracy: 0.5 + Math.abs(pa - pb) / 2 };
  });
  return { byCue, conditionalBitInformation: mean(byCue.map(row => row.bitInformation)),
    bestBitMappingAccuracy: mean(byCue.map(row => row.bestBitMappingAccuracy)) };
};

if (thresholdMode) {
  function traces(seed) {
    const source = read(`two_cue_source_02_${seed}_lanes02.json`).source;
    const index = source.sourceIds.indexOf(2132);
    assert(index >= 0);
    const trained = [0, 1].map(mapping => source.runs.find(row => row.mapping === mapping && row.trained));
    const untrained = [0, 1].map(mapping => source.runs.find(row => row.mapping === mapping && !row.trained));
    for (const mapping of [0, 1]) {
      const diagnostic = read(`two_cue_diag${mapping}_02_${seed}_lanes02.json`).flyA;
      assert.deepEqual(trained[mapping].training, diagnostic.training.map(row => row.accuracy));
      assert(diagnostic.frozen.filter(row => row.mapping === mapping)
        .every(row => row.accuracy >= 0.75), 'Fly A did not learn its mapping');
    }
    const pairs = trained[0].samples.map((row, i) => {
      const other = trained[1].samples[i];
      assert.deepEqual([row.test, row.round, row.cue], [other.test, other.round, other.cue]);
      assert.equal(untrained[0].samples[i].trace[index], untrained[1].samples[i].trace[index]);
      return { cue: row.cue, test: row.test, a: row.trace[index], b: other.trace[index],
        before: untrained[0].samples[i].trace[index] };
    });
    return pairs;
  }
  const quantiles = values => {
    const sorted = [...values].sort((a, b) => a - b);
    return [0, .25, .5, .75, 1].map(q => sorted[Math.floor(q * (sorted.length - 1))]);
  };
  const evaluate = (pairs, threshold) => {
    const coded = pairs.map(row => ({ cue: row.cue, a: Number(row.a < threshold),
      b: Number(row.b < threshold) }));
    const stats = information(coded);
    return { ...stats, byCue: stats.byCue.map(row => {
      const samples = coded.filter(pair => pair.cue === row.cue);
      return { ...row, mappingABit1: samples.reduce((sum, pair) => sum + pair.a, 0),
        mappingBBit1: samples.reduce((sum, pair) => sum + pair.b, 0), nPerMapping: samples.length };
    }) };
  };
  const calibratedAccuracy = (pairs, threshold, calibration) => [0, 1].map(cue => {
    const samples = pairs.filter(row => row.cue === cue);
    const cal = calibration.byCue[cue], bit1MeansA = cal.mappingABit1 > cal.mappingBBit1;
    const correct = samples.reduce((sum, row) => sum +
      Number((row.a < threshold) === bit1MeansA) +
      Number((row.b < threshold) !== bit1MeansA), 0);
    return { cue, correct, total: 2 * samples.length,
      byTest: [0, 1, 2].map(test => {
        const rows = samples.filter(row => row.test === test);
        return rows.reduce((sum, row) => sum + Number((row.a < threshold) === bit1MeansA) +
          Number((row.b < threshold) !== bit1MeansA), 0);
      }) };
  });
  const shuffledControl = (pairs, threshold, seed) => {
    const random = rng(seed * 7919 + 2132), values = [];
    for (let repeat = 0; repeat < 2000; repeat++) {
      const shuffled = pairs.map(row => random() < .5 ? { ...row, a: row.b, b: row.a } : row);
      values.push(evaluate(shuffled, threshold).conditionalBitInformation);
    }
    values.sort((a, b) => a - b);
    return { permutations: values.length, percentile95: values[Math.floor(.95 * values.length)],
      pAtLeastObserved: (1 + values.filter(x => x >= evaluate(pairs, threshold).conditionalBitInformation).length) /
        (values.length + 1) };
  };
  const calibrationPairs = traces(11);
  const unique = [...new Set(calibrationPairs.flatMap(row => [row.a, row.b]))].sort((a, b) => a - b);
  const candidates = unique.slice(0, -1).map((value, i) => (value + unique[i + 1]) / 2);
  const ranked = candidates.map(threshold => {
    const result = evaluate(calibrationPairs, threshold);
    return { threshold, weakestCueAccuracy: Math.min(...result.byCue.map(row => row.bestBitMappingAccuracy)),
      averageAccuracy: result.bestBitMappingAccuracy };
  }).sort((a, b) => b.weakestCueAccuracy - a.weakestCueAccuracy ||
    b.averageAccuracy - a.averageAccuracy || a.threshold - b.threshold);
  const selected = ranked[0].threshold;
  const evaluationPairs = traces(12);
  const distributions = pairs => [0, 1].map(cue => {
    const rows = pairs.filter(row => row.cue === cue);
    return { cue, mappingAQuantiles: quantiles(rows.map(row => row.a)),
      mappingBQuantiles: quantiles(rows.map(row => row.b)),
      beforeTrainingQuantiles: quantiles(rows.map(row => row.before)) };
  });
  const compare = (pairs, seed) => [0.5, selected].map(threshold => {
    const calibration = evaluate(calibrationPairs, threshold);
    return { threshold, ...evaluate(pairs, threshold),
      mappingAccuracyByCue: calibratedAccuracy(pairs, threshold, calibration),
      pairedShuffle: shuffledControl(pairs, threshold, seed) };
  });
  const result = { command: 'node tools/check_two_cue_source_2132.mjs threshold',
    sourceNeuron: 2132, sampleMs: 120, bitRule: 'trace<threshold',
    calibrationSeed: 11, evaluationSeed: 12, selectedThreshold: selected,
    selection: 'Maximize the weaker cue mapping accuracy on seed 11; break ties by mean accuracy',
    calibration: { distributions: distributions(calibrationPairs), comparisons: compare(calibrationPairs, 11) },
    evaluation: { distributions: distributions(evaluationPairs), comparisons: compare(evaluationPairs, 12) },
    note: 'Mapping accuracy is a diagnostic using cue-wise bit polarity fixed from calibration; no decoder was added to Fly B.' };
  writeFileSync(output, JSON.stringify(result, null, 2));
  for (const row of result.evaluation.comparisons)
    console.log(`seed 12 threshold ${row.threshold.toFixed(3)}: mapping accuracy ` +
      `${row.mappingAccuracyByCue.map(x => `${x.correct}/${x.total}`).join(', ')}, ` +
      `conditional information ${row.conditionalBitInformation.toFixed(3)} bits`);
  console.log(`saved ${output.pathname}`);
} else {
const runs = [];
for (const seed of [11, 12]) {
  const sourceFile = `two_cue_source_02_${seed}_lanes02.json`;
  const source = read(sourceFile).source, index = source.sourceIds.indexOf(2132);
  assert(index >= 0);
  const trained = [0, 1].map(mapping => source.runs.find(row => row.mapping === mapping && row.trained));
  const untrained = [0, 1].map(mapping => source.runs.find(row => row.mapping === mapping && !row.trained));
  const frozen = [0, 1].map(mapping => {
    const diagnostic = read(`two_cue_diag${mapping}_02_${seed}_lanes02.json`).flyA;
    assert.deepEqual(trained[mapping].training, diagnostic.training.map(row => row.accuracy));
    const evaluations = diagnostic.frozen.filter(row => row.mapping === mapping);
    return { correct: evaluations.reduce((sum, row) => sum + row.accuracy * 24, 0), total: 72 };
  });
  assert(frozen.every(row => row.correct >= 54), 'Mapping not reliably learned');
  const pairs = trained[0].samples.map((row, i) => {
    const other = trained[1].samples[i], naiveA = untrained[0].samples[i], naiveB = untrained[1].samples[i];
    assert.deepEqual([row.test, row.round, row.cue], [other.test, other.round, other.cue]);
    assert.equal(naiveA.trace[index], naiveB.trace[index]);
    return { cue: row.cue, test: row.test, a: bit(row.trace[index]), b: bit(other.trace[index]) };
  });
  const observed = information(pairs), random = rng(seed * 7919 + 2132), shuffled = [];
  for (let permutation = 0; permutation < 2000; permutation++) {
    const swapped = pairs.map(row => random() < 0.5
      ? { ...row, a: row.b, b: row.a } : row);
    shuffled.push(information(swapped).conditionalBitInformation);
  }
  shuffled.sort((a, b) => a - b);
  const activity = [0, 1].map(mapping => {
    const rows = trained[mapping].samples;
    return { mapping, lateTrainingAccuracy: mean(trained[mapping].training.slice(-3)),
      frozen: frozen[mapping], byCue: [0, 1].map(cue => {
        const samples = rows.filter(row => row.cue === cue);
        return { cue, n: samples.length, meanTrace: mean(samples.map(row => row.trace[index])),
          bit1: samples.filter(row => bit(row.trace[index]) === 1).length,
          bit1ByTest: [0, 1, 2].map(test => samples.filter(row => row.test === test &&
            bit(row.trace[index]) === 1).length) };
      }) };
  });
  const beforeTraining = [0, 1].map(cue => {
    const samples = untrained[0].samples.filter(row => row.cue === cue);
    return { cue, n: samples.length, meanTrace: mean(samples.map(row => row.trace[index])),
      bit1: samples.filter(row => bit(row.trace[index]) === 1).length };
  });
  runs.push({ seed, sourceFile, frozenFiles: [0, 1].map(mapping =>
    `two_cue_diag${mapping}_02_${seed}_lanes02.json`), beforeTraining, activity, observed,
    pairedShuffle: { permutations: shuffled.length, mean: mean(shuffled),
      percentile95: shuffled[Math.floor(shuffled.length * 0.95)],
      pAtLeastObserved: (1 + shuffled.filter(x => x >= observed.conditionalBitInformation).length) /
        (shuffled.length + 1) } });
}
const result = { command: 'node tools/check_two_cue_source_2132.mjs', sampleMs: 120,
  neuron: 2132, bitEncoding: 'trace<0.5', mapping: ['cue0->action0,cue1->action2',
    'cue0->action2,cue1->action0'], note: 'Paired shuffle swaps mapping labels within the same cue and trial', runs };
writeFileSync(output, JSON.stringify(result, null, 2));
for (const run of runs) console.log(`seed ${run.seed}: mapping bit information ${run.observed.conditionalBitInformation.toFixed(3)} bits; ` +
  `paired shuffle 95th ${run.pairedShuffle.percentile95.toFixed(3)}; bit mapping accuracy ceiling ${run.observed.bestBitMappingAccuracy.toFixed(3)}`);
console.log(`saved ${output.pathname}`);
}
