// Saved-trace source check only. No Fly B decoder or communication changes.
// node tools/check_two_cue_two_source.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { rng } from '../js/sim.js';

const output = new URL('../data/two_cue_two_source_check.json', import.meta.url);
if (existsSync(output)) throw Error(`Refusing to overwrite ${output.pathname}`);
const patterns = ['00', '01', '10', '11']; // 2132 bit, then 2134 bit
const average = xs => xs.reduce((a, b) => a + b, 0) / xs.length;
const entropy = xs => xs.reduce((sum, n) => n ? sum - n * Math.log2(n) : sum, 0);

function recording(seed) {
  const source = JSON.parse(readFileSync(new URL(`../data/two_cue_source_02_${seed}_lanes02.json`, import.meta.url))).source;
  const indices = [2132, 2134].map(id => source.sourceIds.indexOf(id));
  if (indices.includes(-1)) throw Error(`Teacher ${seed} is missing a source neuron`);
  const runs = [0, 1].map(mapping => source.runs.find(r => r.mapping === mapping && r.trained));
  const frozenAccuracy = runs.map(r => r?.frozenAccuracy && average(r.frozenAccuracy));
  if (frozenAccuracy.some(x => x === undefined || x < .75))
    throw Error(`Teacher ${seed} failed frozen accuracy gate: ${frozenAccuracy}`);
  const rows = runs.flatMap((run, mapping) => run.samples.map((sample, i) => {
    const match = runs[1 - mapping].samples[i];
    if (sample.test !== match.test || sample.round !== match.round || sample.cue !== match.cue)
      throw Error(`Teacher ${seed} has unpaired samples`);
    return { teacher: seed, cue: sample.cue, mapping,
      pattern: `${Number(sample.trace[indices[0]] < 1.915)}${Number(sample.trace[indices[1]] < .2760)}` };
  }));
  return { frozenAccuracy, rows };
}

function information(rows, cue, width) {
  const part = rows.filter(r => r.cue === cue), values = width === 2 ? patterns : ['0', '1'];
  const pick = r => width === 2 ? r.pattern : r.pattern[width === 0 ? 0 : 1];
  const n = part.length, joint = [0, 1].map(m => values.map(p =>
    part.filter(r => r.mapping === m && pick(r) === p).length));
  const totals = values.map((_, i) => joint[0][i] + joint[1][i]);
  const hMapping = entropy([0, 1].map(m => joint[m].reduce((a, b) => a + b, 0) / n));
  const hConditional = totals.reduce((sum, count, i) => count ? sum + count / n *
    entropy([joint[0][i] / count, joint[1][i] / count]) : sum, 0);
  return { counts: joint, bits: hMapping - hConditional };
}

function fitRule(rows) {
  return Object.fromEntries(patterns.map(pattern => {
    const subset = rows.filter(r => r.pattern === pattern);
    const a = subset.filter(r => r.mapping === 0).length;
    return [pattern, a >= subset.length - a ? 0 : 1]; // fixed A tie break
  }));
}

function assess(rows, rule) {
  const byCue = [0, 1].map(cue => {
    const part = rows.filter(r => r.cue === cue), joint = information(rows, cue, 2);
    const b2132 = information(rows, cue, 0).bits;
    const b2134 = information(rows, cue, 1).bits;
    const accuracy = predicate => part.filter(predicate).length / part.length;
    return { cue, nPerMapping: part.length / 2,
      patternCounts: Object.fromEntries(patterns.map((p, i) => [p, [joint.counts[0][i], joint.counts[1][i]]])),
      jointInformationBits: joint.bits, neuron2132InformationBits: b2132,
      neuron2134InformationBits: b2134,
      additionalOver2132Bits: joint.bits - b2132,
      additionalOver2134Bits: joint.bits - b2134,
      fixedRuleAccuracy: accuracy(r => rule[r.pattern] === r.mapping),
      locked2132Accuracy: accuracy(r => (r.pattern[0] === '1' ? 0 : 1) === r.mapping),
      locked2134Accuracy: accuracy(r => (r.pattern[1] === '1' ? 1 : 0) === r.mapping) };
  });
  return { byCue, conditionalInformationBits: average(byCue.map(r => r.jointInformationBits)),
    fixedRuleAccuracy: average(byCue.map(r => r.fixedRuleAccuracy)) };
}

function shuffle(rows, seed) {
  const random = rng(seed), out = rows.map(r => ({ ...r }));
  for (const teacher of [...new Set(out.map(r => r.teacher))]) for (const cue of [0, 1]) {
    const positions = out.flatMap((r, i) => r.teacher === teacher && r.cue === cue ? [i] : []);
    const labels = positions.map(i => out[i].mapping);
    for (let i = labels.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [labels[i], labels[j]] = [labels[j], labels[i]];
    }
    positions.forEach((position, i) => { out[position].mapping = labels[i]; });
  }
  return out;
}

const teachers = Object.fromEntries([501, 502, 503, 504].map(seed => [seed, recording(seed)]));
const rows = seeds => seeds.flatMap(seed => teachers[seed].rows);
const rule = fitRule(rows([501, 502]));
const evaluation = rows([503, 504]);
const shuffles = Array.from({ length: 200 }, (_, i) => assess(shuffle(evaluation, 99173 + i), rule));
const sorted = [0, 1].map(cue => shuffles.map(s => s.byCue[cue].jointInformationBits).sort((a, b) => a - b));
const result = {
  command: 'node tools/check_two_cue_two_source.mjs',
  thresholds: { neuron2132: 1.915, neuron2134: .2760 },
  patternOrder: '2132 then 2134', mappingOrder: ['A', 'B'],
  frozenAccuracy: Object.fromEntries(Object.entries(teachers).map(([seed, value]) => [seed, value.frozenAccuracy])),
  calibrationRule: rule,
  calibration: assess(rows([501, 502]), rule),
  evaluationPooled: assess(evaluation, rule),
  evaluationByTeacher: Object.fromEntries([503, 504].map(seed => [seed, assess(rows([seed]), rule)])),
  cueMatchedShuffledEvaluation: {
    repeats: shuffles.length,
    meanInformationBitsByCue: [0, 1].map(cue => average(shuffles.map(s => s.byCue[cue].jointInformationBits))),
    information95thPercentileByCue: sorted.map(xs => xs[Math.floor(.95 * xs.length)]),
    meanFixedRuleAccuracyByCue: [0, 1].map(cue => average(shuffles.map(s => s.byCue[cue].fixedRuleAccuracy)))
  }
};
writeFileSync(output, JSON.stringify(result, null, 2));
console.log(JSON.stringify({ rule, calibration: result.calibration.byCue.map(r => r.fixedRuleAccuracy),
  evaluation: result.evaluationByTeacher, shuffled: result.cueMatchedShuffledEvaluation }, null, 2));
console.log(`saved ${output.pathname}`);
