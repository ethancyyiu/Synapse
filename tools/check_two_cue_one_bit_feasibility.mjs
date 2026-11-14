import { readFileSync, writeFileSync, existsSync } from 'node:fs';

const teachers = [603, 606, 609, 612, 615, 618, 621, 624];
const threshold = 1.915;
const output = new URL('../data/two_cue_one_bit_feasibility.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite existing feasibility result');

const rows = [];
for (const teacher of teachers) {
  const path = new URL(`../data/two_cue_source_02_${teacher}_lanes02.json`, import.meta.url);
  const source = JSON.parse(readFileSync(path)).source;
  if (source.sampleMs !== 120) throw Error(`Unexpected sample time for ${teacher}`);
  const index = source.sourceIds.indexOf(2132);
  if (index < 0) throw Error(`Missing neuron 2132 for ${teacher}`);
  for (const trained of [true, false]) for (const mapping of [0, 1]) {
    const run = source.runs.find(run => run.trained === trained && run.mapping === mapping);
    if (!run) throw Error(`Missing run for ${teacher}/${trained}/${mapping}`);
    for (const sample of run.samples.filter(sample => sample.test === 2)) {
      rows.push({ teacher, trained, mapping, cue: sample.cue, round: sample.round,
        bit: Number(sample.trace[index] < threshold) });
    }
  }
}

const byCue = [0, 1].map(cue => {
  const perTeacher = teachers.map(teacher => {
    const selected = rows.filter(row => row.teacher === teacher && row.cue === cue);
    const rounds = [...new Set(selected.map(row => row.round))].sort((a, b) => a - b);
    if (rounds.length !== 12 || selected.length !== 48) throw Error(`Unbalanced source for ${teacher}/${cue}`);
    const score = (trained, shuffle = false) => {
      const group = selected.filter(row => row.trained === trained);
      const counts = [0, 1].map(mapping => {
        const subset = group.filter(row => row.mapping === mapping);
        if (subset.length !== 12) throw Error(`Unbalanced mapping for ${teacher}/${cue}/${mapping}`);
        return { mapping: mapping === 0 ? 'A' : 'B', bit0: subset.filter(row => row.bit === 0).length,
          bit1: subset.filter(row => row.bit === 1).length };
      });
      const correct = group.filter(row => {
        const label = shuffle && rounds.indexOf(row.round) % 2 === 1 ? 1 - row.mapping : row.mapping;
        return (row.bit === 1 ? 0 : 1) === label;
      }).length;
      return { correct, total: 24, balancedAccuracy: correct / 24, messageDistribution: counts };
    };
    return { teacher, trained: score(true), shuffledLabels: score(true, true), untrained: score(false) };
  });
  const mean = condition => perTeacher.reduce((n, row) => n + row[condition].correct, 0) / teachers.length;
  const trained = mean('trained'), shuffled = mean('shuffledLabels'), untrained = mean('untrained');
  const gate = { meanAtLeast18: trained >= 18,
    sixTeachersAtLeast16: perTeacher.filter(row => row.trained.correct >= 16).length >= 6,
    shuffledMarginAtLeast4: trained - shuffled >= 4,
    untrainedMarginAtLeast4: trained - untrained >= 4 };
  return { cue, perTeacher, meanCorrectOutOf24: { trained, shuffledLabels: shuffled, untrained, disconnected: 12 },
    pooledMessageDistribution: [0, 1].map(mapping => ({ mapping: mapping === 0 ? 'A' : 'B',
      bit0: perTeacher.reduce((n, row) => n + row.trained.messageDistribution[mapping].bit0, 0),
      bit1: perTeacher.reduce((n, row) => n + row.trained.messageDistribution[mapping].bit1, 0) })),
    gate, pass: Object.values(gate).every(Boolean) };
});
const result = { command: 'node tools/check_two_cue_one_bit_feasibility.mjs',
  hypothesis: 'The fixed one-bit message distinguishes mappings on each cue',
  sourceNeuron: 2132, sampleMs: 120, bit: 'trace < 1.915',
  classifier: 'bit 1 => A, bit 0 => B', teachers, heldOutTest: 2,
  shuffle: 'Swap mapping labels on alternating source rounds within each teacher and cue',
  byCue, proceedToTwoCueLearning: byCue.every(row => row.pass) };
writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({ byCue: byCue.map(({ cue, meanCorrectOutOf24, pooledMessageDistribution, gate, pass, perTeacher }) =>
  ({ cue, meanCorrectOutOf24, pooledMessageDistribution, individualTrained: perTeacher.map(row => row.trained.correct), gate, pass })),
  proceedToTwoCueLearning: result.proceedToTwoCueLearning }, null, 2));
