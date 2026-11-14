import { readFileSync, writeFileSync } from 'node:fs';

const root = new URL('../data/', import.meta.url);
const ties = JSON.parse(readFileSync(new URL('two_cue_graded_population_ties_02_203_lanes02.json', root))).gradedPopulationDecode.pathway;
const choice = JSON.parse(readFileSync(new URL('two_cue_graded_population_choice_02_203_lanes02.json', root))).gradedPopulationDecode.pathway;
const conditions = ['correct', 'shuffled', 'disconnected'];
const result = { hypothesis: 'Calibration-only MBON mapping information persists on all-zero Fly B decision-rate trials', gate: 'Correct-message zero-rate trials: MBON prediction >= 75% accurate and at least 3 more correct than Fly B for each teacher', source: ['two_cue_graded_population_ties_02_203_lanes02.json', 'two_cue_graded_population_choice_02_203_lanes02.json'], teachers: {} };

for (const teacher of ['505', '506']) {
  result.teachers[teacher] = {};
  for (const condition of conditions) {
    const predictions = ties.choice[teacher][condition].trials;
    const archived = choice.choice[teacher][condition].trials;
    const rates = ties.drive[teacher][condition].trials;
    if (predictions.length !== 24 || archived.length !== 24 || rates.length !== 24) throw new Error('Expected 24 paired trials');
    const groups = { zero: { trials: 0, mbonCorrect: 0, flyCorrect: 0, byMapping: [0, 0], mbonCorrectByMapping: [0, 0], predictions: [0, 0] }, nonzero: { trials: 0, mbonCorrect: 0, flyCorrect: 0, byMapping: [0, 0], mbonCorrectByMapping: [0, 0], predictions: [0, 0] } };
    for (let i = 0; i < 24; i++) {
      const p = predictions[i], a = archived[i], r = rates[i];
      if (p.mapping !== a.mapping || p.mapping !== r.mapping || p.sourceRound !== a.sourceRound || p.sourceRound !== r.sourceRound || p.action !== a.action || p.action !== r.action || p.correct !== a.correct || p.correct !== r.correct || p.mbonPrediction !== a.mbonPrediction) throw new Error(`Trial mismatch: ${teacher} ${condition} ${i}`);
      const group = groups[r.rates.every(x => x === 0) ? 'zero' : 'nonzero'];
      group.trials++;
      group.mbonCorrect += Number(p.mbonPrediction === p.mapping);
      group.mbonCorrectByMapping[p.mapping] += Number(p.mbonPrediction === p.mapping);
      group.flyCorrect += p.correct;
      group.byMapping[p.mapping]++;
      group.predictions[p.mbonPrediction]++;
    }
    result.teachers[teacher][condition] = groups;
  }
}
result.numericalGatePassed = Object.values(result.teachers).every(t => {
  const z = t.correct.zero;
  return z.trials > 0 && z.mbonCorrect / z.trials >= 0.75 && z.mbonCorrect - z.flyCorrect >= 3;
});
result.constantClassControl = Object.fromEntries(Object.entries(result.teachers).map(([teacher, outcomes]) => {
  const z = outcomes.correct.zero;
  return [teacher, { majorityClassAccuracy: Math.max(...z.byMapping) / z.trials, mbonAdvantageOverMajority: z.mbonCorrect - Math.max(...z.byMapping) }];
}));
result.informativeSilenceSupported = result.numericalGatePassed && Object.values(result.constantClassControl).every(x => x.mbonAdvantageOverMajority > 0);
writeFileSync(new URL('two_cue_graded_population_silent_archive.json', root), JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify(result));
