// Locked analysis from docs/stale-teacher-cohort-protocol.md. Reads the cohort seed files (indices 3-17) only.
// node tools/analyze_stale_teacher_cohort.mjs
import { readFile, readdir, writeFile } from 'node:fs/promises';
import { rng } from '../js/sim.js';

const dir = new URL('../data/stale_teacher_cohort/', import.meta.url);
const files = (await readdir(dir)).filter(name => /^seed_\d+\.json$/.test(name))
  .sort((a, b) => parseInt(a.slice(5)) - parseInt(b.slice(5)));
const seeds = [];
for (const name of files) seeds.push({ file: name, ...JSON.parse(await readFile(new URL(name, dir))) });
const done = seeds.filter(x => x.status === 'completed');
const at = (x, rounds) => x.points.find(p => p.rounds === rounds);
const mean = list => list.reduce((a, b) => a + b, 0) / list.length;
const bootstrap = (values, seed) => {
  const random = rng(seed), boots = [];
  for (let i = 0; i < 10000; i++) boots.push(mean(values.map(() => values[Math.floor(random() * values.length)])));
  boots.sort((a, b) => a - b);
  return [boots[249], boots[9749]];
};
const difference = done.map(x => at(x, 48).trained - at(x, 48).stale);
const interval = bootstrap(difference, 2026111001);
const staleVsAlone = rounds => done.map(x => at(x, rounds).stale - at(x, rounds).alone);
const arms = ['alone', 'trained', 'stale'];
const meanAt = rounds => Object.fromEntries(arms.map(arm => [arm, mean(done.map(x => at(x, rounds)[arm]))]));
const result = { protocol: 'docs/stale-teacher-cohort-protocol.md', seedsFound: seeds.length, completed: done.length,
  stoppedTeacher: seeds.filter(x => x.status === 'stopped_teacher').length,
  stoppedStaleTeacher: seeds.filter(x => x.status === 'stopped_stale_teacher').length,
  meanTrainedMinusStale48: mean(difference), bootstrap95: interval,
  passed: done.length >= 10 && interval[0] > 4,
  staleMinusAlone48: { mean: mean(staleVsAlone(48)), bootstrap95: bootstrap(staleVsAlone(48), 2026111002) },
  staleMinusAlone96: { mean: mean(staleVsAlone(96)), bootstrap95: bootstrap(staleVsAlone(96), 2026111002),
    shareStaleBelowAloneBy4: staleVsAlone(96).filter(d => d < -4).length / done.length },
  meanWins: Object.fromEntries([0, 24, 48, 96].map(rounds => [rounds, meanAt(rounds)])),
  messageAgreement: { agreed: done.reduce((a, x) => a + x.preferences.agreed, 0), bothSent: done.reduce((a, x) => a + x.preferences.bothSent, 0) },
  integrity: done.every(x => x.teachersFrozen && x.synapsesUnchanged),
  seeds: seeds.map(x => ({ file: x.file, status: x.status, teacherWins: x.teacherWins, staleTeacherWins: x.staleTeacherWins ?? null,
    trainedMinusStale48: x.trainedMinusStale48 ?? null, points: x.points ?? null })) };
await writeFile(new URL('../data/stale_teacher_cohort_summary.json', import.meta.url), JSON.stringify(result, null, 1) + '\n');
console.log(JSON.stringify({ ...result, seeds: undefined }));
console.log(seeds.map(x => `${x.file.slice(5, -5)}:${x.status === 'completed' ? x.trainedMinusStale48 : x.status}`).join(' '));
