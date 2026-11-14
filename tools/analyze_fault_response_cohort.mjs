// Locked analysis from docs/fault-response-cohort-protocol.md. Reads the cohort seed files (indices 20-34) only.
// node tools/analyze_fault_response_cohort.mjs
import { readFile, readdir, writeFile } from 'node:fs/promises';
import { rng } from '../js/sim.js';

const dir = new URL('../data/fault_response_cohort/', import.meta.url);
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
const controls = ['alone', 'random', 'untrained', 'shuffled'], arms = ['alone', 'trained', 'random', 'untrained', 'shuffled'];
const advantageAt = rounds => done.map(x => at(x, rounds).trained - Math.max(...controls.map(arm => at(x, rounds)[arm])));
const summarize = (values, seed) => ({ mean: mean(values), bootstrap95: bootstrap(values, seed), perSeed: values });
const primary = summarize(advantageAt(48), 2026111130);
const result = { protocol: 'docs/fault-response-cohort-protocol.md', seedsFound: seeds.length, completed: done.length,
  stoppedTeacher: seeds.filter(x => x.status === 'stopped_teacher').length,
  advantage48: primary, passed: done.length >= 10 && primary.bootstrap95[0] > 4,
  reportedNotTested: {
    shareAdvantageAtMost4: advantageAt(48).filter(d => d <= 4).length / done.length,
    trainedMinusAlone48: summarize(done.map(x => at(x, 48).trained - at(x, 48).alone), 2026111131),
    advantage24: summarize(advantageAt(24), 2026111132),
    advantage96: summarize(advantageAt(96), 2026111133) },
  meanWins: Object.fromEntries([0, 24, 48, 96].map(r => [r, Object.fromEntries(arms.map(arm => [arm, mean(done.map(x => at(x, r)[arm]))]))])),
  integrity: done.every(x => x.teachersFrozen && x.synapsesUnchanged),
  seeds: seeds.map(x => ({ file: x.file, status: x.status, teacherWins: x.teacherWins,
    points: x.points ? x.points.map(p => ({ ...p, byAlarm: undefined })) : null })) };
await writeFile(new URL('../data/fault_response_cohort_summary.json', import.meta.url), JSON.stringify(result, null, 1) + '\n');
console.log(JSON.stringify({ ...result, seeds: undefined }));
