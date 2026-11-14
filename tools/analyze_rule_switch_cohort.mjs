// Locked analysis from docs/rule-switch-cohort-protocol.md. Reads the 15 seed files only.
// node tools/analyze_rule_switch_cohort.mjs
import { readFile, readdir, writeFile } from 'node:fs/promises';
import { rng } from '../js/sim.js';

const dir = new URL('../data/rule_switch_cohort/', import.meta.url);
const files = (await readdir(dir)).filter(name => /^seed_\d+\.json$/.test(name))
  .sort((a, b) => parseInt(a.slice(5)) - parseInt(b.slice(5)));
const seeds = [];
for (const name of files) seeds.push({ file: name, ...JSON.parse(await readFile(new URL(name, dir))) });
const done = seeds.filter(x => x.status === 'completed');
const at = (x, rounds) => x.points.find(p => p.rounds === rounds);
const mean = list => list.reduce((a, b) => a + b, 0) / list.length;
const values = done.map(x => x.advantage48);
const random = rng(2026110905), boots = [];
for (let i = 0; i < 10000; i++) boots.push(mean(values.map(() => values[Math.floor(random() * values.length)])));
boots.sort((a, b) => a - b);
const interval = [boots[249], boots[9749]];
const vsRandom = done.map(x => at(x, 48).trained - at(x, 48).random);
const arms = ['alone', 'trained', 'random', 'untrained', 'shuffled'];
const meanAt = rounds => Object.fromEntries(arms.map(arm => [arm, mean(done.map(x => at(x, rounds)[arm]))]));
const result = { protocol: 'docs/rule-switch-cohort-protocol.md', seedsFound: seeds.length, completed: done.length,
  stoppedTeacher: seeds.filter(x => x.status === 'stopped_teacher').length,
  meanAdvantage48: mean(values), bootstrap95: interval,
  passed: done.length >= 10 && interval[0] > 4,
  randomMatchesTeacherShare: vsRandom.filter(d => d <= 4).length / done.length,
  meanTrainedMinusRandom48: mean(vsRandom), meanWins48: meanAt(48), meanWins96: meanAt(96),
  integrity: done.every(x => x.teachersFrozen && x.synapsesUnchanged),
  seeds: seeds.map(x => ({ file: x.file, status: x.status, teacherWins: x.teacherWins,
    advantage48: x.advantage48 ?? null, points: x.points ?? null })) };
await writeFile(new URL('../data/rule_switch_cohort_summary.json', import.meta.url), JSON.stringify(result, null, 1) + '\n');
console.log(JSON.stringify({ ...result, seeds: undefined }));
console.log(seeds.map(x => `${x.file.slice(5, -5)}:${x.status === 'completed' ? x.advantage48 : x.status}`).join(' '));
