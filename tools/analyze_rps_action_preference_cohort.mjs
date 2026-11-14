// Locked cohort analysis from docs/action-preference-cohort-protocol.md. Reads the 20 pair files only.
// node tools/analyze_rps_action_preference_cohort.mjs
import { readFile, readdir, writeFile } from 'node:fs/promises';
import { rng } from '../js/sim.js';

const dir = new URL('../data/rps_action_preference_cohort/', import.meta.url);
const files = (await readdir(dir)).filter(name => /^pair_\d\d\.json$/.test(name)).sort();
const controls = ['alone', 'untrained', 'shuffled', 'generic'];
const pairs = [];
for (const name of files) {
  const pair = JSON.parse(await readFile(new URL(name, dir)));
  const summary = pair.pilot?.summary;
  pairs.push({ file: name, status: pair.status, teacherWins: pair.teacher?.wins ?? null,
    completed: Boolean(summary), passed: summary?.passed ?? false,
    advantage: summary ? summary.early.trained - Math.max(...controls.map(arm => summary.early[arm])) : null,
    early: summary?.early ?? null, first58: summary?.first58 ?? null, final: summary?.final ?? null });
}
const done = pairs.filter(pair => pair.completed), values = done.map(pair => pair.advantage);
const mean = list => list.reduce((a, b) => a + b, 0) / list.length;
const random = rng(2026110901), boots = [];
for (let i = 0; i < 10000; i++)
  boots.push(mean(values.map(() => values[Math.floor(random() * values.length)])));
boots.sort((a, b) => a - b);
const interval = [boots[249], boots[9749]];
const passRate = done.filter(pair => pair.passed).length / done.length;
const stopped = Object.fromEntries([...new Set(pairs.filter(p => !p.completed).map(p => p.status))]
  .map(status => [status, pairs.filter(p => p.status === status).length]));
const finalMean = Object.fromEntries(['trained', ...controls].map(arm => [arm, mean(done.map(p => p.final[arm]))]));
const result = { protocol: 'docs/action-preference-cohort-protocol.md', pairsFound: pairs.length,
  completed: done.length, stopped, perPairPassed: done.filter(p => p.passed).length, passRate,
  meanAdvantage: mean(values), bootstrap95: interval, finalMean,
  cohortPassed: done.length >= 15 && interval[0] > 4 && passRate >= .75, pairs };
await writeFile(new URL('../data/rps_action_preference_cohort_summary.json', import.meta.url),
  JSON.stringify(result, null, 1) + '\n');
console.log(JSON.stringify({ ...result, pairs: undefined }));
console.log(pairs.map(p => `${p.file.slice(5, 7)}:${p.completed ? p.advantage.toFixed(1) + (p.passed ? 'P' : 'F') : p.status}`).join(' '));
