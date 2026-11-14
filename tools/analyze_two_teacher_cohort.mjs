// Locked analysis from docs/two-teacher-cohort-protocol.md. Reads the cohort seed files (indices 3-17) only.
// node tools/analyze_two_teacher_cohort.mjs
import { readFile, readdir, writeFile } from 'node:fs/promises';
import { rng } from '../js/sim.js';

const dir = new URL('../data/two_teacher_cohort/', import.meta.url);
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
const contrast = (a, b, rounds, seed, rows = done) => {
  const values = rows.map(x => at(x, rounds)[a] - at(x, rounds)[b]);
  return { mean: mean(values), bootstrap95: bootstrap(values, seed), perSeed: values };
};
const choosing = contrast('bothGated', 'bothUngated', 96, 2026111111);
const cost = contrast('bothGated', 'trained', 96, 2026111112);
const holds = { choosingHelps: choosing.bootstrap95[0] > 4, badAdviserCostsLittle: cost.bootstrap95[0] > -4 };
const arms = ['alone', 'trained', 'bothUngated', 'bothGated'], CHECKS = [0, 24, 48, 96];
const bySlot = slot => {
  const rows = done.filter(x => x.newRuleSlot === slot);
  return { seeds: rows.length, choosing96: mean(rows.map(x => at(x, 96).bothGated - at(x, 96).bothUngated)),
    cost96: mean(rows.map(x => at(x, 96).bothGated - at(x, 96).trained)) };
};
const result = { protocol: 'docs/two-teacher-cohort-protocol.md', seedsFound: seeds.length, completed: done.length,
  stoppedTeacher: seeds.filter(x => x.status === 'stopped_teacher').length,
  stoppedStaleTeacher: seeds.filter(x => x.status === 'stopped_stale_teacher').length,
  choosingHelps96: choosing, badAdviserCost96: cost, holds,
  passed: done.length >= 10 && Object.values(holds).every(Boolean),
  reportedNotTested: {
    choosingHelps48: contrast('bothGated', 'bothUngated', 48, 2026111113),
    badAdviserCost48: contrast('bothGated', 'trained', 48, 2026111114),
    choosingHelps24: contrast('bothGated', 'bothUngated', 24, 2026111115),
    badAdviserCost24: contrast('bothGated', 'trained', 24, 2026111116),
    gatedMinusAlone96: contrast('bothGated', 'alone', 96, 2026111117),
    newRuleInSlot0: bySlot(0), newRuleInSlot1: bySlot(1) },
  meanWins: Object.fromEntries(CHECKS.map(r => [r, Object.fromEntries(arms.map(arm => [arm, mean(done.map(x => at(x, r)[arm]))]))])),
  meanTrust: Object.fromEntries(CHECKS.map(r => [r, Object.fromEntries(['newRule', 'oldRule'].map(k => [k, mean(done.map(x => at(x, r).trust[k]))]))])),
  integrity: done.every(x => x.teachersFrozen && x.synapsesUnchanged),
  seeds: seeds.map(x => ({ file: x.file, status: x.status, teacherWins: x.teacherWins, staleTeacherWins: x.staleTeacherWins ?? null,
    newRuleSlot: x.newRuleSlot ?? null, points: x.points ?? null })) };
await writeFile(new URL('../data/two_teacher_cohort_summary.json', import.meta.url), JSON.stringify(result, null, 1) + '\n');
console.log(JSON.stringify({ ...result, seeds: undefined }));
