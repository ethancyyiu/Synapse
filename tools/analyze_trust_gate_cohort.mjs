// Locked analysis from docs/trust-gate-cohort-protocol.md. Reads the cohort seed files (indices 3-17) only.
// node tools/analyze_trust_gate_cohort.mjs
import { readFile, readdir, writeFile } from 'node:fs/promises';
import { rng } from '../js/sim.js';

const dir = new URL('../data/trust_gate_cohort/', import.meta.url);
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
const contrast = (a, b, rounds, seed) => {
  const values = done.map(x => at(x, rounds)[a] - at(x, rounds)[b]);
  return { mean: mean(values), bootstrap95: bootstrap(values, seed), perSeed: values };
};
const protection = contrast('gatedStale', 'stale', 96, 2026111101);
const noHarm = contrast('gatedStale', 'alone', 96, 2026111102);
const benefitKept = contrast('gatedTrained', 'trained', 48, 2026111103);
const arms = ['alone', 'trained', 'stale', 'gatedTrained', 'gatedStale'], CHECKS = [0, 24, 48, 96];
const holds = { protection: protection.bootstrap95[0] > 4, noHarm: noHarm.bootstrap95[0] > -4, benefitKept: benefitKept.bootstrap95[0] > -4 };
const result = { protocol: 'docs/trust-gate-cohort-protocol.md', seedsFound: seeds.length, completed: done.length,
  stoppedTeacher: seeds.filter(x => x.status === 'stopped_teacher').length,
  stoppedStaleTeacher: seeds.filter(x => x.status === 'stopped_stale_teacher').length,
  protection96: protection, noHarm96: noHarm, benefitKept48: benefitKept, holds,
  passed: done.length >= 10 && Object.values(holds).every(Boolean),
  gatedTrainedMinusAlone48: contrast('gatedTrained', 'alone', 48, 2026111104),
  meanWins: Object.fromEntries(CHECKS.map(r => [r, Object.fromEntries(arms.map(arm => [arm, mean(done.map(x => at(x, r)[arm]))]))])),
  meanTrust: Object.fromEntries(CHECKS.map(r => [r, Object.fromEntries(['gatedTrained', 'gatedStale'].map(arm => [arm, mean(done.map(x => at(x, r).trust[arm]))]))])),
  integrity: done.every(x => x.teachersFrozen && x.synapsesUnchanged),
  seeds: seeds.map(x => ({ file: x.file, status: x.status, teacherWins: x.teacherWins, staleTeacherWins: x.staleTeacherWins ?? null,
    verdict: x.verdict ?? null, points: x.points ?? null })) };
await writeFile(new URL('../data/trust_gate_cohort_summary.json', import.meta.url), JSON.stringify(result, null, 1) + '\n');
console.log(JSON.stringify({ ...result, seeds: undefined }));
