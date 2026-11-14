// Locked analysis from docs/engine-teacher-scaling-protocol.md.
// node tools/analyze_engine_teacher_scaling.mjs
import { readFile, readdir, writeFile } from 'node:fs/promises';

const dir = new URL('../data/engine_teacher_scaling/', import.meta.url);
const files = (await readdir(dir)).filter(name => /^seed_\d+\.json$/.test(name))
  .sort((a, b) => parseInt(a.slice(5)) - parseInt(b.slice(5)));
const seeds = [];
for (const name of files) seeds.push({ file: name, ...JSON.parse(await readFile(new URL(name, dir))) });
const BAR = 58, arms = ['simConstants', 'engineConstants'];
const mean = list => list.reduce((a, b) => a + b, 0) / list.length;
const passes = Object.fromEntries(arms.map(arm => [arm, seeds.filter(x => x.wins[arm] >= BAR).length]));
const sameConstants = seeds.every(x => JSON.stringify(x.engineConstants) === JSON.stringify(seeds[0].engineConstants));
const result = { protocol: 'docs/engine-teacher-scaling-protocol.md', seeds: seeds.length, bar: BAR, passes,
  supported: seeds.length === 15 && passes.engineConstants >= 13 && passes.engineConstants > passes.simConstants,
  meanWins: Object.fromEntries(arms.map(arm => [arm, mean(seeds.map(x => x.wins[arm]))])),
  simConstants: seeds[0].simConstants, engineConstants: seeds[0].engineConstants, sameConstantsOnEverySeed: sameConstants,
  perSeed: seeds.map(x => ({ file: x.file, teacherSeed: x.teacherSeed, ...x.wins })) };
await writeFile(new URL('../data/engine_teacher_scaling_summary.json', import.meta.url), JSON.stringify(result, null, 1) + '\n');
console.log(JSON.stringify(result));
