// Recomputes every headline number of the locked cohorts from the per-seed result files, then checks that
// (1) the committed summary files agree and (2) the README and the page quote those numbers. No simulation; runs in
// under a second. Exits 1 on any mismatch.
// node tools/verify_claims.mjs
import { readFile, readdir } from 'node:fs/promises';
import { rng } from '../js/sim.js';

const root = new URL('../', import.meta.url);
const json = async path => JSON.parse(await readFile(new URL(path, root)));
const text = async path => (await readFile(new URL(path, root), 'utf8')).replace(/\s+/g, ' ');
const mean = list => list.reduce((a, b) => a + b, 0) / list.length;
const bootstrap = (values, seed) => {
  const random = rng(seed), boots = [];
  for (let i = 0; i < 10000; i++) boots.push(mean(values.map(() => values[Math.floor(random() * values.length)])));
  boots.sort((a, b) => a - b);
  return [boots[249], boots[9749]];
};
const completed = async folder => {
  const names = (await readdir(new URL(`data/${folder}/`, root))).filter(name => /^seed_\d+\.json$/.test(name))
    .sort((a, b) => parseInt(a.slice(5)) - parseInt(b.slice(5))); // the locked analyses resample in seed order
  const seeds = [];
  for (const name of names) seeds.push(await json(`data/${folder}/${name}`));
  return seeds.filter(x => x.status === 'completed');
};
const at = (x, rounds) => x.points.find(p => p.rounds === rounds);
const contrast = (seeds, a, b, rounds, seed) => {
  const values = seeds.map(x => at(x, rounds)[a] - at(x, rounds)[b]);
  return { mean: mean(values), ci: bootstrap(values, seed) };
};
const signed = x => (x >= 0 ? '+' : '') + x.toFixed(1);
const range = ci => `${signed(ci[0])} to ${signed(ci[1])}`;

const failures = [], rows = [];
const same = (label, got, want) => {
  const ok = Math.abs(got - want) < 1e-9;
  if (!ok) failures.push(`${label}: recomputed ${got}, summary file has ${want}`);
  return ok;
};
const readme = await text('README.md'), page = await text('index.html');
const quoted = (label, strings, where = { README: readme, 'index.html': page }) => {
  for (const [name, body] of Object.entries(where)) for (const s of strings)
    if (!body.includes(s)) failures.push(`${label}: "${s}" not found in ${name}`);
};
const report = (study, claim, result, passed) => rows.push({ study, claim, value: `${signed(result.mean)} (${range(result.ci)})`, holds: passed });

// 1. Rule-switch cohort: lead over the best control at 48 rounds.
{
  const seeds = await completed('rule_switch_cohort'), summary = await json('data/rule_switch_cohort_summary.json');
  const values = seeds.map(x => at(x, 48).trained - Math.max(...['alone', 'random', 'untrained', 'shuffled'].map(arm => at(x, 48)[arm])));
  const result = { mean: mean(values), ci: bootstrap(values, 2026110905) };
  same('rule-switch mean', result.mean, summary.meanAdvantage48);
  same('rule-switch CI low', result.ci[0], summary.bootstrap95[0]); same('rule-switch CI high', result.ci[1], summary.bootstrap95[1]);
  for (const arm of ['alone', 'trained', 'random', 'untrained', 'shuffled'])
    same(`rule-switch mean wins ${arm}`, mean(seeds.map(x => at(x, 48)[arm])), summary.meanWins48[arm]);
  quoted('rule-switch', [signed(result.mean), range(result.ci)]);
  report('Rule switch (15 runs)', 'taught - best control, 48 rounds', result, seeds.length >= 10 && result.ci[0] > 4);
  if (summary.passed !== (seeds.length >= 10 && result.ci[0] > 4)) failures.push('rule-switch: pass flag disagrees');
}
// 2. Stale-teacher cohort: new-rule teacher against old-rule teacher at 48 rounds.
{
  const seeds = await completed('stale_teacher_cohort'), summary = await json('data/stale_teacher_cohort_summary.json');
  const result = contrast(seeds, 'trained', 'stale', 48, 2026111001);
  same('stale mean', result.mean, summary.meanTrainedMinusStale48);
  same('stale CI low', result.ci[0], summary.bootstrap95[0]); same('stale CI high', result.ci[1], summary.bootstrap95[1]);
  quoted('stale teacher', [signed(result.mean), range(result.ci)]);
  report('Out-of-date teacher (15 runs)', 'new-rule - old-rule teacher, 48 rounds', result, seeds.length >= 10 && result.ci[0] > 4);
}
// 3. Trust-gate cohort: three locked tests.
{
  const seeds = await completed('trust_gate_cohort'), summary = await json('data/trust_gate_cohort_summary.json');
  const protection = contrast(seeds, 'gatedStale', 'stale', 96, 2026111101);
  const noHarm = contrast(seeds, 'gatedStale', 'alone', 96, 2026111102);
  const kept = contrast(seeds, 'gatedTrained', 'trained', 48, 2026111103);
  for (const [name, result, key] of [['protection', protection, 'protection96'], ['no harm', noHarm, 'noHarm96'], ['benefit kept', kept, 'benefitKept48']]) {
    same(`trust ${name} mean`, result.mean, summary[key].mean);
    same(`trust ${name} CI low`, result.ci[0], summary[key].bootstrap95[0]); same(`trust ${name} CI high`, result.ci[1], summary[key].bootstrap95[1]);
  }
  quoted('trust gate', [signed(protection.mean), range(protection.ci), (-noHarm.mean).toFixed(1)]);
  report(`Trust gate (${seeds.length} runs)`, 'gated - ungated, old-rule teacher, 96 rounds', protection, protection.ci[0] > 4);
  report(`Trust gate (${seeds.length} runs)`, 'gated old-rule - alone, 96 rounds', noHarm, noHarm.ci[0] > -4);
  report(`Trust gate (${seeds.length} runs)`, 'gated - ungated, new-rule teacher, 48 rounds', kept, kept.ci[0] > -4);
  const pass = seeds.length >= 10 && protection.ci[0] > 4 && noHarm.ci[0] > -4 && kept.ci[0] > -4;
  if (summary.passed !== pass) failures.push('trust gate: pass flag disagrees');
  if (pass) failures.push('trust gate: recomputed as passed, but the README reports it as failed');
}
// 4. Two-teacher cohort: two locked tests at 96 rounds.
{
  const seeds = await completed('two_teacher_cohort'), summary = await json('data/two_teacher_cohort_summary.json');
  const choosing = contrast(seeds, 'bothGated', 'bothUngated', 96, 2026111111);
  const cost = contrast(seeds, 'bothGated', 'trained', 96, 2026111112);
  for (const [name, result, key] of [['choosing', choosing, 'choosingHelps96'], ['cost', cost, 'badAdviserCost96']]) {
    same(`two-teacher ${name} mean`, result.mean, summary[key].mean);
    same(`two-teacher ${name} CI low`, result.ci[0], summary[key].bootstrap95[0]); same(`two-teacher ${name} CI high`, result.ci[1], summary[key].bootstrap95[1]);
  }
  quoted('two teachers', [signed(choosing.mean), range(choosing.ci)]);
  report(`Two teachers (${seeds.length} runs)`, 'with trust - followed equally, 96 rounds', choosing, choosing.ci[0] > 4);
  report(`Two teachers (${seeds.length} runs)`, 'with trust - good teacher only, 96 rounds', cost, cost.ci[0] > -4);
  const pass = seeds.length >= 10 && choosing.ci[0] > 4 && cost.ci[0] > -4;
  if (summary.passed !== pass) failures.push('two teachers: pass flag disagrees');
  if (pass) failures.push('two teachers: recomputed as passed, but the README reports it as failed');
}
// 5. Fault-response cohort: lead over the best control at 48 rounds on the second task.
{
  const seeds = await completed('fault_response_cohort'), summary = await json('data/fault_response_cohort_summary.json');
  const values = seeds.map(x => at(x, 48).trained - Math.max(...['alone', 'random', 'untrained', 'shuffled'].map(arm => at(x, 48)[arm])));
  const result = { mean: mean(values), ci: bootstrap(values, 2026111130) };
  same('fault-response mean', result.mean, summary.advantage48.mean);
  same('fault-response CI low', result.ci[0], summary.advantage48.bootstrap95[0]); same('fault-response CI high', result.ci[1], summary.advantage48.bootstrap95[1]);
  quoted('fault response', [signed(result.mean), range(result.ci)]);
  const pass = seeds.length >= 10 && result.ci[0] > 4;
  report(`Second task (${seeds.length} runs)`, 'taught - best control, 48 rounds', result, pass);
  if (summary.passed !== pass) failures.push('fault response: pass flag disagrees');
}
// 6. Rule-switch cohort repeated on js/engine.js.
{
  const seeds = await completed('engine_replication_cohort'), summary = await json('data/engine_replication_cohort_summary.json');
  const values = seeds.map(x => at(x, 48).trained - Math.max(...['alone', 'random', 'untrained', 'shuffled'].map(arm => at(x, 48)[arm])));
  const result = { mean: mean(values), ci: bootstrap(values, 2026111150) };
  same('engine mean', result.mean, summary.advantage48.mean);
  same('engine CI low', result.ci[0], summary.advantage48.bootstrap95[0]); same('engine CI high', result.ci[1], summary.advantage48.bootstrap95[1]);
  same('engine completed runs', seeds.length, summary.completed);
  quoted('own engine', [signed(result.mean), range(result.ci), '10 of 15']);
  const pass = seeds.length >= 10 && result.ci[0] > 4;
  report(`Own engine (${seeds.length} of ${summary.seedsFound} runs)`, 'taught - best control, 48 rounds', result, pass);
  if (summary.passed !== pass) failures.push('own engine: pass flag disagrees');
}

console.table(rows);
if (failures.length) { console.error(failures.map(f => 'MISMATCH  ' + f).join('\n')); process.exit(1); }
console.log(`claims verified: ${rows.length} locked tests recomputed from seed files; summaries, README, and page agree`);
