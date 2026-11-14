// Exact balanced-label control for archived, frozen KC predictions.
// node tools/check_two_cue_population_kc_permutation.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';

const input = new URL('../data/two_cue_population_kc_fixed_rng.json', import.meta.url);
const output = new URL('../data/two_cue_population_kc_permutation.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite permutation result');
const archived = JSON.parse(readFileSync(input));
const choose = (n, k) => {
  if (k < 0 || k > n) return 0;
  let result = 1;
  for (let i = 1; i <= Math.min(k, n - k); i++) result = result * (n - i + 1) / i;
  return Math.round(result);
};
const evaluation = Object.fromEntries([505, 506].map(seed => {
  const trials = archived.evaluation[seed].trials;
  if (trials.length !== 24 || trials.filter(r => r.mapping === 1).length !== 12 ||
      trials.some(r => ![0, 1].includes(r.prediction)) ||
      trials.filter(r => r.mapping === r.prediction).length !== archived.evaluation[seed].trained.score)
    throw Error(`Archived prediction mismatch: ${seed}`);
  const predictedOne = trials.filter(r => r.prediction === 1).length;
  const observed = archived.evaluation[seed].trained.score;
  const distribution = [];
  for (let overlap = 0; overlap <= 12; overlap++) {
    const count = choose(predictedOne, overlap) * choose(24 - predictedOne, 12 - overlap);
    if (count) distribution.push({ score: 12 - predictedOne + 2 * overlap, count });
  }
  const total = choose(24, 12);
  if (distribution.reduce((sum, row) => sum + row.count, 0) !== total) throw Error('Count mismatch');
  const tailCount = distribution.filter(row => row.score >= observed)
    .reduce((sum, row) => sum + row.count, 0);
  return [seed, { observed, predictedOne, shuffledScore: archived.evaluation[seed].shuffledLabels.score,
    totalBalancedAssignments: total, tailCount, oneSidedP: tailCount / total, distribution }];
}));
const result = { command: 'node tools/check_two_cue_population_kc_permutation.mjs',
  source: 'data/two_cue_population_kc_fixed_rng.json',
  hypothesis: 'The archived KC prediction score is unusual under all balanced label assignments',
  gate: 'one-sided P(score >= observed 20/24) <= 0.05 for each teacher',
  evaluation, passed: [505, 506].every(seed => evaluation[seed].oneSidedP <= .05) };
writeFileSync(output, JSON.stringify(result, null, 2));
console.log(JSON.stringify({ passed: result.passed, evaluation: Object.fromEntries([505, 506]
  .map(seed => [seed, { observed: evaluation[seed].observed, shuffledScore: evaluation[seed].shuffledScore,
    tailCount: evaluation[seed].tailCount, total: evaluation[seed].totalBalancedAssignments,
    oneSidedP: evaluation[seed].oneSidedP }])) }, null, 2));
