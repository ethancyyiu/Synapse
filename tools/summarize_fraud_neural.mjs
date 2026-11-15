// Aggregate only the five predeclared final seeds; never rerun or select seeds.
import { readFile, open } from 'node:fs/promises';

const seeds = [];
for (const seed of [420, 421, 422, 423, 424]) {
  const result = JSON.parse(await readFile(new URL(`../data/fraud_neural/seed_${seed}.json`, import.meta.url)));
  if (result.seed !== seed || !['completed', 'stopped_teacher'].includes(result.status))
    throw Error(`Invalid result for seed ${seed}`);
  seeds.push(result);
}
const completed = seeds.filter(s => s.status === 'completed');
const advantages = completed.map(s => s.advantage48);
const meanAdvantage48 = advantages.length ? advantages.reduce((a, b) => a + b, 0) / advantages.length : null;
const positiveAdvantageSeeds = advantages.filter(x => x > 0).length;
const totalMessages = completed.reduce((sum, s) => sum + Object.values(s.messages).reduce((a, b) => a + b, 0), 0);
const totalDecoded = completed.reduce((sum, s) => sum + Object.values(s.decoded).reduce((a, b) => a + b, 0), 0);
const successGatePassed = completed.length >= 4 && meanAdvantage48 >= 5 && positiveAdvantageSeeds >= 4 &&
  totalMessages === totalDecoded && completed.every(s => s.teacherFrozen && s.synapsesUnchanged);
const summary = { protocol: 'docs/synapse-fraud-neural-protocol.md', seedSet: [420, 421, 422, 423, 424],
  threshold: 58, checkpointRounds: [0, 24, 48, 96], evaluationTrials: 72,
  completedSeeds: completed.length, positiveAdvantageSeeds, meanAdvantage48,
  totalMessages, totalDecoded, successGatePassed,
  seeds: seeds.map(s => ({ seed: s.seed, status: s.status, teacherWins: s.teacherWins,
    teacherGate: s.teacherGate, teacherEpisodeWins: s.teacherEpisodeWins,
    points: s.points ?? null, recoveryAt: s.recoveryAt ?? null, advantage48: s.advantage48 ?? null,
    messages: s.messages ?? null, decoded: s.decoded ?? null,
    spikeCounts: s.spikeCounts ?? null, teacherFrozen: s.teacherFrozen ?? null,
    synapsesUnchanged: s.synapsesUnchanged ?? null })) };
const file = await open(new URL('../data/fraud_neural/summary.json', import.meta.url), 'wx');
try { await file.writeFile(JSON.stringify(summary, null, 2) + '\n'); }
finally { await file.close(); }
console.log(JSON.stringify({ successGatePassed, completedSeeds: completed.length,
  meanAdvantage48, positiveAdvantageSeeds, totalMessages, totalDecoded,
  results: summary.seeds.map(s => ({ seed: s.seed, status: s.status, at48: s.points?.find(p => p.rounds === 48),
    recoveryAt: s.recoveryAt, advantage48: s.advantage48 })) }));
