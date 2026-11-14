// Quantify whether the saved messages predict the next move beyond the previous move.
// Run: node tools/rps_signal_information.mjs
import { readFileSync, writeFileSync } from 'node:fs';
import { Opponent } from '../js/rps.js';
import { rng } from '../js/sim.js';

const results = JSON.parse(readFileSync(new URL('../data/rps_results.json', import.meta.url)));
const CONDITIONS = ['trained', 'untrained', 'shuffled_signal'];
const PERMUTATIONS = 200;

function entropy(values) {
  const counts = new Map();
  for (const value of values) counts.set(value, (counts.get(value) ?? 0) + 1);
  let bits = 0;
  for (const count of counts.values()) {
    const probability = count / values.length;
    bits -= probability * Math.log2(probability);
  }
  return bits;
}

function information(rows, x, y) {
  return entropy(rows.map(x)) + entropy(rows.map(y))
    - entropy(rows.map(row => `${x(row)}|${y(row)}`));
}

function conditionalInformation(rows) {
  return [0, 1, 2].reduce((sum, previous) => {
    const subset = rows.filter(row => row.previous === previous);
    return sum + subset.length / rows.length * information(subset,
      row => row.signal, row => row.next);
  }, 0);
}

function records(condition) {
  const rows = [];
  for (const seed of results.meta.seeds) {
    const transmitted = results.conditions[condition].runs[seed - 1].transmitted;
    for (let episode = 0; episode < results.meta.episodes; episode++) {
      const rival = new Opponent({ ...results.meta.opponent, seed: seed * 1009 + episode });
      for (let round = 0; round < results.meta.roundsPerEpisode; round++) {
        const previous = rival.observe().previousOpponent;
        const next = rival.play(0).opponent;
        if (previous !== null) rows.push({ previous, next,
          signal: transmitted[episode * results.meta.roundsPerEpisode + round].join(',') });
      }
    }
  }
  return rows;
}

function permutationReference(rows, seed) {
  const random = rng(seed), values = [];
  for (let trial = 0; trial < PERMUTATIONS; trial++) {
    const shuffled = rows.map(row => ({ ...row }));
    for (const previous of [0, 1, 2]) {
      const indices = shuffled.map((row, index) => row.previous === previous ? index : -1)
        .filter(index => index >= 0);
      const signals = indices.map(index => shuffled[index].signal);
      for (let i = signals.length - 1; i > 0; i--) {
        const j = Math.floor(random() * (i + 1));
        [signals[i], signals[j]] = [signals[j], signals[i]];
      }
      indices.forEach((index, i) => { shuffled[index].signal = signals[i]; });
    }
    values.push(conditionalInformation(shuffled));
  }
  return values.sort((a, b) => a - b);
}

const output = { meta: { permutations: PERMUTATIONS, excludedFirstRoundOfEachEpisode: true,
  meaning: 'Empirical mutual information in bits; signals shuffled within previous-move groups for reference.' },
conditions: {} };
for (const [index, condition] of CONDITIONS.entries()) {
  const rows = records(condition);
  const observed = conditionalInformation(rows);
  const nullValues = permutationReference(rows, 22 + index);
  const counts = new Map();
  for (const row of rows) counts.set(row.signal, (counts.get(row.signal) ?? 0) + 1);
  output.conditions[condition] = {
    rounds: rows.length,
    rawSignalToNextBits: information(rows, row => row.signal, row => row.next),
    conditionalBits: observed,
    permutationMedianBits: nullValues[Math.floor(PERMUTATIONS / 2)],
    permutation95thBits: nullValues[Math.floor(PERMUTATIONS * 0.95)],
    permutationP: (1 + nullValues.filter(value => value >= observed).length) / (PERMUTATIONS + 1),
    allZeroShare: (counts.get('0,0,0') ?? 0) / rows.length,
    allMaxShare: (counts.get('2,2,2') ?? 0) / rows.length,
  };
}
const rows = records('trained');
output.previousMovePredictor = { correct: rows.filter(row => (row.previous + 1) % 3 === row.next).length,
  rounds: rows.length };
writeFileSync(new URL('../data/rps_signal_information.json', import.meta.url), JSON.stringify(output, null, 2));
console.log(JSON.stringify(output, null, 2));
