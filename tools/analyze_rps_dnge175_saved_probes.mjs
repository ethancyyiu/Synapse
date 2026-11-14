// Read-only paired analysis of the archived DNge175 teaching-readiness probes.
// node tools/analyze_rps_dnge175_saved_probes.mjs
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const data = JSON.parse(readFileSync(new URL('../data/rps_dnge175_teaching.json', import.meta.url)));
assert.equal(data.status, 'stopped_before_training');
assert.equal(data.smokePassed, false);
const rows = data.smokeRows.filter(r => !r.firstRound);
assert.equal(rows.length, 324);
const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;
const round = x => Number(x.toFixed(3));
const bySeed = [741, 742, 743].map(seed => {
  const groups = [0, 1, 2].map(cue => {
    const paired = Array.from({ length: 12 }, (_, repeat) => {
      const match = rows.filter(r => r.seed === seed && r.cue === cue && r.repeat === repeat);
      assert.equal(match.length, 3);
      const get = condition => {
        const found = match.filter(r => r.condition === condition);
        assert.equal(found.length, 1);
        return found[0];
      };
      const none = get('none'), bit0 = get('bit0'), bit1 = get('bit1');
      assert.equal(none.key, bit0.key);
      assert.equal(none.key, bit1.key);
      assert.deepEqual(none.dnCounts, [0, 0]);
      return { repeat, none, bit0, bit1 };
    });
    const noneCorrect = paired.filter(p => p.none.cuePrediction === cue).length;
    const conditions = Object.fromEntries(['bit0', 'bit1'].map((condition, bit) => {
      const correct = paired.filter(p => p[condition].cuePrediction === cue).length;
      const newErrors = paired.filter(p => p.none.cuePrediction === cue &&
        p[condition].cuePrediction !== cue).map(p => p.repeat);
      const rescued = paired.filter(p => p.none.cuePrediction !== cue &&
        p[condition].cuePrediction === cue).map(p => p.repeat);
      const deltaRates = [0, 1, 2].map(j => round(mean(paired.map(p =>
        p[condition].rates[j] - p.none.rates[j]))));
      const targetCounts = paired.map(p => p[condition].dnCounts[bit]);
      const otherCounts = paired.map(p => p[condition].dnCounts[1 - bit]);
      return [condition, { correct, newErrors, rescued, deltaRates,
        meanDnCounts: round(mean(targetCounts)), minTargetCount: Math.min(...targetCounts),
        maxOtherCount: Math.max(...otherCounts), bitSeparations: paired.filter(p =>
          p[condition].dnCounts[bit] > p[condition].dnCounts[1 - bit]).length,
        actionChanges: paired.filter(p => p[condition].action !== p.none.action).length }];
    }));
    return { cue, noneCorrect, conditions };
  });
  const saved = data.smokeBySeed.find(s => s.seed === seed);
  assert.equal(saved.checks.weightsUnchanged, true);
  const all = rows.filter(r => r.seed === seed);
  assert.equal(all.length, 108);
  return { seed, groups, weightsUnchanged: saved.checks.weightsUnchanged,
    frozenActions: [...new Set(all.map(r => r.action))],
    frozenScoresAllZero: all.every(r => r.scores.every(x => x === 0)) };
});
console.log(JSON.stringify({ source: 'data/rps_dnge175_teaching.json',
  simulatedAgain: false, bySeed }, null, 2));
