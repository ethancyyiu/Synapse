// Read-only attribution of the archived 841/842 early cue-head experiment.
// node tools/analyze_rps_teacher_guided_cue_heads_early.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import assert from 'node:assert/strict';

const load = name => JSON.parse(readFileSync(new URL(`../data/${name}.json`, import.meta.url)));
const partial = load('rps_teacher_guided_cue_heads_early');
const complete = load('rps_teacher_guided_cue_heads_early_continuation');
const earlier = load('rps_teacher_guided_cue_heads_attribution');
const output = new URL('../data/rps_teacher_guided_cue_heads_early_attribution.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite derived attribution');
const modes = ['correct', 'shuffled', 'untrained', 'disconnected', 'lookup'];
const action = ['rock', 'paper', 'scissors'];
const count = (rows, pred) => rows.filter(pred).length;
const byAction = (rows, field) => [0, 1, 2].map(a => count(rows, r => r[field] === a));
const correctAction = r => (r.opponent + 1) % 3;
const mean = x => x.reduce((a, b) => a + b, 0) / x.length;
const q = (head, features) => head.W.map((w, a) =>
  head.b[a] + w.reduce((sum, value, j) => sum + value * features[j], 0));
const close = (a, b) => Math.abs(a - b) < 1e-9;

assert.equal(partial.status, 'stopped_invalid_shuffle');
assert.equal(complete.status, 'failed_or_invalid_early_teaching');
assert.deepEqual(complete.students.map(s => s.seed), [841, 842]);
assert.deepEqual(earlier.students.map(s => s.seed), [831, 832]);
assert.deepEqual(partial.students[0].training, complete.students[0].training);
for (const mode of ['correct', 'shuffled'])
  assert.deepEqual(partial.students[1].training[mode], complete.students[1].training[mode]);

function trainingStats(rows) {
  assert([12, 24].includes(rows.length));
  assert.deepEqual(rows.map(r => r.round), Array.from({ length: rows.length }, (_, i) => i));
  const observed = rows.filter(r => r.cue !== null);
  assert.equal(observed.length, rows.length - 1);
  const message = observed.filter(r => r.emittedBit !== null);
  for (const r of observed) {
    assert.equal(r.selectedHead, r.cue);
    assert.equal(r.reward === 1, r.action === correctAction(r));
    if (r.emittedBit !== null) {
      assert.equal(r.decodedBit, r.emittedBit);
      assert(r.dnCounts[r.emittedBit] > r.dnCounts[1 - r.emittedBit]);
      if (r.emittedBit === 1) assert.equal(r.action, r.candidate);
      if (r.emittedBit === 0) assert.notEqual(r.action, r.candidate);
    } else assert.equal(r.action, r.candidate);
  }
  const byCue = [0, 1, 2].map(cue => {
    const xs = observed.filter(r => r.cue === cue);
    const firstReward = xs.find(r => r.reward === 1 && r.update.rewardedWeightChange);
    return { cue, n: xs.length, provisionalActions: byAction(xs, 'candidate'),
      playedActions: byAction(xs, 'action'), correctProposals: count(xs, r => r.candidate === correctAction(r)),
      correctPlayed: count(xs, r => r.action === correctAction(r)),
      rewardedUpdatesByAction: [0, 1, 2].map(a => count(xs, r =>
        r.action === a && r.reward === 1 && r.update.rewardedWeightChange)),
      firstCorrectReward: firstReward ? { round: firstReward.round,
        candidate: action[firstReward.candidate], played: action[firstReward.action],
        changedByMessage: firstReward.action !== firstReward.candidate,
        emittedBit: firstReward.emittedBit } : null };
  });
  const changed = observed.filter(r => r.action !== r.candidate);
  return { n: observed.length, firstCueFreeRound: { action: action[rows[0].action],
      reward: rows[0].reward },
    provisionalActions: byAction(observed, 'candidate'), playedActions: byAction(observed, 'action'),
    approvals: count(message, r => r.emittedBit === 1),
    rejections: count(message, r => r.emittedBit === 0),
    noMessage: observed.length - message.length,
    changed: changed.length,
    helpfulChanges: count(changed, r => r.candidate !== correctAction(r) &&
      r.action === correctAction(r)),
    harmfulChanges: count(changed, r => r.candidate === correctAction(r) &&
      r.action !== correctAction(r)),
    neutralWrongChanges: count(changed, r => r.candidate !== correctAction(r) &&
      r.action !== correctAction(r)),
    unchanged: observed.length - changed.length,
    correctProposals: count(observed, r => r.candidate === correctAction(r)),
    correctPlayed: count(observed, r => r.action === correctAction(r)),
    byCue };
}
function checkpointStats(checkpoint, params) {
  assert.equal(checkpoint.rows.length, 72);
  assert.equal(checkpoint.weightsUnchanged, true);
  const byCue = [0, 1, 2].map(cue => {
    const rows = checkpoint.rows.filter(r => r.cue === cue);
    assert.equal(rows.length, 23);
    const margins = rows.map(r => {
      const head = params.heads[cue], scores = q(head, r.features);
      assert(scores.every((v, i) => close(v, r.scores[i])));
      assert.equal(r.action, r.scores.indexOf(Math.max(...r.scores)));
      const correct = correctAction(r);
      return scores[correct] - Math.max(...scores.filter((_, i) => i !== correct));
    });
    return { cue, wins: count(rows, r => r.reward === 1),
      correctMarginMean: mean(margins), correctMarginMin: Math.min(...margins),
      correctMarginMax: Math.max(...margins), positiveMarginTrials: margins.filter(x => x > 0).length };
  });
  return { wins: checkpoint.summary.wins, actions: checkpoint.summary.actions,
    winsByAction: checkpoint.summary.winsByAction, byCue };
}

const derived = { source: [
  'data/rps_teacher_guided_cue_heads_early.json',
  'data/rps_teacher_guided_cue_heads_early_continuation.json',
  'data/rps_teacher_guided_cue_heads_attribution.json'],
  command: 'node tools/analyze_rps_teacher_guided_cue_heads_early.mjs',
  scope: 'read-only trial and saved-checkpoint analysis; no neural simulation or training',
  limitations: [
    'No frozen evaluations or full controller snapshots were saved after individual training rounds 1–11; the exact first margin-crossing round cannot be determined.',
    'A positive round-12 margin after one rewarded correct-action update is association, not proof that that update alone caused dominance.',
    'Seed 842 shuffled advice violates its locked <=60% quality limit; it is descriptive, not a valid shuffled control.',
    'Saved arms do not match action-change frequency or action distributions; causal attribution to teacher information versus exploration remains unresolved.' ],
  students: [] };
for (const student of complete.students) {
  const out = { seed: student.seed,
    shuffle: { moved: student.shuffle.moved, quality: student.shuffle.quality,
      valid: student.shuffle.valid }, arms: {} };
  for (const mode of modes) {
    const arm = student.training[mode];
    assert.deepEqual(arm.before, student.initialParams);
    assert.equal(arm.checkpointNoninvasive, true);
    assert.equal(arm.synapsesUnchanged, true);
    assert.equal(arm.middle.rows.length, 72);
    assert.deepEqual(arm.middle, student.middle[mode]);
    const first12 = arm.rows.slice(0, 12);
    const first24 = arm.rows;
    out.arms[mode] = { training12: trainingStats(first12), training24: trainingStats(first24),
      frozen12: checkpointStats(student.middle[mode], arm.middleParams),
      frozen24: checkpointStats(student.final[mode], arm.after) };
    for (const cue of [0, 1, 2]) {
      const rewards = out.arms[mode].training12.byCue[cue].rewardedUpdatesByAction[cue];
      const margin = out.arms[mode].frozen12.byCue[cue].correctMarginMean;
      out.arms[mode].training12.byCue[cue].oneRewardPositiveAt12 = rewards === 1 && margin > 0;
      if (rewards === 1) {
        const selected = first12.filter(r => r.cue === cue && r.action === cue);
        assert.equal(selected.length, 1);
        assert.equal(selected[0].reward, 1);
        const head = arm.middleParams.heads[cue], step = selected[0].update.step;
        assert(close(head.b[cue], step));
        assert(head.W[cue].every((v, j) => close(v, step * selected[0].features[j])));
      }
    }
  }
  derived.students.push(out);
}
writeFileSync(output, `${JSON.stringify(derived, null, 2)}\n`);
console.log(JSON.stringify(derived.students.map(s => ({ seed: s.seed, shuffle: s.shuffle,
  arms: Object.fromEntries(modes.map(mode => {
    const a = s.arms[mode];
    return [mode, { proposals: a.training12.provisionalActions,
      played: a.training12.playedActions,
      approvals: a.training12.approvals, rejections: a.training12.rejections,
      changed: a.training12.changed, helpful: a.training12.helpfulChanges,
      harmful: a.training12.harmfulChanges, neutralWrong: a.training12.neutralWrongChanges,
      correctPlayed: a.training12.correctPlayed,
      rewardsByCueAction: a.training12.byCue.map(x => x.rewardedUpdatesByAction),
      firstCorrectReward: a.training12.byCue.map(x => x.firstCorrectReward),
      frozen12: a.frozen12.wins,
      margin12: a.frozen12.byCue.map(x => Number(x.correctMarginMean.toFixed(3))) }];
  })) })), null, 2));
