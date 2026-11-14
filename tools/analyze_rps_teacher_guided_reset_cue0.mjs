// Read-only, write-once algebraic audit of the saved reset-schedule teaching preflight.
// node tools/analyze_rps_teacher_guided_reset_cue0.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';

const output = new URL('../data/rps_teacher_guided_reset_cue0_audit.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite cue-0 audit');
const source = readFileSync(new URL('../data/rps_teacher_guided_reset_preflight.json', import.meta.url));
const data = JSON.parse(source);
if (data.stage !== 4 || data.status !== 'failed_retained_learning' || data.students.length !== 2)
  throw Error('Saved preflight is missing or changed');
const arms = ['correct', 'shuffled', 'untrained', 'disconnected', 'lookup'];
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const close = (a, b) => Math.abs(a - b) < 1e-9;
const mean = xs => xs.reduce((s, x) => s + x, 0) / xs.length;
const stats = xs => {
  const sorted = [...xs].sort((a, b) => a - b), m = mean(xs);
  return { n: xs.length, mean: m, sd: Math.sqrt(mean(xs.map(x => (x - m) ** 2))),
    min: sorted[0], median: sorted[Math.floor((sorted.length - 1) / 2)], max: sorted.at(-1) };
};
const counts = xs => [0, 1, 2].map(a => xs.filter(x => x === a).length);
const dot = (a, b) => a.reduce((s, x, j) => s + x * b[j], 0);
const q = (params, f) => params.W.map((w, a) => params.b[a] + dot(w, f));
const cueKey = cue => cue === null ? 'none' : String(cue);
const cueKeys = ['none', '0', '1', '2'];
function trainSummary(rows) {
  const byCue = Object.fromEntries(cueKeys.map(key => {
    const x = rows.filter(r => cueKey(r.cue) === key);
    return [key, { n: x.length, provisional: counts(x.map(r => r.candidate)),
      final: counts(x.map(r => r.action)),
      rewarded: counts(x.filter(r => r.reward === 1).map(r => r.action)),
      rewardedWeightChanges: counts(x.filter(r =>
        r.update.rewardedWeightChange).map(r => r.action)),
      allWeightChanges: counts(x.filter(r => Math.abs(r.update.step) > 1e-12).map(r => r.action)) }];
  }));
  return { byCue,
    changed: rows.filter(r => r.candidate !== r.action).length,
    signedStepByAction: [0, 1, 2].map(a => rows.filter(r => r.action === a)
      .reduce((s, r) => s + r.update.step, 0)),
    absoluteStepByAction: [0, 1, 2].map(a => rows.filter(r => r.action === a)
      .reduce((s, r) => s + Math.abs(r.update.step), 0)) };
}
function scoreContributions(trainRows, evalRows) {
  const contributions = Object.fromEntries(cueKeys.map(key => {
    const sourceRows = trainRows.filter(r => cueKey(r.cue) === key);
    const values = [0, 1, 2].map(a => mean(evalRows.map(e => sourceRows
      .filter(r => r.action === a)
      .reduce((sum, r) => sum + r.update.step * (1 + dot(r.features, e.features)), 0))));
    return [key, { scores: values, rockMinusPaper: values[0] - values[1],
      rockMinusScissors: values[0] - values[2] }];
  }));
  return contributions;
}
const students = [];
for (const student of data.students) {
  if (![831, 832].includes(student.seed) || !arms.every(a => student.training[a] && student.final[a]))
    throw Error('Student or arm missing');
  const reference = student.final.correct.rows.filter(r => r.cue === 0);
  if (reference.length !== 23 || !student.initial.W.flat().every(v => v === 0) ||
      !student.initial.b.every(v => v === 0)) throw Error('Initial or cue-0 records invalid');
  const features = { standardized: [0, 1, 2].map(j => stats(reference.map(r => r.features[j]))),
    rawRates: [0, 1, 2].map(j => stats(reference.map(r => r.rates[j]))),
    fixedDecoderCorrect: reference.filter(r => r.cueGuess === 0).length };
  const result = { seed: student.seed, features, arms: {}, crossController: {} };
  for (const arm of arms) {
    const train = student.training[arm], frozen = student.final[arm];
    const cue0 = frozen.rows.filter(r => r.cue === 0);
    if (train.rows.length !== 24 || frozen.rows.length !== 72 || cue0.length !== 23 ||
        !same(cue0.map(r => [r.sequenceSeed, r.round, r.opponent, r.features, r.rates]),
          reference.map(r => [r.sequenceSeed, r.round, r.opponent, r.features, r.rates])))
      throw Error(`Unmatched cue-0 neural inputs: ${student.seed}/${arm}`);
    const reconstructed = { W: [0, 1, 2].map(() => [0, 0, 0]), b: [0, 0, 0] };
    for (const r of train.rows) {
      const a = r.update.selectedAction;
      if (a !== r.action || r.features.length !== 3 || !Number.isFinite(r.update.step))
        throw Error('Missing selected-action update');
      reconstructed.b[a] += r.update.step;
      for (let j = 0; j < 3; j++) reconstructed.W[a][j] += r.update.step * r.features[j];
    }
    if (!train.after.W.every((w, a) => w.every((v, j) => close(v, reconstructed.W[a][j]))) ||
        !train.after.b.every((v, a) => close(v, reconstructed.b[a])))
      throw Error(`Saved updates do not reconstruct final Q: ${student.seed}/${arm}`);
    for (const r of frozen.rows) {
      const scores = q(train.after, r.features);
      if (!scores.every((v, a) => close(v, r.scores[a])) ||
          r.action !== scores.indexOf(Math.max(...scores)))
        throw Error(`Frozen score/action mismatch: ${student.seed}/${arm}`);
    }
    const scores = cue0.map(r => r.scores);
    const margins = scores.map(s => s[0] - Math.max(s[1], s[2]));
    const wrongGaps = cue0.filter(r => r.action !== 0).map(r =>
      Math.max(r.scores[1], r.scores[2]) - r.scores[0]);
    const bySourceCue = scoreContributions(train.rows, cue0);
    const meanScores = [0, 1, 2].map(a => mean(scores.map(s => s[a])));
    if (![0, 1, 2].every(a => close(meanScores[a], cueKeys.reduce((s, key) =>
      s + bySourceCue[key].scores[a], 0))))
      throw Error(`Cue-source decomposition mismatch: ${student.seed}/${arm}`);
    result.arms[arm] = {
      frozenCue0: { n: cue0.length, actions: counts(cue0.map(r => r.action)),
        correct: cue0.filter(r => r.action === 0).length,
        q: [0, 1, 2].map(a => stats(scores.map(s => s[a]))),
        rockMargin: stats(margins), wrongGap: wrongGaps.length ? stats(wrongGaps) : null,
        wrongWinner: counts(cue0.filter(r => r.action !== 0).map(r => r.action)) },
      training: trainSummary(train.rows),
      parameters: { W: train.after.W, b: train.after.b,
        weightNorm: train.after.W.map(w => Math.sqrt(dot(w, w))) },
      cue0ScoreContributionByTrainingCue: bySourceCue,
      targetRewardedUpdates: train.targetRewardedUpdates,
      totalFrozenWins: frozen.summary.wins };
  }
  const correct = student.final.correct.rows.filter(r => r.cue === 0);
  for (const arm of arms.filter(a => a !== 'correct')) {
    const other = student.final[arm].rows.filter(r => r.cue === 0);
    result.crossController[arm] = { selectedActionDiffers: correct.filter((r, i) =>
      r.action !== other[i].action).length,
      correctOnlyRock: correct.filter((r, i) => r.action === 0 && other[i].action !== 0).length,
      otherOnlyRock: correct.filter((r, i) => r.action !== 0 && other[i].action === 0).length,
      bothRock: correct.filter((r, i) => r.action === 0 && other[i].action === 0).length };
  }
  students.push(result);
}
const derived = { source: 'data/rps_teacher_guided_reset_preflight.json',
  command: 'node tools/analyze_rps_teacher_guided_reset_cue0.mjs',
  method: 'saved trials only; exact matched frozen cue-0 MBON vectors; Q=b+W*f; score contributions reconstructed from recorded selected-action updates',
  students };
writeFileSync(output, `${JSON.stringify(derived, null, 2)}\n`);
console.log(JSON.stringify(students.map(s => ({ seed: s.seed,
  cue0: Object.fromEntries(arms.map(a => [a, {
    correct: s.arms[a].frozenCue0.correct,
    meanMargin: s.arms[a].frozenCue0.rockMargin.mean,
    rewardedRockUpdates: s.arms[a].training.byCue['0'].rewardedWeightChanges[0]
  }])) })), null, 2));
