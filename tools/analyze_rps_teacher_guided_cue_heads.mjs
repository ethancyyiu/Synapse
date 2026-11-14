// Saved-data-only, write-once attribution audit; no neural simulation or training.
// node tools/analyze_rps_teacher_guided_cue_heads.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { result } from '../js/rps.js';

const output = new URL('../data/rps_teacher_guided_cue_heads_attribution.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite cue-head attribution');
const read = path => JSON.parse(readFileSync(new URL(path, import.meta.url)));
const data = read('../data/rps_teacher_guided_cue_heads.json');
const shared = read('../data/rps_teacher_guided_reset_preflight.json');
const calibration = read('../data/rps_readout_information.json');
const modes = ['correct', 'shuffled', 'untrained', 'disconnected', 'lookup'];
if (data.status !== 'failed_learning_or_teaching' || shared.status !== 'failed_retained_learning' ||
    data.students.length !== 2 || calibration.centroids.length !== 3)
  throw Error('Source experiment status or fixed decoder is missing');
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const close = (a, b) => Math.abs(a - b) < 1e-9;
const count = (xs, value) => xs.filter(x => x === value).length;
const actions = xs => [0, 1, 2].map(a => count(xs, a));
const mean = xs => xs.reduce((s, x) => s + x, 0) / xs.length;
const stats = xs => ({ n: xs.length, mean: mean(xs), min: Math.min(...xs), max: Math.max(...xs) });
const correctAction = opponent => (opponent + 1) % 3;
const dot = (a, b) => a.reduce((s, x, j) => s + x * b[j], 0);
const score = (head, f) => head.W.map((w, a) => head.b[a] + dot(w, f));
const blankHead = () => ({ W: [0, 1, 2].map(() => [0, 0, 0]), b: [0, 0, 0] });
const fixedCue = rates => {
  const x = rates.map(Math.log1p);
  const distances = calibration.centroids.map(c =>
    c.reduce((sum, value, j) => sum + (x[j] - value) ** 2, 0));
  return distances.indexOf(Math.min(...distances));
};
const matrixByCue = (rows, select) => [0, 1, 2].map(cue => [0, 1, 2].map(action =>
  rows.filter(r => r.cue === cue && select(r) === action).length));
function advice(rows) {
  const x = rows.filter(r => r.cue !== null), delivered = x.filter(r => r.emittedBit !== null);
  const changed = delivered.filter(r => r.action !== r.candidate);
  const correctProposal = r => r.candidate === correctAction(r.opponent);
  const approvedCorrect = delivered.filter(r => r.emittedBit === 1 && correctProposal(r)).length;
  const approvedIncorrect = delivered.filter(r => r.emittedBit === 1 && !correctProposal(r)).length;
  const rejectedCorrect = delivered.filter(r => r.emittedBit === 0 && correctProposal(r)).length;
  const rejectedIncorrect = delivered.filter(r => r.emittedBit === 0 && !correctProposal(r)).length;
  return { observed: x.length, delivered: delivered.length,
    approved: count(delivered.map(r => r.emittedBit), 1),
    rejected: count(delivered.map(r => r.emittedBit), 0),
    approvedCorrect, approvedIncorrect, rejectedCorrect, rejectedIncorrect,
    correctProposals: x.filter(correctProposal).length,
    changed: changed.length,
    helpfulChanges: changed.filter(r => !correctProposal(r) && r.reward === 1).length,
    harmfulChanges: changed.filter(r => correctProposal(r) && r.reward !== 1).length,
    otherChanges: changed.filter(r => !correctProposal(r) && r.reward !== 1).length,
    changedToRewarded: changed.filter(r => r.reward === 1).length,
    bitDecoded: delivered.filter(r => r.decodedBit === r.emittedBit).length,
    minDnTargetLead: delivered.length ? Math.min(...delivered.map(r =>
      r.dnCounts[r.emittedBit] - r.dnCounts[1 - r.emittedBit])) : null };
}
function reconstruct(training) {
  const heads = [0, 1, 2, 3].map(blankHead), selectedHeadUpdates = [0, 0, 0, 0];
  for (const r of training.rows) {
    const h = r.update.selectedHead, a = r.update.selectedAction, step = r.update.step;
    if (h !== r.selectedHead || a !== r.action || !Number.isFinite(step))
      throw Error('Selected update identity mismatch');
    selectedHeadUpdates[h]++;
    heads[h].b[a] += step;
    for (let j = 0; j < 3; j++) heads[h].W[a][j] += step * r.features[j];
  }
  if (!heads.every((h, i) => h.b.every((v, a) => close(v, training.after.heads[i].b[a])) &&
      h.W.every((w, a) => w.every((v, j) => close(v, training.after.heads[i].W[a][j])))))
    throw Error('Saved selected-head updates do not reconstruct final parameters');
  return { selectedHeadUpdates,
    headParameterNorms: heads.map(h => Math.sqrt(h.W.flat().reduce((s, v) => s + v * v, 0) +
      h.b.reduce((s, v) => s + v * v, 0))),
    noCrossHeadUpdate: training.rows.every(r => r.update.otherHeadScoreMaxChange === 0) };
}
function frozenMeasure(training, frozen, archivedFrozen) {
  if (frozen.rows.length !== 72 || !frozen.weightsUnchanged ||
      !same(frozen.rows.map(r => [r.sequenceSeed, r.round, r.cue, r.opponent,
        r.rates, r.features]), archivedFrozen.rows.map(r => [r.sequenceSeed, r.round,
        r.cue, r.opponent, r.rates, r.features])))
    throw Error('Frozen neural inputs differ from the archived shared-head comparator');
  let reroutedWins = 0, reroutedActionChanges = 0, reroutedCueErrors = 0;
  const byCue = [0, 1, 2].map(cue => {
    const x = frozen.rows.filter(r => r.cue === cue);
    const margins = [], rerouted = { wins: 0, actionChanges: 0, cueCorrect: 0 };
    for (const r of x) {
      const scores = score(training.after.heads[r.selectedHead], r.features);
      if (!scores.every((v, a) => close(v, r.scores[a])) ||
          scores.indexOf(Math.max(...scores)) !== r.action || r.selectedHead !== cue)
        throw Error('Saved frozen score/action or VPN routing mismatch');
      const winning = correctAction(r.opponent);
      margins.push(scores[winning] - Math.max(...scores.filter((_, a) => a !== winning)));
      const estimate = fixedCue(r.rates), alternative = score(training.after.heads[estimate], r.features);
      const alternateAction = alternative.indexOf(Math.max(...alternative));
      rerouted.cueCorrect += Number(estimate === cue);
      rerouted.wins += Number(result(alternateAction, r.opponent) === 1);
      rerouted.actionChanges += Number(alternateAction !== r.action);
    }
    reroutedWins += rerouted.wins;
    reroutedActionChanges += rerouted.actionChanges;
    reroutedCueErrors += x.length - rerouted.cueCorrect;
    return { cue, n: x.length, wins: count(x.map(r => r.reward), 1),
      actions: actions(x.map(r => r.action)), margin: stats(margins),
      fixedMbonCue: rerouted };
  });
  for (const r of frozen.rows.filter(r => r.cue === null)) {
    if (r.selectedHead !== 3) throw Error('Cue-free head mismatch');
    reroutedWins += Number(r.reward === 1);
  }
  return { wins: frozen.summary.wins, byCue, actions: frozen.summary.actions,
    vpnRoutingCorrect: frozen.summary.routed,
    analyticalFixedMbonRoute: { wins: reroutedWins,
      observedCueErrors: reroutedCueErrors, observedActionChanges: reroutedActionChanges,
      note: 'evaluation-only counterfactual on saved frozen MBON vectors and the already trained heads; no MBON-routed training' } };
}
const students = [];
for (const student of data.students) {
  if (![831, 832].includes(student.seed) || !modes.every(mode =>
      student.training[mode] && student.final[mode])) throw Error('Student or arm missing');
  const archivedStudent = shared.students.find(s => s.seed === student.seed);
  const arms = {};
  for (const mode of modes) {
    const training = student.training[mode];
    if (training.rows.length !== 24 || !training.synapsesUnchanged ||
        !same(training.before, student.initialParams) ||
        !training.before.heads.every(h => h.b.every(v => v === 0) &&
          h.W.flat().every(v => v === 0))) throw Error('Initial or training record mismatch');
    const x = training.rows.filter(r => r.cue !== null);
    arms[mode] = { training: {
      wins: training.summary.wins,
      provisionalByCueAction: matrixByCue(x, r => r.candidate),
      finalByCueAction: matrixByCue(x, r => r.action),
      rewardedByCueAction: [0, 1, 2].map(cue => [0, 1, 2].map(action =>
        x.filter(r => r.cue === cue && r.action === action &&
          r.update.rewardedWeightChange).length)),
      advice: advice(training.rows),
      targetUpdates: training.targetUpdates,
      parameters: reconstruct(training)
    }, frozen: frozenMeasure(training, student.final[mode], archivedStudent.final.correct) };
  }
  students.push({ seed: student.seed, initialFrozenWins: student.initial.summary.wins,
    archivedSharedCorrect: { wins: archivedStudent.final.correct.summary.wins,
      cueWins: archivedStudent.final.correct.summary.byCue.map(v => v.wins) },
    arms, correctMinusShuffled: {
      trainingWins: arms.correct.training.wins - arms.shuffled.training.wins,
      targetUpdates: arms.correct.training.targetUpdates.total - arms.shuffled.training.targetUpdates.total,
      frozenWins: arms.correct.frozen.wins - arms.shuffled.frozen.wins,
      frozenMarginByCue: [0, 1, 2].map(cue =>
        arms.correct.frozen.byCue[cue].margin.mean - arms.shuffled.frozen.byCue[cue].margin.mean)
    } });
}
const derived = { source: 'data/rps_teacher_guided_cue_heads.json',
  command: 'node tools/analyze_rps_teacher_guided_cue_heads.mjs',
  checkpoints: 'Only initial and final frozen evaluations were archived; no intermediate frozen checkpoint or time-to-criterion measurement.',
  attributionLimit: 'Cue-head isolation and VPN-spike routing changed together in training. Fixed-MBON-reroute numbers change only evaluation-time routing of already trained heads and cannot isolate the training contributions.',
  originalGates: { cueHeadTeachingPassed: data.passed,
    cueHeadTeachingAdvantage: data.gates.frozen.teachingAdvantage,
    sharedHeadStatus: shared.status }, students };
writeFileSync(output, `${JSON.stringify(derived, null, 2)}\n`);
console.log(JSON.stringify(students.map(s => ({ seed: s.seed,
  arms: Object.fromEntries(modes.map(mode => [mode, {
    trainWins: s.arms[mode].training.wins,
    targetUpdates: s.arms[mode].training.targetUpdates.total,
    frozenWins: s.arms[mode].frozen.wins,
    frozenMargins: s.arms[mode].frozen.byCue.map(v => v.margin.mean),
    mbonRouteAnalyticalWins: s.arms[mode].frozen.analyticalFixedMbonRoute.wins }])),
  correctMinusShuffled: s.correctMinusShuffled })), null, 2));
