// Read-only diagnostic of the saved RPS teaching pilot.
// node tools/analyze_rps_hidden_strategy_dn_use.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { result } from '../js/rps.js';

const input = new URL('../data/rps_hidden_strategy_teaching_pilot.json', import.meta.url);
const output = new URL('../data/rps_hidden_strategy_dn_use.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite DN-use diagnostic');
const pilot = JSON.parse(readFileSync(input));
if (pilot.status !== 'complete' || !pilot.allValid || pilot.passed ||
    pilot.studentResults.length !== 4) throw Error('Saved pilot status or sample changed');
const scale = pilot.design.dnScale;
const mean = xs => xs.reduce((s, x) => s + x, 0) / xs.length;
const approx = (a, b) => Math.abs(a - b) < 1e-10;
const equal = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const frozen = (controller, mbon, counts, opponent) => {
  const f = [...mbon, ...counts.map(x => x / scale)];
  const scores = controller.W.map((w, a) =>
    w.reduce((s, v, j) => s + v * f[j], controller.b[a]));
  const order = [0, 1, 2].sort((a, b) => scores[b] - scores[a] || a - b);
  const action = order[0];
  return { counts, scores, action, margin: scores[order[0]] - scores[order[1]],
    win: result(action, opponent) === 1 };
};
const summarize = rows => {
  const modes = ['actual', 'none', 'bit0', 'bit1'].filter(mode =>
    rows.every(row => row[mode]));
  const byMode = Object.fromEntries(modes.map(mode => [mode, {
    wins: rows.filter(row => row[mode].win).length,
    actions: [0, 1, 2].map(a => rows.filter(row => row[mode].action === a).length),
    meanMargin: mean(rows.map(row => row[mode].margin)) }]));
  const changes = {};
  for (const mode of modes.filter(x => x !== 'actual')) {
    const switched = rows.filter(row => row.actual.action !== row[mode].action);
    changes[`actualVs${mode[0].toUpperCase()}${mode.slice(1)}`] = {
      switched: switched.length, fraction: switched.length / rows.length,
      actualWinsOnly: switched.filter(row => row.actual.win && !row[mode].win).length,
      alternativeWinsOnly: switched.filter(row => !row.actual.win && row[mode].win).length,
      neitherWins: switched.filter(row => !row.actual.win && !row[mode].win).length };
  }
  if (modes.includes('bit0') && modes.includes('bit1')) {
    const switched = rows.filter(row => row.bit0.action !== row.bit1.action);
    changes.bit0VsBit1 = { switched: switched.length, fraction: switched.length / rows.length,
      bit0WinsOnly: switched.filter(row => row.bit0.win && !row.bit1.win).length,
      bit1WinsOnly: switched.filter(row => !row.bit0.win && row.bit1.win).length,
      neitherWins: switched.filter(row => !row.bit0.win && !row.bit1.win).length };
  }
  return { trials: rows.length, byMode, changes,
    byCue: [0, 1, 2].map(cue => {
      const xs = rows.filter(row => row.cue === cue);
      return { cue, trials: xs.length,
        actualWins: xs.filter(row => row.actual.win).length,
        noMessageWins: xs.filter(row => row.none.win).length,
        actualVsNoneSwitched: xs.filter(row => row.actual.action !== row.none.action).length };
    }) };
};
const evalAt = (run, checkpoint, mode) => run.evaluations.find(e =>
  e.afterEpisodes === checkpoint && e.mode === mode);
const coeffs = (run, rows) => {
  const W = run.final.W, mbonW = W.map(w => w.slice(0, 3)), dnW = W.map(w => w.slice(3));
  const mb = rows.map(row => row.features.slice(0, 3));
  const dn = rows.map(row => row.features.slice(3));
  const spread = xs => Math.max(...xs) - Math.min(...xs);
  const effects = rows.map(row => {
    const f = row.features;
    const mbScores = W.map((w, a) => w.slice(0, 3).reduce((s, v, j) =>
      s + v * f[j], run.final.b[a]));
    const dnScores = W.map(w => w.slice(3).reduce((s, v, j) => s + v * f[j + 3], 0));
    return { mbonSpread: spread(mbScores), dnSpread: spread(dnScores),
      dnAbsPerAction: mean(dnScores.map(Math.abs)) };
  });
  return { mbW: mbonW, dnW, mbWNorm: Math.hypot(...mbonW.flat()),
    dnWNorm: Math.hypot(...dnW.flat()),
    featureRms: [0, 1, 2].map(j => Math.sqrt(mean(mb.map(f => f[j] ** 2))))
      .concat([0, 1].map(j => Math.sqrt(mean(dn.map(f => f[j] ** 2))))),
    meanMbonScoreSpread: mean(effects.map(x => x.mbonSpread)),
    meanDnScoreSpread: mean(effects.map(x => x.dnSpread)),
    meanDnAbsPerAction: mean(effects.map(x => x.dnAbsPerAction)) };
};

const students = [];
for (const student of pilot.studentResults) {
  const runs = Object.fromEntries(student.runs.map(run => [run.arm, run]));
  if (!['correct', 'shuffled', 'untrained', 'disconnected'].every(x => runs[x]) ||
      !student.runs.every(run => run.initial.W.flat().every(v => v === 0) &&
        run.initial.b.every(v => v === 0) && run.synapsesUnchanged))
    throw Error(`Saved initial state or arm missing for ${student.seed}`);
  const correct = runs.correct;
  const connected = evalAt(correct, 12, 'correct').rows;
  const disconnected = evalAt(correct, 12, 'disconnected').rows;
  if (connected.length !== 72 || disconnected.length !== 72)
    throw Error('Final held-out records incomplete');
  const paired = connected.map((row, i) => {
    const no = disconnected[i];
    if (!equal([row.sequenceSeed, row.round, row.key, row.cue, row.opponent],
        [no.sequenceSeed, no.round, no.key, no.cue, no.opponent]) ||
        row.features.length !== 5 || no.features.length !== 5 ||
        !equal(no.features.slice(3), no.dnCounts.map(x => x / scale)))
      throw Error('Unmatched final held-out rows');
    return { row, no };
  }).filter(pair => pair.row.cue !== null);
  const heldoutRows = paired.map(({ row, no }) => {
    const mbon = row.features.slice(0, 3);
    const actual = frozen(correct.final, mbon, row.dnCounts, row.opponent);
    const none = frozen(correct.final, mbon, no.dnCounts, row.opponent);
    if (actual.action !== row.action || !actual.scores.every((q, a) => approx(q, row.scores[a])))
      throw Error(`Final held-out score/action reproduction failed for ${student.seed}`);
    return { sequenceSeed: row.sequenceSeed, round: row.round, cue: row.cue,
      opponent: row.opponent, actual, none };
  });
  const commonInputControlComparison = Object.fromEntries(student.runs.map(run => {
    const rows = paired.map(({ row, no }) => ({ cue: row.cue,
      actual: frozen(run.final, row.features.slice(0, 3), row.dnCounts, row.opponent),
      none: frozen(run.final, row.features.slice(0, 3), no.dnCounts, row.opponent) }));
    return [run.arm, { coefficients: coeffs(run, connected.filter(r => r.cue !== null)),
      heldoutSameInput: summarize(rows) }];
  }));

  // Archived training arms contain paired, actually simulated counts for both forced bits
  // on a subset of trial keys. These are in-sample input states scored by the FINAL frozen
  // controller; no missing intermediate controller state is reconstructed.
  const forcedRows = [];
  for (let i = 0; i < correct.trainingRows.length; i++) {
    const a = correct.trainingRows[i], sh = runs.shuffled.trainingRows[i],
      un = runs.untrained.trainingRows[i], no = runs.disconnected.trainingRows[i];
    if (![sh, un, no].every(r => equal([r.episode, r.round, r.key, r.cue, r.opponent],
        [a.episode, a.round, a.key, a.cue, a.opponent])) ||
        un.bit !== (a.cue === null ? null : 0) || no.bit !== null)
      throw Error(`Unmatched training count vectors for ${student.seed}`);
    if (a.cue === null) continue;
    const bit1 = a.bit === 1 ? a : sh.bit === 1 ? sh : null;
    if (!bit1) continue;
    const bit0 = a.bit === 0 ? a : un;
    const mbon = a.features.slice(0, 3);
    forcedRows.push({ episode: a.episode, round: a.round, cue: a.cue,
      opponent: a.opponent, sourceBit: a.bit,
      actual: frozen(correct.final, mbon, a.dnCounts, a.opponent),
      none: frozen(correct.final, mbon, no.dnCounts, a.opponent),
      bit0: frozen(correct.final, mbon, bit0.dnCounts, a.opponent),
      bit1: frozen(correct.final, mbon, bit1.dnCounts, a.opponent) });
  }

  const earlyObserved = pilot.design.checkpoints.map(checkpoint => {
    const withMessage = evalAt(correct, checkpoint, 'correct').rows;
    const withoutMessage = evalAt(correct, checkpoint, 'disconnected').rows;
    let changed = 0, messageWinsOnly = 0, disconnectedWinsOnly = 0, mbonChanged = 0;
    for (let i = 0; i < withMessage.length; i++) {
      const a = withMessage[i], b = withoutMessage[i];
      if (!equal([a.sequenceSeed, a.round, a.cue, a.opponent],
          [b.sequenceSeed, b.round, b.cue, b.opponent])) throw Error('Checkpoint pairing failed');
      if (a.cue === null) continue;
      if (a.action !== b.action) {
        changed++;
        if (a.reward === 1 && b.reward !== 1) messageWinsOnly++;
        if (b.reward === 1 && a.reward !== 1) disconnectedWinsOnly++;
      }
      if (!equal(a.features.slice(0, 3), b.features.slice(0, 3))) mbonChanged++;
    }
    return { checkpoint, connectedWins72: withMessage.filter(r => r.reward === 1).length,
      disconnectedWins72: withoutMessage.filter(r => r.reward === 1).length,
      observedActionChanges: changed, messageWinsOnly, disconnectedWinsOnly,
      mbonFeaturesChanged: mbonChanged, observedTrials: 69 };
  });
  students.push({ seed: student.seed, strategy: student.strategy,
    intermediateControllerParametersSaved: false,
    finalHeldout: { summary: summarize(heldoutRows), rows: heldoutRows },
    controlsOnIdenticalHeldoutInputs: commonInputControlComparison,
    matchedTrainingForcedBits: { summary: summarize(forcedRows), rows: forcedRows,
      limitation: 'Paired by student/episode/round/key/cue across archived training arms; prior neural histories and contemporaneous controllers can differ. Final controller applied analytically to in-sample MBON inputs.' },
    earlyObserved,
    limitation: 'Exact forced-bit-0 and forced-bit-1 counts for each held-out trial and intermediate checkpoint controller weights were not archived.' });
}
const derived = { command: 'node tools/analyze_rps_hidden_strategy_dn_use.mjs',
  source: 'data/rps_hidden_strategy_teaching_pilot.json',
  method: 'Read-only frozen controller calculations on saved actual neural features and DN spike counts; no simulation, rewards, or learning.',
  scale, students, originalPilotPassed: pilot.passed, originalPilotGates: pilot.gates };
writeFileSync(output, `${JSON.stringify(derived)}\n`);
console.log(JSON.stringify(students.map(s => ({ seed: s.seed, strategy: s.strategy,
  dnWeightNorm: s.controlsOnIdenticalHeldoutInputs.correct.coefficients.dnWNorm,
  dnScoreSpread: s.controlsOnIdenticalHeldoutInputs.correct.coefficients.meanDnScoreSpread,
  heldoutSwitches: s.finalHeldout.summary.changes.actualVsNone,
  matchedForcedTrials: s.matchedTrainingForcedBits.summary.trials,
  bit0VsBit1: s.matchedTrainingForcedBits.summary.changes.bit0VsBit1 })), null, 2));
