// One locked frozen single-teacher mixed-strategy source feasibility test.
// node tools/check_rps_single_teacher_mixed_source.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { Opponent, RpsContextualBandit, result as rpsResult } from '../js/rps.js';

const output = new URL('../data/rps_single_teacher_mixed_source.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite single-teacher mixed-source result');
const read = name => JSON.parse(readFileSync(new URL(`../data/${name}.json`, import.meta.url)));
const archive = read('rps_independent_teachers');
const reversal = read('rps_blocked_teacher_source');
const preflight = read('rps_hidden_strategy_interaction_reward_preflight');
const sub = read('subgraph');
const teacherSeed = 783;
const sequences = [
  { sequenceIndex: 0, opponentSeed: 53000, blocks: ['A', 'B', 'A', 'B'], role: 'calibration' },
  { sequenceIndex: 1, opponentSeed: 53003, blocks: ['B', 'A', 'B', 'A'], role: 'heldout' }
];
const teacher = archive.teachers.find(t => t.seed === teacherSeed);
const { featureMean, featureStd } = archive.design;
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const mean = xs => xs.reduce((sum, v) => sum + v, 0) / xs.length;
const snapshot = agent => ({ W: agent.W.map(w => [...w]), b: [...agent.b], baseline: agent.baseline });
const sourceBit = (scores, cue) => Number(scores[cue] >
  Math.max(...scores.filter((_, action) => action !== cue)));
const offset = mapping => mapping === 'A' ? 1 : mapping === 'B' ? 2 : NaN;
if (!archive.allValid || !teacher?.passed || teacher.mapping !== 'A' ||
    !teacher.frozen.controllerUnchanged || !teacher.frozen.synapsesUnchanged ||
    !Object.values(teacher.validity).every(Boolean) ||
    reversal.adaptationPassed || preflight.experimentRun ||
    !same(sequences.map(s => s.blocks), [['A', 'B', 'A', 'B'], ['B', 'A', 'B', 'A']]))
  throw Error('Archived teacher, reversal, or locked schedule changed');

// Pure opponent schedule check before any neural simulation.
const schedules = sequences.map(sequence => {
  const initialMove = new Opponent({ strategy: 'previous_opponent', offset: 1,
    noise: 0, seed: sequence.opponentSeed }).initial;
  let previous = null;
  const rounds = [];
  for (let round = 0; round < 96; round++) {
    const block = Math.floor(round / 24), mapping = sequence.blocks[block];
    const cue = previous;
    const opponent = cue === null ? initialMove : (cue + offset(mapping)) % 3;
    rounds.push({ sequenceIndex: sequence.sequenceIndex, opponentSeed: sequence.opponentSeed,
      role: sequence.role, block, blockRound: round % 24, round, mapping,
      cue, opponent });
    previous = opponent;
  }
  const coverage = sequence.blocks.map((mapping, block) => ({ block, mapping,
    cues: [0, 1, 2].map(cue => rounds.filter(r => r.block === block && r.cue === cue).length) }));
  if (!same(coverage.map(c => c.cues), [[8, 8, 7], [8, 8, 8], [8, 8, 8], [8, 8, 8]]) ||
      initialMove !== (sequence.sequenceIndex === 0 ? 0 : 1))
    throw Error('Opponent cue coverage or initial move changed');
  return { ...sequence, initialMove, rounds, coverage };
});

const net = new Network(sub, wiring(sub, 'real', teacherSeed), { seed: teacherSeed });
const agent = new RpsContextualBandit(net, { seed: teacherSeed, featureMean,
  featureStd, readoutLearn: false });
agent.W.forEach((w, action) => w.set(teacher.final.W[action]));
agent.b.set(teacher.final.b); agent.baseline = teacher.final.baseline;
agent.explore = false;
const originalController = snapshot(agent);
if (!same(originalController, teacher.final) || net.m.some(x => x !== 1))
  throw Error('Teacher final state cannot be reconstructed');

const rows = [], firstRounds = [];
for (const sequence of schedules) {
  net.resetState();
  let precedingCue = null;
  for (const task of sequence.rounds) {
    const { sequenceIndex, round, cue, opponent, mapping } = task;
    const key = teacherSeed * 10000000 + (200 + sequenceIndex) * 1000 + round + 1;
    net.rand = rng(key);
    const vpn = cue === null ? [] : Array.from(net.laneVpn[cue].slice(2));
    if (vpn.length) net.setDrive(vpn, 150);
    let decision = null;
    for (let ms = 0; ms < 680; ms++) {
      if (ms === 50) net.resetCounts();
      if (ms === 240) {
        if (vpn.length) net.setDrive(vpn, 0);
        if (cue !== null) {
          const action = agent.decide(net), scores = [...agent.scores];
          const rates = [...agent.rates], features = [...agent.f];
          const expectedFeatures = rates.map((rate, j) =>
            (Math.log1p(rate) - featureMean[j]) / featureStd[j]);
          const expectedScores = agent.W.map((w, a) =>
            w.reduce((sum, value, j) => sum + value * features[j], agent.b[a]));
          if (features.length !== 3 || features.some((value, j) =>
              !Number.isFinite(value) || Math.abs(value - expectedFeatures[j]) > 1e-10) ||
              scores.some((value, j) => !Number.isFinite(value) ||
                Math.abs(value - expectedScores[j]) > 1e-10) ||
              action !== scores.indexOf(Math.max(...scores)))
            throw Error(`Invalid teacher neural Q on sequence ${sequenceIndex}, round ${round}`);
          decision = { ...task, teacherSeed, key, precedingCue, decisionMs: 240,
            elapsedMs: round * 680 + 240, rates, features, scores, action,
            bit: sourceBit(scores, cue), winningAction: (opponent + 1) % 3,
            win: Number(rpsResult(action, opponent) === 1),
            untrainedBit: 0 };
        }
        net.resetCounts();
      }
      net.step();
    }
    if (cue === null) firstRounds.push({ ...task, teacherSeed, key });
    else {
      if (!decision) throw Error('Observed cue lacked a decision');
      rows.push(decision);
    }
    precedingCue = cue;
  }
}
if (!same(snapshot(agent), originalController) || net.m.some(x => x !== 1) ||
    firstRounds.length !== 2 || rows.length !== 190)
  throw Error('Frozen teacher or complete sequence check failed');

const calibration = rows.filter(r => r.sequenceIndex === 0);
const heldout = rows.filter(r => r.sequenceIndex === 1);
const bitCounts = xs => [0, 1].map(bit => xs.filter(r => r.bit === bit).length);
const cueMapping = (xs, cue, mapping) => xs.filter(r => r.cue === cue && r.mapping === mapping);
const lookup = [0, 1, 2].map(cue => [0, 1].map(bit => {
  const xs = calibration.filter(r => r.cue === cue && r.bit === bit);
  return xs.filter(r => r.mapping === 'B').length > xs.filter(r => r.mapping === 'A').length ? 'B' : 'A';
}));
const decoded = row => lookup[row.cue][row.bit];
const correct = xs => xs.filter(r => decoded(r) === r.mapping).length;
const balanced = xs => mean(['A', 'B'].map(mapping => {
  const subset = xs.filter(r => r.mapping === mapping);
  if (!subset.length) throw Error('Missing mapping in balanced score');
  return correct(subset) / subset.length;
}));
const shuffled = heldout.map(r => ({ ...r }));
const random = rng(2026110507);
for (const cue of [0, 1, 2]) {
  const indexes = shuffled.map((r, i) => r.cue === cue ? i : -1).filter(i => i >= 0);
  const bits = indexes.map(i => shuffled[i].bit);
  for (let i = bits.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [bits[i], bits[j]] = [bits[j], bits[i]];
  }
  indexes.forEach((index, j) => { shuffled[index].bit = bits[j]; });
}
const byCue = [0, 1, 2].map(cue => {
  const cal = calibration.filter(r => r.cue === cue);
  const test = heldout.filter(r => r.cue === cue);
  const testShuffled = shuffled.filter(r => r.cue === cue);
  const untrained = test.map(r => ({ ...r, bit: 0 }));
  return { cue, calibrationBits: bitCounts(cal), heldoutBits: bitCounts(test),
    calibration: ['A', 'B'].map(mapping => ({ mapping,
      n: cueMapping(calibration, cue, mapping).length,
      bits: bitCounts(cueMapping(calibration, cue, mapping)) })),
    heldout: ['A', 'B'].map(mapping => ({ mapping,
      n: cueMapping(heldout, cue, mapping).length,
      bits: bitCounts(cueMapping(heldout, cue, mapping)),
      decodedCorrect: correct(cueMapping(heldout, cue, mapping)) })),
    lookup: lookup[cue], balanced: balanced(test), shuffled: balanced(testShuffled),
    untrained: balanced(untrained), cueOnly: 0.5 };
});
const byBlock = [0, 1, 2, 3].map(block => {
  const xs = heldout.filter(r => r.block === block);
  const late = xs.filter(r => r.blockRound >= 12);
  return { block, mapping: sequences[1].blocks[block], n: xs.length,
    decodedCorrect: correct(xs), decodingAccuracy: correct(xs) / xs.length,
    lateN: late.length, lateDecodedCorrect: correct(late),
    lateAccuracy: correct(late) / late.length,
    teacherWins: xs.filter(r => r.win).length,
    teacherLateWins: late.filter(r => r.win).length,
    bits: bitCounts(xs),
    byCue: [0, 1, 2].map(cue => ({ cue,
      n: xs.filter(r => r.cue === cue).length,
      bits: bitCounts(xs.filter(r => r.cue === cue)) })) };
});
const calibrationBlocks = [0, 1, 2, 3].map(block => {
  const xs = calibration.filter(r => r.block === block);
  return { block, mapping: sequences[0].blocks[block], n: xs.length,
    teacherWins: xs.filter(r => r.win).length,
    bits: bitCounts(xs) };
});
const gates = {
  integrity: same(snapshot(agent), originalController) && net.m.every(x => x === 1) &&
    calibration.length === 95 && heldout.length === 95 &&
    schedules.every(s => s.coverage.every(c => c.cues.every(n => n >= 7))),
  calibrationDecodable: byCue.every(x => x.calibrationBits.every(n => n >= 4)),
  heldoutCoverage: byCue.every(x => x.heldout.every(y => y.n >= 15)),
  withinCueInformation: byCue.every(x => x.balanced >= 0.8 &&
    x.balanced - x.cueOnly >= 0.1 && x.balanced - x.shuffled >= 0.1 &&
    x.balanced - x.untrained >= 0.2),
  acrossBlocks: byBlock.every(x => x.decodingAccuracy >= 0.75) &&
    byBlock.slice(1).every(x => x.lateAccuracy >= 0.75)
};
const result = { command: 'node tools/check_rps_single_teacher_mixed_source.mjs',
  status: 'complete', teacherSeed, teacherStatus: 'frozen_A_trained',
  adaptationTested: false, teacherChanged: false, rewardDelivered: false,
  receiverStatus: 'not_run', studentTrainingStatus: 'not_run',
  design: { sequenceSeeds: sequences.map(x => x.opponentSeed),
    orders: sequences.map(x => x.blocks), blocksPerSequence: 4,
    roundsPerBlock: 24, neuralReset: 'once per 96-round sequence, not at strategy switches',
    teacherStream: 'rng(783*10000000+(200+sequenceIndex)*1000+round+1)',
    message: 'bit=1 iff Q[cue] > max(other Q), tie 0',
    neuralSchedule: '680ms; cue 150Hz 0-240ms, count reset 50ms, frozen Q and bit 240ms, idle 240-680ms; no reward or dopamine',
    decoder: 'per-cue bit-to-mapping majority on sequence 53000 only, ties A',
    shuffle: 'one within-cue heldout Fisher-Yates bit shuffle, rng(2026110507)',
    gates: 'integrity; calibration each bit>=4/cue; heldout each mapping>=15/cue; heldout balanced>=80%/cue and +10pp over cue-only/shuffle, +20pp over untrained; every block>=75%, post-switch late12>=75%' },
  initialController: originalController, finalController: snapshot(agent),
  scheduleCoverage: schedules.map(({ role, opponentSeed, initialMove, coverage }) =>
    ({ role, opponentSeed, initialMove, coverage })),
  firstRounds, calibrationBlocks, byBlock, byCue, lookup,
  calibrationRows: calibration, heldoutRows: heldout,
  shuffledHeldout: shuffled.map(r => ({ sequenceIndex: r.sequenceIndex, round: r.round,
    cue: r.cue, mapping: r.mapping, bit: r.bit })),
  gates, passed: Object.values(gates).every(Boolean),
  limitation: 'One A-trained frozen teacher, two dependent continuous 96-round sequences; no teacher reward adaptation or student receiving/learning.' };
writeFileSync(output, `${JSON.stringify(result)}\n`);
console.log(JSON.stringify({ teacherSeed, lookup, byCue, calibrationBlocks,
  heldoutBlocks: byBlock, gates, passed: result.passed }, null, 2));
