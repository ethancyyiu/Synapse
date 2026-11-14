// One locked, frozen, continuous-round teacher-first hidden-strategy source test.
// node tools/check_rps_hidden_strategy_continuous_source.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { Opponent, RpsContextualBandit } from '../js/rps.js';

const output = new URL('../data/rps_hidden_strategy_continuous_source.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite continuous source result');
const read = path => JSON.parse(readFileSync(new URL(path, import.meta.url)));
const sub = read('../data/subgraph.json');
const original = read('../data/rps_independent_teachers.json');
const replication = read('../data/rps_strategy_b_replication.json');
const priorSource = read('../data/rps_hidden_strategy_source.json');
const priorReceiver = read('../data/rps_hidden_strategy_receiver.json');
const calibrationIds = [781, 784, 785, 791], heldoutIds = [782, 783, 786, 792, 793];
const evalSeeds = [52000, 52003, 52004], all = [...original.teachers, ...replication.teachers];
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const mean = xs => xs.reduce((s, x) => s + x, 0) / xs.length;
const { featureMean, featureStd } = original.design;
const lookup = priorSource.lookup;
if (!original.allValid || original.passed || !replication.allValid || !replication.passed ||
    !priorSource.passed || priorReceiver.passed ||
    !same(original.design.featureMean, replication.design.featureMean) ||
    !same(original.design.featureStd, replication.design.featureStd) ||
    !same(lookup, [['A', 'B'], ['A', 'B'], ['A', 'B']]) ||
    !same(all.map(t => t.seed).sort((a, b) => a - b),
      [...calibrationIds, ...heldoutIds].sort((a, b) => a - b)) ||
    all.find(t => t.seed === 784)?.passed !== false)
  throw Error('Archived teacher split, prior failure, or decoder changed');
const snapshot = agent => ({ W: agent.W.map(w => [...w]), b: [...agent.b],
  baseline: agent.baseline });
const countBits = xs => [0, 1].map(bit => xs.filter(r => r.bit === bit).length);
const phaseOf = round => round <= 8 ? 'early' : round <= 16 ? 'middle' : 'late';
const sourceBit = (scores, cue) => Number(scores[cue] >
  Math.max(...scores.filter((_, action) => action !== cue)));

function runTeacher(saved) {
  if (!Object.values(saved.validity).every(Boolean) ||
      !saved.frozen.controllerUnchanged || !saved.frozen.synapsesUnchanged ||
      !saved.initial.W.flat().every(x => x === 0) ||
      !saved.initial.b.every(x => x === 0))
    throw Error(`Invalid archived teacher ${saved.seed}`);
  const { seed, mapping, offset } = saved;
  if ((mapping === 'A' ? 1 : 2) !== offset) throw Error('Strategy assignment changed');
  const net = new Network(sub, wiring(sub, 'real', seed), { seed });
  const agent = new RpsContextualBandit(net, { seed, featureMean, featureStd,
    readoutLearn: false });
  agent.W.forEach((w, a) => w.set(saved.final.W[a]));
  agent.b.set(saved.final.b); agent.baseline = saved.final.baseline;
  agent.explore = false;
  const initialController = snapshot(agent);
  if (!same(initialController, saved.final) || net.m.some(x => x !== 1))
    throw Error(`Cannot reconstruct frozen teacher ${seed}`);
  const rows = [], cueFree = [];
  for (const [sequenceIndex, sequenceSeed] of evalSeeds.entries()) {
    const initialOpponentMove = new Opponent({ strategy: 'previous_opponent',
      offset, noise: 0, seed: sequenceSeed }).initial;
    net.resetState();
    let previousOpponent = null, precedingCue = null;
    for (let round = 0; round < 24; round++) {
      const key = seed * 10000000 + (100 + sequenceIndex) * 1000 + round + 1;
      net.rand = rng(key); // Same current-round stream as the archived 400 ms source study.
      const cue = previousOpponent;
      const vpn = cue === null ? [] : Array.from(net.laneVpn[cue].slice(2));
      if (vpn.length) net.setDrive(vpn, 150);
      let row = null;
      for (let ms = 0; ms < 680; ms++) {
        if (ms === 50) net.resetCounts();
        if (ms === 240) {
          if (vpn.length) net.setDrive(vpn, 0);
          if (cue !== null) {
            const action = agent.decide(net), scores = [...agent.scores];
            const bit = sourceBit(scores, cue);
            const f = [...agent.f], rates = [...agent.rates];
            const expectedF = rates.map((rate, j) =>
              (Math.log1p(rate) - featureMean[j]) / featureStd[j]);
            const expectedQ = agent.W.map((w, a) => w.reduce((s, v, j) =>
              s + v * f[j], agent.b[a]));
            if (f.length !== 3 || f.some((x, j) => !Number.isFinite(x) ||
                Math.abs(x - expectedF[j]) > 1e-10) ||
                scores.some((x, j) => !Number.isFinite(x) ||
                  Math.abs(x - expectedQ[j]) > 1e-10) ||
                action !== scores.indexOf(Math.max(...scores)))
              throw Error(`Invalid frozen Q at teacher ${seed}, round ${round}`);
            row = { teacherSeed: seed, mapping, offset, sequenceIndex, sequenceSeed,
              round, phase: phaseOf(round), cue, precedingCue, key,
              decisionMs: 240, elapsedMs: round * 680 + 240,
              neuralStateCarried: true, rates, features: f, scores,
              margin: scores[cue] - Math.max(...scores.filter((_, a) => a !== cue)),
              bit, action, untrainedBit: 0 };
          }
          net.resetCounts();
        }
        net.step();
      }
      if (cue === null) cueFree.push({ teacherSeed: seed, sequenceIndex,
        sequenceSeed, round, key, initialOpponentMove, elapsedMs: 680 });
      else rows.push(row);
      previousOpponent = cue === null ? initialOpponentMove : (cue + offset) % 3;
      precedingCue = cue;
    }
  }
  const frozen = same(snapshot(agent), initialController) && net.m.every(x => x === 1);
  if (!frozen || rows.length !== 69 || cueFree.length !== 3 ||
      [0, 1, 2].some(cue => rows.filter(r => r.cue === cue).length !== 23))
    throw Error(`Frozen timing, coverage, or weight validity failed for ${seed}`);
  return { seed, mapping, offset, archivedReliabilityPassed: saved.passed,
    initialController, finalController: snapshot(agent), frozen, cueFree, rows };
}

const teachers = all.map(runTeacher);
const heldout = teachers.filter(t => heldoutIds.includes(t.seed));
const byTeacher = teachers.map(t => ({ seed: t.seed, mapping: t.mapping,
  archivedReliabilityPassed: t.archivedReliabilityPassed,
  byCue: [0, 1, 2].map(cue => {
    const xs = t.rows.filter(r => r.cue === cue);
    const byPhase = ['early', 'middle', 'late'].map(phase => {
      const part = xs.filter(r => r.phase === phase);
      return { phase, n: part.length, bitOne: part.filter(r => r.bit === 1).length,
        meanMargin: mean(part.map(r => r.margin)) };
    });
    return { cue, bits: countBits(xs), decodedCorrect: xs.filter(r =>
      lookup[cue][r.bit] === t.mapping).length,
      meanMargin: mean(xs.map(r => r.margin)), byPhase };
  }) }));
const heldoutRows = heldout.flatMap(t => t.rows);
const shuffledRows = heldoutRows.map(r => ({ ...r }));
const random = rng(2026110402);
for (const cue of [0, 1, 2]) {
  const indices = shuffledRows.map((r, i) => r.cue === cue ? i : -1).filter(i => i >= 0);
  const bits = indices.map(i => shuffledRows[i].bit);
  for (let i = bits.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [bits[i], bits[j]] = [bits[j], bits[i]];
  }
  indices.forEach((index, j) => { shuffledRows[index].bit = bits[j]; });
}
const balanced = xs => {
  const classes = ['A', 'B'].map(mapping => {
    const subset = xs.filter(r => r.mapping === mapping);
    return { mapping, n: subset.length,
      correct: subset.filter(r => lookup[r.cue][r.bit] === mapping).length };
  });
  return { classes, accuracy: mean(classes.map(x => x.correct / x.n)) };
};
const byCue = [0, 1, 2].map(cue => {
  const xs = heldoutRows.filter(r => r.cue === cue);
  const actual = balanced(xs), shuffled = balanced(shuffledRows.filter(r => r.cue === cue));
  const untrained = balanced(xs.map(r => ({ ...r, bit: 0 })));
  return { cue, trainedBits: countBits(xs), actual, shuffled, untrained,
    cueOnlyChance: 0.5, aboveCueOnly: actual.accuracy - 0.5,
    aboveShuffle: actual.accuracy - shuffled.accuracy };
});
const validity = teachers.length === 9 && teachers.every(t => t.frozen &&
  t.rows.length === 69 && t.cueFree.length === 3 &&
  [0, 1, 2].every(cue => t.rows.filter(r => r.cue === cue).length === 23)) &&
  heldoutRows.length === 345 && byCue.every(x =>
    x.actual.classes[0].n === 46 && x.actual.classes[1].n === 69) &&
  byTeacher.find(t => t.seed === 784)?.archivedReliabilityPassed === false;
const gates = {
  validity,
  trainedVsUntrained: byTeacher.filter(t => heldoutIds.includes(t.seed) &&
    t.mapping === 'B').every(t => t.byCue.every(x => x.bits[1] >= 18)),
  everyHeldoutTeacher: byTeacher.filter(t => heldoutIds.includes(t.seed)).every(t =>
    t.byCue.every(x => x.decodedCorrect >= 18)),
  withinCueMapping: byCue.every(x => x.actual.accuracy >= 0.8 &&
    x.aboveCueOnly >= 0.1 && x.aboveShuffle >= 0.1),
  nonsaturatedPooled: byCue.every(x => x.trainedBits.every(n => n / 115 >= 0.2)),
  bothStrategiesEveryCue: byCue.every(x =>
    x.actual.classes[0].n === 46 && x.actual.classes[1].n === 69)
};
const result = { command: 'node tools/check_rps_hidden_strategy_continuous_source.mjs',
  design: { calibrationIds, heldoutIds, evalSeeds, roundsPerSequence: 24,
    sequenceCount: 3, observedPerTeacher: 69, observedPerCueTeacher: 23,
    message: 'bit=1 iff Q[observed cue] > max(other Q); strict tie to zero',
    neuralSchedule: 'reset once/sequence; first 680 ms round cue-free; cue 150 Hz 0-240 ms; reset counts 50 ms; Q/bit 240 ms; no drive/reward/dopamine 240-680 ms; preserve state through all 680 ms',
    stream: 'rng(teacherSeed*10000000+(100+sequenceIndex)*1000+round+1) at each round start',
    strategy: 'seeded initial move from existing Opponent; thereafter (previous+offset)%3; offset A=1,B=2; hidden label used only by evaluator',
    decoder: 'unchanged prior calibration-only bit0->A, bit1->B lookup',
    shuffle: 'one heldout bit permutation pooled within cue, rng(2026110402)',
    temporalBins: 'rounds 1-8,9-16,17-23; descriptive only',
    gates: 'all teachers and cues complete; frozen weights; heldout B bit1>=18/23 each cue; each heldout teacher decoded>=18/23 each cue; balanced mapping>=80% and >=10pp over cue-only and shuffle each cue; both pooled bits>=20%; both strategies each cue' },
  priorSourcePassed: priorSource.passed, priorReceivingPassed: priorReceiver.passed,
  calibration: byTeacher.filter(t => calibrationIds.includes(t.seed)),
  heldout: byTeacher.filter(t => heldoutIds.includes(t.seed)),
  teachers, byCue, shuffledRows: shuffledRows.map(r => ({ teacherSeed: r.teacherSeed,
    mapping: r.mapping, cue: r.cue, sequenceSeed: r.sequenceSeed, round: r.round,
    bit: r.bit })), gates, passed: Object.values(gates).every(Boolean),
  receiverStatus: 'not_run', studentTrainingStatus: 'not_run' };
writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({ heldout: result.heldout.map(t => ({ seed: t.seed,
  mapping: t.mapping, bitOne: t.byCue.map(c => c.bits[1]),
  decodedCorrect: t.byCue.map(c => c.decodedCorrect) })),
  calibration: result.calibration.map(t => ({ seed: t.seed, mapping: t.mapping,
    bitOne: t.byCue.map(c => c.bits[1]) })),
  byCue: byCue.map(x => ({ cue: x.cue, trainedBits: x.trainedBits,
    balanced: x.actual.accuracy, shuffled: x.shuffled.accuracy,
    untrained: x.untrained.accuracy })), gates, passed: result.passed }, null, 2));
