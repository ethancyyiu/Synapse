// One frozen matched source-context check for trained B786, cue 1.
// node tools/analyze_rps_teacher786_cue1_context.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { RpsContextualBandit } from '../js/rps.js';

const output = new URL('../data/rps_teacher786_cue1_context.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite B786 diagnostic');
const read = path => JSON.parse(readFileSync(new URL(path, import.meta.url)));
const sub = read('../data/subgraph.json');
const original = read('../data/rps_independent_teachers.json');
const source = read('../data/rps_hidden_strategy_source.json');
const receiving = read('../data/rps_hidden_strategy_receiver.json');
const teacher = original.teachers.find(t => t.seed === 786);
if (!original.allValid || original.passed || !source.passed || receiving.passed ||
    !teacher?.passed || teacher.mapping !== 'B' ||
    !teacher.frozen.controllerUnchanged || !teacher.frozen.synapsesUnchanged)
  throw Error('Prior locked teacher or receiving result changed');
const { featureMean, featureStd, evalSeeds } = original.design;
const initial = teacher.final;
const net = new Network(sub, wiring(sub, 'real', 786), { seed: 786 });
const agent = new RpsContextualBandit(net, { seed: 786, featureMean, featureStd,
  readoutLearn: false });
agent.W.forEach((w, a) => w.set(initial.W[a]));
agent.b.set(initial.b); agent.baseline = initial.baseline; agent.explore = false;
const snapshot = () => ({ W: agent.W.map(w => [...w]), b: [...agent.b],
  baseline: agent.baseline });
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const close = (a, b) => Math.abs(a - b) < 1e-10;
if (!same(snapshot(), initial) || net.m.some(x => x !== 1))
  throw Error('Teacher reconstruction or synaptic state changed');
const features = rates => rates.map((rate, j) =>
  (Math.log1p(rate) - featureMean[j]) / featureStd[j]);
const scores = f => initial.W.map((w, a) =>
  w.reduce((q, value, j) => q + value * f[j], initial.b[a]));
const margin = q => q[1] - Math.max(q[0], q[2]);
const record = (label, key, rates, f, q, extra = {}) => ({ label, key,
  rates: [...rates], features: [...f], scores: [...q],
  margin: margin(q), bit: Number(margin(q) > 0), expectedBBit: 1,
  correct: margin(q) > 0, cueHz: 150, cueMs: [0, 240],
  countResetMs: 50, decisionMs: 240, ...extra });
const archive = teacher.frozen.rows.filter(r => r.cue === 1).map(r => {
  const i = evalSeeds.indexOf(r.sequenceSeed);
  if (i < 0 || r.round < 1 || r.round >= 24) throw Error('Invalid archive episode/round');
  const key = 786 * 10000000 + (100 + i) * 1000 + r.round;
  const f = features(r.rates), q = scores(f);
  if (f.some((v, j) => !close(v, r.features[j])) ||
      q.some((v, j) => !close(v, r.scores[j])) ||
      r.action !== q.indexOf(Math.max(...q)))
    throw Error('Archived feature, score, or action mismatch');
  return record('archived_continuous', key + 1, r.rates, f, q,
    { sequenceSeed: r.sequenceSeed, round: r.round,
      initialNeuralState: 'carried from earlier rounds within this evaluation episode' });
});
const archivedSource = source.sourceRows.find(t => t.seed === 786)?.rows.filter(r => r.cue === 1) ?? [];
if (archive.length !== 23 || archivedSource.length !== 23 ||
    archive.some(a => !archivedSource.some(s => s.sequenceSeed === a.sequenceSeed &&
      s.round === a.round && s.bit === a.bit)))
  throw Error('Original B786 source result does not match archived rows');

function resetReplay(a) {
  net.resetState(); net.rand = rng(a.key);
  const cue = Array.from(net.laneVpn[1].slice(2));
  net.setDrive(cue, 150);
  for (let ms = 0; ms < 240; ms++) {
    if (ms === 50) net.resetCounts();
    net.step();
  }
  net.setDrive(cue, 0);
  agent.decide(net);
  const f = features(agent.rates), q = scores(f);
  if (f.some((v, j) => !close(v, agent.f[j])) ||
      q.some((v, j) => !close(v, agent.scores[j])) ||
      agent.action !== q.indexOf(Math.max(...q)))
    throw Error('Replay feature, score, or action mismatch');
  return record('matched_reset_replay', a.key, agent.rates, f, q,
    { sequenceSeed: a.sequenceSeed, round: a.round,
      initialNeuralState: 'reset immediately before this cue' });
}
const replay = archive.map(resetReplay);
const teacherFirst = receiving.sourceRows.filter(r => r.teacherSeed === 786 &&
  r.cue === 1 && r.repeat >= 6).map(r => {
  const f = features(r.rates), q = scores(f);
  if (q.some((v, j) => !close(v, r.scores[j])) ||
      r.bit !== Number(margin(q) > 0)) throw Error('Saved teacher-first bit mismatch');
  return record('saved_teacher_first_unmatched',
    2026160400 + 1000 + 10 * r.repeat + 786 * 100000 + 1,
    r.rates, f, q, { repeat: r.repeat,
      initialNeuralState: 'reset immediately before this cue' });
});
const mean = xs => xs.reduce((sum, x) => sum + x, 0) / xs.length;
const stats = xs => {
  const margins = xs.map(x => x.margin).sort((a, b) => a - b);
  return { n: xs.length, bitOne: xs.filter(x => x.bit === 1).length,
    meanMargin: mean(margins), medianMargin: margins[Math.floor(margins.length / 2)],
    minMargin: margins[0], maxMargin: margins.at(-1),
    nearZeroAbsLt0_1: margins.filter(x => Math.abs(x) < 0.1).length,
    meanRates: [0, 1, 2].map(j => mean(xs.map(x => x.rates[j]))),
    meanFeatures: [0, 1, 2].map(j => mean(xs.map(x => x.features[j]))),
    meanScores: [0, 1, 2].map(j => mean(xs.map(x => x.scores[j]))) };
};
const pairs = archive.map((a, i) => ({ sequenceSeed: a.sequenceSeed, round: a.round,
  key: a.key, archivedMargin: a.margin, replayMargin: replay[i].margin,
  archivedBit: a.bit, replayBit: replay[i].bit,
  featureDelta: replay[i].features.map((v, j) => v - a.features[j]) }));
const positiveToZero = pairs.filter(p => p.archivedBit === 1 && p.replayBit === 0).length;
const zeroToPositive = pairs.filter(p => p.archivedBit === 0 && p.replayBit === 1).length;
const summary = { archived: stats(archive), matchedReset: stats(replay),
  teacherFirstUnmatched: stats(teacherFirst), positiveToZero, zeroToPositive,
  meanPairedMarginDrop: mean(pairs.map(p => p.archivedMargin - p.replayMargin)),
  meanAbsoluteFeatureDelta: [0, 1, 2].map(j =>
    mean(pairs.map(p => Math.abs(p.featureDelta[j])))),
  materialStateHistoryEffect: positiveToZero >= 6 &&
    mean(pairs.map(p => p.archivedMargin - p.replayMargin)) >= 0.20 };
if (teacherFirst.length !== 12 || summary.archived.bitOne !== 20 ||
    summary.teacherFirstUnmatched.bitOne !== 5 ||
    !same(snapshot(), initial) || net.m.some(x => x !== 1))
  throw Error('Saved counts or frozen weight integrity failed');
const result = { command: 'node tools/analyze_rps_teacher786_cue1_context.mjs',
  hypothesis: 'Preceding neural state/history changes cue-1 B786 MBON features and Q preference',
  design: { teacherSeed: 786, cue: 1, archivedEvalSeeds: evalSeeds,
    matchedStream: 'rng(786*10000000+(100+evalSeedIndex)*1000+round+1)',
    cueHz: 150, cueMs: [0, 240], countResetMs: 50, decisionMs: 240,
    singleChangedFactor: 'carry neural state across original episode versus reset before each matched cue',
    materialEffectCriterion: 'at least 6/23 archived-positive margins flip nonpositive and mean paired margin falls at least 0.20',
    teacherFirstComparison: '12 saved receiving-study held-out repeats, unmatched different streams' },
  archive, matchedResetReplay: replay, savedTeacherFirst: teacherFirst,
  pairs, summary, teacherWeightsUnchanged: true,
  priorReceivingGatePassed: false, studentTrainingStatus: 'not_started' };
writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify(summary, null, 2));
