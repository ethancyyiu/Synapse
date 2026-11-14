// Frozen, write-once diagnostic of the stopped teacher-guided receiver preflight.
// node tools/check_rps_teacher_guided_receiver_history.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { RpsContextualBandit } from '../js/rps.js';

const output = new URL('../data/rps_teacher_guided_receiver_history.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite receiver-history result');
const read = path => JSON.parse(readFileSync(new URL(path, import.meta.url)));
const saved = read('../data/rps_teacher_guided_exploration_preflight.json');
const sub = read('../data/subgraph.json');
const context = read('../data/rps_contextual_bandit.json');
const calibration = read('../data/rps_readout_information.json');
if (saved.stage !== 2 || saved.status !== 'stopped_receiver' ||
    saved.receiver.rows.length !== 216 || !context.allValid || !calibration.passed)
  throw Error('Saved receiver or calibration state changed');
const { featureMean, featureStd } = context.design;
const seeds = [831, 832], firstCues = [0, 1, 2], repeats = 12;
const branches = ['none', 'bit0', 'bit1'];
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const guess = rates => {
  const x = rates.map(Math.log1p);
  const distance = calibration.centroids.map(c => c.reduce((sum, value, j) =>
    sum + (x[j] - value) ** 2, 0));
  return distance.indexOf(Math.min(...distance));
};
const snapshot = a => ({ W: a.W.map(w => [...w]), b: [...a.b], baseline: a.baseline });
function fly(seed) {
  const net = new Network(sub, wiring(sub, 'real', seed), { seed });
  const agent = new RpsContextualBandit(net,
    { seed, featureMean, featureStd, readoutLearn: true });
  net.resetState(); agent.explore = false;
  return { net, agent, before: snapshot(agent) };
}
function round(f, cue, key, schedule) {
  const { net, agent } = f, cueIds = Array.from(net.laneVpn[cue].slice(2));
  if (cueIds.length === 0 || cueIds.some(i => i === 3141 || i === 3165))
    throw Error('Missing or overlapping cue input');
  net.rand = rng(key + 2); agent.rand = rng(key + 501);
  if (schedule === 'revised') net.setDrive(cueIds, 150);
  let row = null, driveAtDecision = false;
  for (let t = 0; t < 680; t++) {
    if (schedule === 'revised') {
      if (t === 50) net.resetCounts();
      if (t === 240) {
        driveAtDecision = cueIds.every(i => net.drive[i] === 150);
        net.setDrive(cueIds, 0);
        const action = agent.decide(net);
        row = { cue, key, schedule, decisionMs: t, countResetMs: [50, 280],
          rates: [...agent.rates], features: [...agent.f], action,
          cueGuess: guess(agent.rates) };
      }
      if (t === 280) net.resetCounts();
    } else {
      if (t === 280) { net.rand = rng(key + 3); net.setDrive(cueIds, 150); }
      if (t === 330) net.resetCounts();
      if (t === 520) {
        driveAtDecision = cueIds.every(i => net.drive[i] === 150);
        net.setDrive(cueIds, 0);
        const action = agent.decide(net);
        row = { cue, key, schedule, decisionMs: t, countResetMs: [330, 520],
          rates: [...agent.rates], features: [...agent.f], action,
          cueGuess: guess(agent.rates) };
        net.resetCounts();
      }
    }
    if (t === 620 && schedule === 'teacher_first') agent.afterDopamine(net, false);
    net.step();
  }
  if (!row || !driveAtDecision || row.cueGuess !== guess(row.rates) ||
      row.features.some((v, j) => Math.abs(v -
        (Math.log1p(row.rates[j]) - featureMean[j]) / featureStd[j]) > 1e-10) ||
      !same(snapshot(agent), f.before) || net.m.some(v => v !== 1) ||
      cueIds.some(i => net.drive[i] !== 0)) throw Error('Frozen probe validity failed');
  row.cueDriveAtDecision = driveAtDecision;
  return row;
}
const archived = saved.receiver.rows;
const findSaved = (seed, cue, repeat, branch) => archived.find(r =>
  r.seed === seed && r.cue === cue && r.repeat === repeat && r.branch === branch);
const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;
const stats = xs => ({ mean: mean(xs), sd: Math.sqrt(mean(xs.map(x => (x - mean(xs)) ** 2))) });
const vectorStats = rows => [0, 1, 2].map(j => stats(rows.map(r => r.rates[j])));
const paired = (base, other, target) => ({
  gained: base.filter((r, i) => r.cueGuess !== target && other[i].cueGuess === target).length,
  lost: base.filter((r, i) => r.cueGuess === target && other[i].cueGuess !== target).length
});

// Recount existing matched probes before any new simulation.
const failed = archived.filter(r => r.seed === 832 && r.cue === 1);
const oldCounts = branches.map(branch => failed.filter(r => r.branch === branch &&
  r.second.cueGuess === 0).length);
if (!same(oldCounts, [6, 9, 9]) || archived.some(r => !r.weightsUnchanged))
  throw Error('Archived 6/12 vs 9/12 receiver result changed');
const isolated = [], teacherFirst = [];
for (const seed of seeds) for (const firstCue of firstCues)
  for (let repeat = 0; repeat < repeats; repeat++) {
    const episodeKey = 200 + firstCue * repeats + repeat;
    const key = seed * 10000000 + episodeKey * 1000;
    const nextCue = (firstCue + 2) % 3;
    const fresh = fly(seed);
    isolated.push({ seed, firstCue, repeat,
      second: round(fresh, nextCue, key + 1, 'revised') });
    const continuous = fly(seed);
    const first = round(continuous, firstCue, key, 'teacher_first');
    const second = round(continuous, nextCue, key + 1, 'teacher_first');
    teacherFirst.push({ seed, firstCue, repeat, first, second });
  }
const summary = [];
for (const seed of seeds) for (const firstCue of firstCues) {
  const target = (firstCue + 2) % 3;
  const base = Array.from({ length: repeats }, (_, i) => findSaved(seed, firstCue, i, 'none'));
  const bit0 = Array.from({ length: repeats }, (_, i) => findSaved(seed, firstCue, i, 'bit0'));
  const bit1 = Array.from({ length: repeats }, (_, i) => findSaved(seed, firstCue, i, 'bit1'));
  const iso = isolated.filter(r => r.seed === seed && r.firstCue === firstCue);
  const prior = teacherFirst.filter(r => r.seed === seed && r.firstCue === firstCue);
  if ([base, bit0, bit1, iso, prior].some(x => x.length !== 12 || x.some(v => !v)))
    throw Error('Missing matched probe');
  const baseFirst = base.map(r => r.first), baseSecond = base.map(r => r.second);
  const row = { seed, firstCue, nextCue: target,
    recognition: {
      revisedNone: [baseFirst.filter(r => r.cueGuess === firstCue).length,
        baseSecond.filter(r => r.cueGuess === target).length],
      revisedBit0: [bit0.filter(r => r.first.cueGuess === firstCue).length,
        bit0.filter(r => r.second.cueGuess === target).length],
      revisedBit1: [bit1.filter(r => r.first.cueGuess === firstCue).length,
        bit1.filter(r => r.second.cueGuess === target).length],
      isolatedSecond: iso.filter(r => r.second.cueGuess === target).length,
      teacherFirst: [prior.filter(r => r.first.cueGuess === firstCue).length,
        prior.filter(r => r.second.cueGuess === target).length] },
    pairedSecond: {
      bit0: paired(baseSecond, bit0.map(r => r.second), target),
      bit1: paired(baseSecond, bit1.map(r => r.second), target),
      isolated: paired(baseSecond, iso.map(r => r.second), target),
      teacherFirst: paired(baseSecond, prior.map(r => r.second), target) },
    rates: {
      revisedFirst: vectorStats(baseFirst), revisedSecond: vectorStats(baseSecond),
      revisedBit0Second: vectorStats(bit0.map(r => r.second)),
      revisedBit1Second: vectorStats(bit1.map(r => r.second)),
      isolatedSecond: vectorStats(iso.map(r => r.second)),
      teacherFirstSecond: vectorStats(prior.map(r => r.second)) },
    dn: [bit0, bit1].map((rows, bit) => ({ bit,
      decoded: rows.filter(r => r.first.decodedBit === bit).length,
      intended: stats(rows.map(r => r.first.dnCounts[bit])),
      offTarget: stats(rows.map(r => r.first.dnCounts[1 - bit])) })) };
  summary.push(row);
}
const target = summary.find(r => r.seed === 832 && r.firstCue === 1);
const interpretation = {
  baselineBelowFloor: target.recognition.revisedNone[1] < 10,
  historyNetCorrectionsAtLeast3: target.pairedSecond.isolated.gained -
    target.pairedSecond.isolated.lost >= 3,
  earlierScheduleNetCorrectionsAtLeast3: target.pairedSecond.teacherFirst.gained -
    target.pairedSecond.teacherFirst.lost >= 3,
  dnInterferenceOverTwoNetLosses: summary.some(r => ['bit0', 'bit1'].some(b =>
    r.pairedSecond[b].lost - r.pairedSecond[b].gained > 2)),
  frozenValidity: isolated.length === 72 && teacherFirst.length === 72 &&
    summary.every(r => r.dn.every(d => d.decoded === 12))
};
const result = { protocol: {
  source: 'data/rps_teacher_guided_exploration_preflight.json',
  seeds, firstCues, repeats, branches, secondCue: '(first+2)%3',
  revisedKeys: 'seed*10000000+(200+12*firstCue+repeat)*1000+roundIndex; rng(key+2)',
  teacherFirstAdditionalStream: 'rng(key+3) from 280ms',
  conditions: 'saved revised continuous none/bit0/bit1; new revised isolated second; new teacher-first continuous no message',
  decoder: 'archived fixed log1p three-MBON centroids, no fit',
  gates: 'history or schedule contribution >=3 paired net corrections/12; DN interference >2 paired net losses/12; original >=10/12 floor unchanged',
  command: 'node tools/check_rps_teacher_guided_receiver_history.mjs'
}, archivedFailureCounts: oldCounts, isolated, teacherFirst, summary, interpretation };
writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({ summary: summary.map(({ seed, firstCue, recognition, pairedSecond, dn }) =>
  ({ seed, firstCue, recognition, pairedSecond, dnDecoded: dn.map(d => d.decoded) })), interpretation }, null, 2));
