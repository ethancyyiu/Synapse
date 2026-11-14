// One locked frozen DNge175 receiving check using saved continuous-state teacher activity.
// node tools/check_rps_hidden_strategy_continuous_receiver.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { RpsContextualBandit } from '../js/rps.js';

const output = new URL('../data/rps_hidden_strategy_continuous_receiver.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite continuous receiving result');
const read = path => JSON.parse(readFileSync(new URL(path, import.meta.url)));
const sub = read('../data/subgraph.json');
const original = read('../data/rps_independent_teachers.json');
const source = read('../data/rps_hidden_strategy_continuous_source.json');
const priorSource = read('../data/rps_hidden_strategy_source.json');
const priorReceiver = read('../data/rps_hidden_strategy_receiver.json');
const receiverCalibration = read('../data/rps_dnge175_receiver.json');
const cueCalibration = read('../data/rps_readout_information.json');
const teacherCalibration = [781, 784, 785, 791], teacherHeldout = [782, 783, 786, 792, 793];
const receiverSeeds = [811, 812], targets = [3141, 3165], dnScale = 7.388888888888889;
const conditions = ['none', 'bit0', 'bit1'];
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const mean = xs => xs.reduce((s, x) => s + x, 0) / xs.length;
const { featureMean, featureStd } = original.design;
if (!source.passed || !priorSource.passed || priorReceiver.passed ||
    !receiverCalibration.passed || !cueCalibration.passed ||
    !same(source.design.calibrationIds, teacherCalibration) ||
    !same(source.design.heldoutIds, teacherHeldout) ||
    !same(source.design.evalSeeds, [52000, 52003, 52004]) ||
    !same(source.design.neuralSchedule,
      'reset once/sequence; first 680 ms round cue-free; cue 150 Hz 0-240 ms; reset counts 50 ms; Q/bit 240 ms; no drive/reward/dopamine 240-680 ms; preserve state through all 680 ms') ||
    !same(priorSource.lookup, [['A', 'B'], ['A', 'B'], ['A', 'B']]) ||
    !same(priorReceiver.design.receiverSeeds, receiverSeeds) ||
    !same(priorReceiver.design.targets, targets) ||
    (receiverCalibration.bitCentroids[0][0] +
      receiverCalibration.bitCentroids[1][1]) / 2 !== dnScale ||
    targets.some(i => sub.neurons.role[i] !== 'dn' || sub.neurons.type[i] !== 'DNge175'))
  throw Error('Locked continuous source or DN receiver protocol changed');
const sourceRows = source.teachers.flatMap(t => t.rows);
const teacherIds = [...teacherCalibration, ...teacherHeldout].sort((a, b) => a - b);
const teacherAudit = source.teachers.map(t => {
  const expected = teacherIds.includes(t.seed) && t.frozen &&
    same(t.initialController, t.finalController) && t.rows.length === 69 &&
    t.cueFree.length === 3 && [0, 1, 2].every(cue =>
      t.rows.filter(r => r.cue === cue).length === 23) &&
    t.rows.every(r => {
      const features = r.rates.map((rate, j) =>
        (Math.log1p(rate) - featureMean[j]) / featureStd[j]);
      const q = t.finalController.W.map((w, a) => w.reduce((v, x, j) =>
        v + x * features[j], t.finalController.b[a]));
      const margin = q[r.cue] - Math.max(...q.filter((_, a) => a !== r.cue));
      return r.decisionMs === 240 && r.neuralStateCarried &&
        r.elapsedMs === r.round * 680 + 240 &&
        r.features.every((x, j) => Math.abs(x - features[j]) < 1e-10) &&
        r.scores.every((x, j) => Math.abs(x - q[j]) < 1e-10) &&
        Math.abs(r.margin - margin) < 1e-10 && r.bit === Number(margin > 0);
    });
  if (!expected) throw Error(`Continuous teacher source invalid: ${t.seed}`);
  return { seed: t.seed, mapping: t.mapping, rows: t.rows.length, valid: expected };
});
if (teacherAudit.length !== 9 || sourceRows.length !== 621 ||
    !same(teacherAudit.map(x => x.seed).sort((a, b) => a - b), teacherIds) ||
    source.teachers.find(t => t.seed === 784)?.archivedReliabilityPassed !== false)
  throw Error('Missing teacher, weak B784, or recorded source rows');

const snapshot = agent => ({ W: agent.W.map(w => [...w]), b: [...agent.b],
  baseline: agent.baseline });
const receivers = receiverSeeds.map(seed => {
  const net = new Network(sub, wiring(sub, 'real', seed), { seed });
  const agent = new RpsContextualBandit(net, { seed, featureMean, featureStd,
    dnCountScale: dnScale, readoutLearn: false });
  agent.explore = false;
  return { seed, net, agent, initial: snapshot(agent) };
});
const cueGuess = rates => {
  const f = rates.map(Math.log1p);
  const d = cueCalibration.centroids.map(c => c.reduce((v, x, j) =>
    v + (f[j] - x) ** 2, 0));
  return d.indexOf(Math.min(...d));
};
const bitGuess = (counts, centroids) => {
  const d = centroids.map(c => c.reduce((v, x, j) =>
    v + (counts[j] - x) ** 2, 0));
  return d.indexOf(Math.min(...d));
};
const keyFor = (seed, cue, repeat) =>
  2026160400 + 100000 * (seed - 811) + 1000 * cue + 10 * repeat;

function receiverProbe(s, cue, repeat, condition) {
  const { seed, net, agent } = s;
  if (cue === null && condition !== 'none') throw Error('First round cannot receive a bit');
  const key = cue === null ? 2026160390 + 100000 * (seed - 811) :
    keyFor(seed, cue, repeat);
  net.resetState(); net.rand = rng(key + 2);
  const vpn = cue === null ? [] : Array.from(net.laneVpn[cue].slice(2));
  const target = condition === 'none' ? null : targets[condition === 'bit0' ? 0 : 1];
  if (vpn.some(i => targets.includes(i))) throw Error('Sensory and DN targets overlap');
  const dnCounts = [0, 0];
  let at280, row, updated = false;
  for (let ms = 0; ms < 680; ms++) {
    if (ms === 240 && target !== null) net.setDrive([target], 300);
    if (ms === 280) {
      at280 = [...dnCounts];
      if (target !== null) net.setDrive([target], 0);
      net.rand = rng(key + 3); // Matched cue-period stream, no neural reset.
      if (vpn.length) net.setDrive(vpn, 150);
    }
    if (ms === 330) net.resetCounts();
    if (ms === 520) {
      const cueActive = vpn.every(i => net.drive[i] === 150);
      if (vpn.length) net.setDrive(vpn, 0);
      agent.setDnWindowCounts([...dnCounts]);
      const action = agent.decide(net);
      row = { receiverSeed: seed, cue, repeat, condition, key, target,
        dnCounts: [...dnCounts], decisionDnCounts: targets.map(i => net.count[i]),
        rates: [...agent.rates], features: [...agent.f], scores: [...agent.scores],
        action, cueGuess: cue === null ? null : cueGuess(agent.rates),
        cueActive, decisionMs: ms };
      net.resetCounts();
    }
    if (ms === 620) { agent.afterDopamine(net, false); updated = true; }
    net.step();
    if (ms >= 240 && ms < 280) for (let k = 0; k < net.nSpikes; k++) {
      if (net.spikes[k] === targets[0]) dnCounts[0]++;
      else if (net.spikes[k] === targets[1]) dnCounts[1]++;
    }
  }
  if (!row || !updated || !row.cueActive || !same(at280, row.dnCounts) ||
      row.decisionMs !== 520 || row.features.length !== 5 ||
      row.features.some((v, j) => !Number.isFinite(v) ||
        Math.abs(v - (j < 3 ?
          (Math.log1p(row.rates[j]) - featureMean[j]) / featureStd[j] :
          row.dnCounts[j - 3] / dnScale)) > 1e-10) ||
      row.action !== row.scores.indexOf(Math.max(...row.scores)))
    throw Error('Receiver timing, frozen decision, or measured DN feature mismatch');
  return row;
}

const firstRounds = receivers.map(s => receiverProbe(s, null, -1, 'none'));
const rows = [];
for (const s of receivers) for (const cue of [0, 1, 2])
  for (let repeat = 0; repeat < 29; repeat++) for (const condition of conditions)
    rows.push(receiverProbe(s, cue, repeat, condition));
const priorRowMap = new Map(priorReceiver.rows.map(r =>
  [`${r.receiverSeed}/${r.cue}/${r.repeat}/${r.condition}`, r]));
const matchedReceiverArchive = rows.filter(r => r.repeat < 18).every(r => {
  const old = priorRowMap.get(`${r.receiverSeed}/${r.cue}/${r.repeat}/${r.condition}`);
  return old && same(r.dnCounts, old.dnCounts) && same(r.rates, old.rates) &&
    same(r.features, old.features) && same(r.scores, old.scores) &&
    r.cueGuess === old.cueGuess && r.action === old.action;
});
const rowMap = new Map(rows.map(r =>
  [`${r.receiverSeed}/${r.cue}/${r.repeat}/${r.condition}`, r]));
const byReceiver = receivers.map(s => {
  const own = rows.filter(r => r.receiverSeed === s.seed);
  const centroids = [0, 1].map(bit => [0, 1].map(j => mean(own.filter(r =>
    r.repeat < 6 && r.condition === `bit${bit}`).map(r => r.dnCounts[j]))));
  const transmissions = [];
  for (const t of source.teachers) for (const cue of [0, 1, 2]) {
    const sourceCue = t.rows.filter(r => r.cue === cue)
      .sort((a, b) => a.sequenceIndex - b.sequenceIndex || a.round - b.round);
    if (sourceCue.length !== 23) throw Error('Source cue coverage changed');
    sourceCue.forEach((p, ordinal) => {
      const receiver = rowMap.get(`${s.seed}/${cue}/${6 + ordinal}/bit${p.bit}`);
      if (!receiver) throw Error('Missing matched forced-bit receiver branch');
      const decodedBit = bitGuess(receiver.dnCounts, centroids);
      transmissions.push({ receiverSeed: s.seed, teacherSeed: t.seed,
        mapping: t.mapping, cue, sequenceIndex: p.sequenceIndex,
        round: p.round, ordinal, receiverRepeat: 6 + ordinal,
        sourceKey: p.key, receiverKey: receiver.key,
        sourceScores: p.scores, sourceMargin: p.margin, sourceBit: p.bit,
        dnCounts: receiver.dnCounts, decodedBit,
        decodedMapping: priorSource.lookup[cue][decodedBit],
        cueGuess: receiver.cueGuess, action: receiver.action });
    });
  }
  const byCue = [0, 1, 2].map(cue => {
    const ownCue = own.filter(r => r.cue === cue);
    const subset = condition => ownCue.filter(r => r.condition === condition &&
      r.repeat >= 6 && r.repeat < 18);
    const no = subset('none');
    const conditionsSummary = Object.fromEntries(conditions.map(condition => {
      const xs = subset(condition);
      return [condition, { cueCorrect: xs.filter(r => r.cueGuess === cue).length,
        meanRates: [0, 1, 2].map(j => mean(xs.map(r => r.rates[j]))),
        actionCounts: [0, 1, 2].map(a => xs.filter(r => r.action === a).length) }];
    }));
    const reach = [0, 1].map(bit => mean(subset(`bit${bit}`).map(r => r.dnCounts[bit])) -
      mean(no.map(r => r.dnCounts[bit])));
    const forced = ownCue.filter(r => r.repeat >= 6 && r.repeat < 12 &&
      r.condition !== 'none').map(r => ({ repeat: r.repeat,
        bit: r.condition === 'bit1' ? 1 : 0,
        decodedBit: bitGuess(r.dnCounts, centroids) }));
    const pairedCue = [0, 1].map(bit => {
      const xs = subset(`bit${bit}`);
      return { bit, newErrors: xs.filter(r => no.find(n => n.repeat === r.repeat)
        .cueGuess === cue && r.cueGuess !== cue).map(r => r.repeat),
      rescuedErrors: xs.filter(r => no.find(n => n.repeat === r.repeat)
        .cueGuess !== cue && r.cueGuess === cue).map(r => r.repeat),
      meanRateDelta: [0, 1, 2].map(j => mean(xs.map(r => r.rates[j] -
        no.find(n => n.repeat === r.repeat).rates[j]))),
      actionChanges: xs.filter(r => r.action !== no.find(n => n.repeat === r.repeat).action).length };
    });
    const byTeacher = source.teachers.map(t => {
      const xs = transmissions.filter(r => r.teacherSeed === t.seed && r.cue === cue);
      return { teacherSeed: t.seed, mapping: t.mapping,
        sourceBitOne: xs.filter(r => r.sourceBit === 1).length,
        bitMatched: xs.filter(r => r.sourceBit === r.decodedBit).length,
        sourceMappingCorrect: xs.filter(r =>
          priorSource.lookup[cue][r.sourceBit] === t.mapping).length,
        receivedMappingCorrect: xs.filter(r => r.decodedMapping === t.mapping).length };
    });
    const held = byTeacher.filter(t => teacherHeldout.includes(t.teacherSeed));
    const classAccuracy = measure => ['A', 'B'].map(mapping => {
      const xs = held.filter(t => t.mapping === mapping);
      return { mapping, n: xs.length * 23,
        correct: xs.reduce((total, t) => total + t[measure], 0) };
    });
    const sourceClasses = classAccuracy('sourceMappingCorrect');
    const receivedClasses = classAccuracy('receivedMappingCorrect');
    const balanced = classes => mean(classes.map(x => x.correct / x.n));
    return { cue, conditions: conditionsSummary, reach, forced,
      forcedCorrect: forced.filter(x => x.bit === x.decodedBit).length,
      pairedCue, byTeacher, sourceClasses, receivedClasses,
      balancedSource: balanced(sourceClasses),
      balancedReceived: balanced(receivedClasses), cueOnlyChance: 0.5 };
  });
  return { receiverSeed: s.seed, centroids, byCue, transmissions };
});
const frozen = receivers.every(s => same(snapshot(s.agent), s.initial) &&
  s.net.m.every(x => x === 1));
const validity = sourceRows.length === 621 && rows.length === 522 &&
  firstRounds.length === 2 && firstRounds.every(r => r.cue === null &&
    r.target === null) && matchedReceiverArchive &&
  receivers.every(s => [0, 1, 2].every(cue => conditions.every(condition =>
    rows.filter(r => r.receiverSeed === s.seed && r.cue === cue &&
      r.condition === condition).length === 29))) &&
  byReceiver.every(s => s.transmissions.length === 621 && s.byCue.every(c =>
    c.forced.length === 12 && c.byTeacher.every(t =>
      source.teachers.find(x => x.seed === t.teacherSeed)?.rows.filter(r =>
        r.cue === c.cue).length === 23) &&
    c.receivedClasses[0].n === 46 && c.receivedClasses[1].n === 69)) && frozen;
const gates = {
  validity,
  physicalReach: byReceiver.every(s => s.byCue.every(c => c.reach.every(x => x >= 1))),
  forcedBitDecoding: byReceiver.every(s => s.byCue.every(c => c.forcedCorrect === 12)),
  sourceBitRecovery: byReceiver.every(s => s.byCue.every(c =>
    c.byTeacher.every(t => t.bitMatched >= 21))),
  cuePreservation: byReceiver.every(s => s.byCue.every(c =>
    ['bit0', 'bit1'].every(k => c.conditions[k].cueCorrect >= 10 &&
      c.conditions[k].cueCorrect >= c.conditions.none.cueCorrect))),
  strategyTransmission: byReceiver.every(s => s.byCue.every(c =>
    c.balancedReceived >= 0.8 && c.balancedReceived - 0.5 >= 0.1)),
  noWeightChanges: frozen
};
const result = { command: 'node tools/check_rps_hidden_strategy_continuous_receiver.mjs',
  design: { sourceArtifact: 'data/rps_hidden_strategy_continuous_source.json',
    teacherCalibration, teacherHeldout, receiverSeeds, targets, dnScale,
    conditions, receiverRepeats: 29, sourceRowsPerTeacherCue: 23,
    receiverKey: '2026160400+100000*(receiverSeed-811)+1000*cue+10*repeat; rng(key+2), cue-period rng(key+3)',
    teacherState: 'saved continuous 680 ms rounds; reset once per 24-round sequence',
    receiverState: 'reset before each 680 ms frozen probe; continuous within probe',
    stimulusHz: 300, stimulusMs: [240, 280], cueHz: 150,
    cueMs: [280, 520], countResetMs: 330, decisionMs: 520,
    noUpdateMs: 620, endMs: 680,
    dnCalibrationRepeats: [0, 5], forcedTestRepeats: [6, 11],
    cueTestRepeats: [6, 17], sourcePairingRepeats: [6, 28],
    pairing: 'source rows sorted by sequenceIndex/round within teacher/cue; ordinal j selects receiver forced-bit probe repeat 6+j; strategy label evaluator-only',
    expected: { teacherSourceRows: 621, receiverCueProbes: 522,
      firstRoundProbes: 2, calibrationPerReceiverBit: 18,
      forcedTestPerReceiverCue: 12, cueTestPerReceiverCueArm: 12,
      sourcePairingPerReceiverTeacherCue: 23 },
    gates: 'reach>=1; forced bit 12/12; actual source bit>=21/23 per teacher/cue/receiver; cue>=10/12 under both bits and no loss vs no-message; balanced mapping>=80% per cue/receiver and >=10pp above chance; frozen integrity' },
  teacherAudit, sourceRows, firstRounds, receiverRows: rows, byReceiver,
  matchedReceiverArchive,
  gates, passed: Object.values(gates).every(Boolean),
  previousResetReceiverPassed: priorReceiver.passed,
  teacherTrainingStatus: 'not_run', studentTrainingStatus: 'not_run' };
writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({ byReceiver: byReceiver.map(s => ({ receiverSeed: s.receiverSeed,
  byCue: s.byCue.map(c => ({ cue: c.cue, reach: c.reach,
    forcedCorrect: c.forcedCorrect,
    cueCorrect: Object.fromEntries(conditions.map(k =>
      [k, c.conditions[k].cueCorrect])),
    balancedSource: c.balancedSource, balancedReceived: c.balancedReceived,
    weakestBitMatch: Math.min(...c.byTeacher.map(t => t.bitMatched)),
    B784: c.byTeacher.find(t => t.teacherSeed === 784) })) })),
  gates, passed: result.passed }, null, 2));
