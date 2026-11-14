// One locked, reward-free hidden-strategy DNge175 receiving experiment.
// node tools/check_rps_hidden_strategy_receiver.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { RpsContextualBandit } from '../js/rps.js';

const output = new URL('../data/rps_hidden_strategy_receiver.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite receiving result');
const read = path => JSON.parse(readFileSync(new URL(path, import.meta.url)));
const sub = read('../data/subgraph.json');
const teachers = read('../data/rps_independent_teachers.json');
const source = read('../data/rps_hidden_strategy_source.json');
const oldReceiver = read('../data/rps_dnge175_receiver.json');
const cueCalibration = read('../data/rps_readout_information.json');
const targets = [3141, 3165], receiverSeeds = [811, 812], teacherSeeds = [782, 786];
const conditions = ['none', 'bit0', 'bit1'], dnScale = 7.388888888888889;
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const mean = xs => xs.reduce((s, x) => s + x, 0) / xs.length;
const { featureMean, featureStd } = teachers.design;
if (!teachers.allValid || teachers.passed || !source.passed || !oldReceiver.passed ||
    !cueCalibration.passed || !same(source.lookup, [['A','B'],['A','B'],['A','B']]) ||
    !same(oldReceiver.design.targets, targets) ||
    (oldReceiver.bitCentroids[0][0] + oldReceiver.bitCentroids[1][1]) / 2 !== dnScale ||
    targets.some(i => sub.neurons.role[i] !== 'dn' || sub.neurons.type[i] !== 'DNge175'))
  throw Error('Previously fixed teacher source or DN receiver changed');
const controller = agent => ({ W: agent.W.map(w => [...w]), b: [...agent.b], baseline: agent.baseline });
const create = (seed, withDn) => {
  const net = new Network(sub, wiring(sub, 'real', seed), { seed });
  const agent = new RpsContextualBandit(net, { seed, featureMean, featureStd,
    ...(withDn ? { dnCountScale: dnScale } : {}), readoutLearn: false });
  agent.explore = false;
  return { seed, net, agent };
};
const ts = teacherSeeds.map(seed => {
  const record = teachers.teachers.find(t => t.seed === seed);
  if (!record || !Object.values(record.validity).every(Boolean) ||
      !record.frozen.controllerUnchanged || !record.frozen.synapsesUnchanged ||
      !record.passed) throw Error(`Missing fixed teacher ${seed}`);
  const t = create(seed, false);
  t.agent.W.forEach((w, a) => w.set(record.final.W[a]));
  t.agent.b.set(record.final.b); t.agent.baseline = record.final.baseline;
  if (!same(controller(t.agent), record.final)) throw Error('Teacher reconstruction mismatch');
  return { ...t, mapping: record.mapping, saved: record.final };
});
const ss = receiverSeeds.map(seed => create(seed, true));
const initial = { teachers: ts.map(t => controller(t.agent)),
  receivers: ss.map(s => controller(s.agent)) };
const sourceKey = (cue, repeat) => 2026160400 + 1000 * cue + 10 * repeat;
const studentKey = (seed, cue, repeat) => sourceKey(cue, repeat) + 100000 * (seed - 811);
const cueGuess = rates => {
  const x = rates.map(Math.log1p);
  return cueCalibration.centroids.map(c => c.reduce((d, v, j) => d + (x[j] - v) ** 2, 0))
    .indexOf(Math.min(...cueCalibration.centroids.map(c =>
      c.reduce((d, v, j) => d + (x[j] - v) ** 2, 0))));
};
const bitGuess = (counts, centroids) => {
  const d = centroids.map(c => c.reduce((v, n, j) => v + (counts[j] - n) ** 2, 0));
  return d.indexOf(Math.min(...d));
};

function teacherProbe(t, cue, repeat) {
  const { net, agent } = t;
  net.resetState(); net.rand = rng(sourceKey(cue, repeat) + t.seed * 100000 + 1);
  const vpn = Array.from(net.laneVpn[cue].slice(2));
  net.setDrive(vpn, 150);
  for (let ms = 0; ms < 240; ms++) {
    if (ms === 50) net.resetCounts();
    net.step();
  }
  net.setDrive(vpn, 0);
  const action = agent.decide(net), scores = [...agent.scores];
  const bit = Number(scores[cue] > Math.max(...scores.filter((_, a) => a !== cue)));
  return { teacherSeed: t.seed, mapping: t.mapping, cue, repeat, action,
    rates: [...agent.rates], scores, bit };
}

function studentProbe(s, cue, repeat, condition) {
  const { net, agent, seed } = s, key = studentKey(seed, cue, repeat);
  net.resetState(); net.rand = rng(key + 2);
  const vpn = Array.from(net.laneVpn[cue].slice(2));
  const target = condition === 'none' ? null : targets[condition === 'bit0' ? 0 : 1];
  if (vpn.some(i => targets.includes(i))) throw Error('VPN and DN targets overlap');
  const dnCounts = [0, 0];
  let at280, at520, row, updated = false;
  for (let ms = 0; ms < 680; ms++) {
    if (ms === 240 && target !== null) net.setDrive([target], 300);
    if (ms === 280) {
      at280 = [...dnCounts];
      if (target !== null) net.setDrive([target], 0);
      net.rand = rng(key + 3);
      net.setDrive(vpn, 150);
    }
    if (ms === 330) net.resetCounts();
    if (ms === 520) {
      const cueActive = vpn.every(i => net.drive[i] === 150);
      net.setDrive(vpn, 0);
      at520 = [...dnCounts];
      agent.setDnWindowCounts(at520);
      const action = agent.decide(net);
      row = { receiverSeed: seed, cue, repeat, condition, key, target,
        dnCounts: at520, decisionDnCounts: targets.map(i => net.count[i]),
        rates: [...agent.rates], features: [...agent.f], scores: [...agent.scores],
        action, cueGuess: cueGuess(agent.rates), cueActive };
      net.resetCounts();
    }
    if (ms === 620) { agent.afterDopamine(net, false); updated = true; }
    net.step();
    if (ms >= 240 && ms < 280) for (let k = 0; k < net.nSpikes; k++) {
      if (net.spikes[k] === targets[0]) dnCounts[0]++;
      else if (net.spikes[k] === targets[1]) dnCounts[1]++;
    }
  }
  if (!row || !updated || !row.cueActive || !same(at280, at520) ||
      row.features.length !== 5 || row.features.some(x => !Number.isFinite(x)) ||
      row.features.some((x, j) => Math.abs(x - (j < 3 ?
        (Math.log1p(row.rates[j]) - featureMean[j]) / featureStd[j] :
        row.dnCounts[j - 3] / dnScale)) > 1e-10) ||
      row.action !== row.scores.indexOf(Math.max(...row.scores)))
    throw Error('Frozen student decision or measured-feature mismatch');
  return row;
}

const sourceRows = [];
for (const t of ts) for (const cue of [0, 1, 2]) for (let repeat = 0; repeat < 18; repeat++)
  sourceRows.push(teacherProbe(t, cue, repeat));
const rows = [];
for (const s of ss) for (const cue of [0, 1, 2]) for (let repeat = 0; repeat < 18; repeat++)
  for (const condition of conditions) rows.push(studentProbe(s, cue, repeat, condition));

const byReceiver = ss.map(s => {
  const own = rows.filter(r => r.receiverSeed === s.seed);
  const centroids = [0, 1].map(bit => [0, 1].map(j => mean(own.filter(r =>
    r.repeat < 6 && r.condition === `bit${bit}`).map(r => r.dnCounts[j]))));
  const byCue = [0, 1, 2].map(cue => {
    const held = own.filter(r => r.cue === cue && r.repeat >= 6);
    const selected = condition => held.filter(r => r.condition === condition);
    const no = selected('none');
    const conditionsSummary = Object.fromEntries(conditions.map(condition => {
      const xs = selected(condition);
      return [condition, { correct: xs.filter(r => r.cueGuess === cue).length,
        meanRates: [0, 1, 2].map(j => mean(xs.map(r => r.rates[j]))),
        actionCounts: [0, 1, 2].map(a => xs.filter(r => r.action === a).length) }];
    }));
    const reach = [0, 1].map(bit => mean(selected(`bit${bit}`).map(r => r.dnCounts[bit])) -
      mean(no.map(r => r.dnCounts[bit])));
    const bitTest = own.filter(r => r.cue === cue && r.repeat >= 6 && r.repeat < 12 &&
      r.condition !== 'none').map(r => ({ repeat: r.repeat, bit: r.condition === 'bit1' ? 1 : 0,
        decoded: bitGuess(r.dnCounts, centroids) }));
    const actionChanges = [0, 1].map(bit => selected(`bit${bit}`).filter(r =>
      r.action !== no.find(n => n.repeat === r.repeat).action).length);
    const pairedCue = [0, 1].map(bit => {
      const xs = selected(`bit${bit}`);
      return { bit, newErrors: xs.filter(r => no.find(n => n.repeat === r.repeat).cueGuess === cue &&
        r.cueGuess !== cue).map(r => r.repeat), rescuedErrors: xs.filter(r =>
        no.find(n => n.repeat === r.repeat).cueGuess !== cue && r.cueGuess === cue).map(r => r.repeat),
        meanRateDelta: [0, 1, 2].map(j => mean(xs.map(r => r.rates[j] -
          no.find(n => n.repeat === r.repeat).rates[j]))) };
    });
    const strategy = teacherSeeds.map(teacherSeed => {
      const teacher = ts.find(t => t.seed === teacherSeed);
      const probes = sourceRows.filter(r => r.teacherSeed === teacherSeed &&
        r.cue === cue && r.repeat >= 6);
      const values = probes.map(p => {
        const received = held.find(r => r.repeat === p.repeat && r.condition === `bit${p.bit}`);
        const decodedBit = bitGuess(received.dnCounts, centroids);
        return { repeat: p.repeat, sourceBit: p.bit, decodedBit,
          sourceMapping: source.lookup[cue][p.bit],
          receivedMapping: source.lookup[cue][decodedBit] };
      });
      return { teacherSeed, mapping: teacher.mapping, sourceCorrect: values.filter(v =>
        v.sourceMapping === teacher.mapping).length, receivedCorrect: values.filter(v =>
        v.receivedMapping === teacher.mapping).length,
        bitMatched: values.filter(v => v.sourceBit === v.decodedBit).length,
        sourceBitOne: values.filter(v => v.sourceBit === 1).length, values };
    });
    const balancedSource = mean(strategy.map(x => x.sourceCorrect / 12));
    const balancedReceived = mean(strategy.map(x => x.receivedCorrect / 12));
    return { cue, conditions: conditionsSummary, reach, bitTest,
      bitCorrect: bitTest.filter(x => x.bit === x.decoded).length,
      actionChanges, pairedCue, strategy, balancedSource, balancedReceived };
  });
  return { receiverSeed: s.seed, centroids, byCue };
});
const frozen = ts.every((t, i) => same(controller(t.agent), initial.teachers[i]) &&
  t.net.m.every(x => x === 1)) && ss.every((s, i) =>
  same(controller(s.agent), initial.receivers[i]) && s.net.m.every(x => x === 1));
const validity = sourceRows.length === 108 && rows.length === 324 && frozen &&
  sourceRows.every(p => p.bit === Number(p.scores[p.cue] >
    Math.max(...p.scores.filter((_, a) => a !== p.cue)))) &&
  ss.every(s => [0, 1, 2].every(cue => conditions.every(condition =>
    rows.filter(r => r.receiverSeed === s.seed && r.cue === cue &&
      r.condition === condition).length === 18))) &&
  rows.every(r => r.condition === 'none' ? r.target === null :
    r.target === targets[r.condition === 'bit0' ? 0 : 1]);
const gates = {
  validity,
  physicalReach: byReceiver.every(s => s.byCue.every(c => c.reach.every(x => x >= 1))),
  bitDecoding: byReceiver.every(s => s.byCue.every(c => c.bitCorrect >= 10)),
  cuePreservation: byReceiver.every(s => s.byCue.every(c => ['bit0', 'bit1'].every(k =>
    c.conditions[k].correct >= 10 && c.conditions.none.correct - c.conditions[k].correct <= 2))),
  strategyTransmission: byReceiver.every(s => s.byCue.every(c =>
    c.balancedReceived >= 0.8 && c.balancedReceived >= 0.6 &&
    c.strategy.every(x => x.bitMatched >= 10))),
  noWeightChanges: frozen
};
const result = { command: 'node tools/check_rps_hidden_strategy_receiver.mjs',
  design: { teacherSeeds, receiverSeeds, cues: [0, 1, 2], repeats: 18, conditions,
    sourceStream: '2026160400+1000*cue+10*repeat+teacherSeed*100000+1',
    receiverStream: '2026160400+100000*(receiverSeed-811)+1000*cue+10*repeat; rng(key+2), cue rng(key+3)',
    sourceRule: 'bit=1 iff Q[cue]>max(other Q); strict tie to zero',
    targets, dnScale, stimulusHz: 300, stimulusMs: [240, 280],
    teacherCueMs: [0, 240], receiverCueMs: [280, 520], cueHz: 150,
    teacherCountResetMs: 50, receiverCountResetMs: 330, decisionMs: 520,
    frozenUpdateMs: 620, roundEndMs: 680,
    calibrationRepeats: [0, 5], bitTestRepeats: [6, 11], cueTestRepeats: [6, 17],
    expected: { sourceProbes: 108, receiverProbes: 324,
      calibrationPerReceiverBit: 18, bitTestPerReceiverCue: 12,
      cueTestPerReceiverCueArm: 12, strategyPerReceiverCueMapping: 12 },
    gates: 'per receiver/cue reach>=1, bit>=10/12, cue>=10/12/bit and drop<=2/12, received balanced mapping>=80% and >=10pp above 50%, matched source bits>=10/12 per teacher, weights unchanged' },
  initial, sourceRows, rows, byReceiver, gates,
  passed: Object.values(gates).every(Boolean), previousSixTeacherGatePassed: false,
  previousTeachingPilotPassed: false, studentTrainingStatus: 'not_started' };
writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({ byReceiver: byReceiver.map(s => ({ receiverSeed: s.receiverSeed,
  byCue: s.byCue.map(c => ({ cue: c.cue, reach: c.reach, bitCorrect: c.bitCorrect,
    cueCorrect: Object.fromEntries(conditions.map(k => [k, c.conditions[k].correct])),
    balancedSource: c.balancedSource, balancedReceived: c.balancedReceived,
    sourceBitMatch: c.strategy.map(x => x.bitMatched),
    actionChanges: c.actionChanges })) })), gates, passed: result.passed }, null, 2));
