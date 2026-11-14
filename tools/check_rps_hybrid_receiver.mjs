// Locked reward-free RPS KC receiving diagnostic. No teacher or student training.
// node tools/check_rps_hybrid_receiver.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { RpsContextualBandit } from '../js/rps.js';

const output = new URL('../data/rps_hybrid_receiver.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite hybrid receiver result');
const read = path => JSON.parse(readFileSync(new URL(path, import.meta.url)));
const sub = read('../data/subgraph.json');
const calibration = read('../data/rps_readout_information.json');
const prior = read('../data/rps_hybrid_message_source.json');
const context = read('../data/rps_contextual_bandit.json');
if (!calibration.passed || prior.passed || prior.threshold !== 1.098235868125541 ||
  !context.allValid || context.passed) throw Error('Archived source or calibration changed');
const targets = [[506, 545], [113, 1421]];
if (targets.flat().some(i => sub.neurons.role[i] !== 'kc')) throw Error('Receiving targets are not KCs');
const seed = 721, repeats = 12, conditions = ['none', 'bit0', 'bit1'];
const net = new Network(sub, wiring(sub, 'real', seed), { seed });
const agent = new RpsContextualBandit(net, { seed,
  featureMean: context.design.featureMean, featureStd: context.design.featureStd,
  readoutLearn: false });
agent.explore = false;
const controllerInitial = { W: agent.W.map(row => [...row]), b: [...agent.b], baseline: agent.baseline };
const feature = rates => rates.map((r, j) => (Math.log1p(r) - context.design.featureMean[j]) /
  context.design.featureStd[j]);
const statePrediction = rates => {
  const x = rates.map(Math.log1p);
  const d = calibration.centroids.map(c => c.reduce((sum, v, j) => sum + (x[j] - v) ** 2, 0));
  return d.indexOf(Math.min(...d));
};
const mean = values => values.reduce((sum, x) => sum + x, 0) / values.length;
const count = (ids, values) => ids.reduce((sum, i) => sum + values[i], 0);
const rows = [];
for (const state of [0, 1, 2]) for (let repeat = 0; repeat < repeats; repeat++)
  for (const condition of conditions) {
    net.resetState();
    net.rand = rng(seed * 100000 + repeat + 1);
    const cue = net.laneVpn[state].slice(2);
    if (!cue.length || cue.some(i => targets.flat().includes(i))) throw Error('Cue and KC targets overlap');
    net.setDrive(cue, 150);
    let windowTargetCounts;
    for (let t = 0; t < 240; t++) {
      if (t === 50) {
        net.resetCounts();
        if (condition !== 'none') net.setDrive(targets[condition === 'bit0' ? 0 : 1], 300);
      }
      if (t === 90 && condition !== 'none') net.setDrive(targets[condition === 'bit0' ? 0 : 1], 0);
      net.step();
      if (t === 89) windowTargetCounts = targets.map(ids => count(ids, net.count));
    }
    const cueActiveBeforeDecision = cue.every(i => net.drive[i] === 150);
    net.setDrive(cue, 0);
    const action = agent.decide(net);
    const rates = [...agent.rates], features = feature(rates);
    if (features.some((x, j) => Math.abs(x - agent.f[j]) > 1e-12) ||
      !cueActiveBeforeDecision || net.m.some(x => x !== 1)) throw Error('Probe validity failed');
    rows.push({ seed, state, repeat, condition, streamSeed: seed * 100000 + repeat + 1,
      windowTargetCounts, decisionTargetCounts: targets.map(ids => count(ids, net.count)),
      activeKcIds: Array.from(net.idx.kc).filter(i => net.count[i] > 0),
      activeKcCount: Array.from(net.idx.kc).filter(i => net.count[i] > 0).length,
      rates, features, statePrediction: statePrediction(rates), action,
      cueActiveBeforeDecision });
  }
if (JSON.stringify(controllerInitial) !== JSON.stringify({ W: agent.W.map(row => [...row]),
  b: [...agent.b], baseline: agent.baseline }) || net.m.some(x => x !== 1))
  throw Error('Frozen parameters changed');

const byState = [0, 1, 2].map(state => {
  const byCondition = Object.fromEntries(conditions.map(condition => {
    const xs = rows.filter(r => r.state === state && r.condition === condition);
    return [condition, { stateCorrect: xs.filter(r => r.statePrediction === state).length,
      meanTargetCountsAt90: [0, 1].map(k => mean(xs.map(r => r.windowTargetCounts[k]))),
      meanActiveKcAt240: mean(xs.map(r => r.activeKcCount)),
      meanMbonRates: [0, 1, 2].map(k => mean(xs.map(r => r.rates[k]))) }];
  }));
  const bitTraining = [0, 1].map(bit => {
    const train = rows.filter(r => r.state === state && r.condition === `bit${bit}` && r.repeat < 6);
    return [0, 1, 2].map(j => mean(train.map(r => r.features[j])));
  });
  const bitTest = rows.filter(r => r.state === state && r.condition !== 'none' && r.repeat >= 6)
    .map(r => {
      const distances = bitTraining.map(c => c.reduce((sum, x, j) => sum + (r.features[j] - x) ** 2, 0));
      return { repeat: r.repeat, bit: r.condition === 'bit0' ? 0 : 1,
        predicted: distances.indexOf(Math.min(...distances)), distances };
    });
  const decoderCorrect = bitTest.filter(r => r.bit === r.predicted).length;
  const perBitDecoderCorrect = [0, 1].map(bit => bitTest.filter(r => r.bit === bit && r.predicted === bit).length);
  const targetedExtraAt90 = [0, 1].map(bit =>
    byCondition[`bit${bit}`].meanTargetCountsAt90[bit] - byCondition.none.meanTargetCountsAt90[bit]);
  return { state, byCondition, targetedExtraAt90, bitTraining, bitTest,
    decoderCorrect, perBitDecoderCorrect };
});
const gates = {
  validity: rows.length === 108 && rows.every(r => r.cueActiveBeforeDecision &&
    r.features.length === 3 && r.rates.every(x => Number.isFinite(x))) &&
    net.m.every(x => x === 1),
  physicalRecruitment: byState.every(x => x.targetedExtraAt90.every(n => n >= 1)),
  bitDistinguishable: byState.every(x => x.decoderCorrect >= 10 &&
    x.perBitDecoderCorrect.every(n => n >= 5)),
  cuePreserved: byState.every(x => x.byCondition.none.stateCorrect >= 9 &&
    ['bit0', 'bit1'].every(name => x.byCondition[name].stateCorrect >= 8 &&
      x.byCondition.none.stateCorrect - x.byCondition[name].stateCorrect <= 2))
};
const result = { command: 'node tools/check_rps_hybrid_receiver.mjs',
  design: { seed, repeats, conditions, targets, messageHz: 300, messageStartMs: 50,
    messageEndMs: 90, cueHz: 150, cueStartMs: 0, cueEndMs: 240,
    stateDecoder: 'fixed reward-free log1p MBON centroids; no refit',
    bitDecoder: 'per-cue nearest Euclidean standardized-feature centroids; train repeats 0-5; test repeats 6-11',
    gates: 'target pair +1 mean spike/cue; bit decoder >=10/12 and >=5/6 per bit/cue; no-message state >=9/12/cue; messages >=8/12/cue and drop <=2/12' },
  byState, gates, passed: Object.values(gates).every(Boolean), rows };
writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({ byState: byState.map(x => ({ state: x.state,
  targetExtra: x.targetedExtraAt90, decoderCorrect: x.decoderCorrect,
  perBitDecoderCorrect: x.perBitDecoderCorrect,
  stateCorrect: Object.fromEntries(conditions.map(c => [c, x.byCondition[c].stateCorrect])),
  meanMbonRates: Object.fromEntries(conditions.map(c => [c, x.byCondition[c].meanMbonRates])) })),
  gates, passed: result.passed }, null, 2));
