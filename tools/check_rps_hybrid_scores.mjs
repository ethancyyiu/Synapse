// Single reward-free frozen diagnostic of the saved hybrid readout.
// node tools/check_rps_hybrid_scores.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { RpsRewardReadout } from '../js/rps.js';

const output = new URL('../data/rps_hybrid_scores.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite hybrid score diagnostic');
const saved = JSON.parse(readFileSync(new URL('../data/rps_reward_readout.json', import.meta.url)));
const initialProbe = JSON.parse(readFileSync(new URL('../data/rps_frozen_input.json', import.meta.url)));
const sub = JSON.parse(readFileSync(new URL('../data/subgraph.json', import.meta.url)));
const seeds = [651, 652, 653], repeats = 12;
if (!saved.allValid || saved.students.length !== 3 ||
  !seeds.every(seed => saved.students.some(s => s.seed === seed))) throw Error('Invalid saved hybrid source');
const { featureMean, featureStd } = saved.design;
const near = (a, b) => Math.abs(a - b) < 1e-12;
const mean = xs => xs.reduce((s, x) => s + x, 0) / xs.length;
const score = (readout, f) => readout.W.map((w, a) =>
  w.reduce((s, x, j) => s + x * f[j], readout.b[a]));
const counts = xs => [0, 1, 2].map(a => xs.filter(x => x === a).length);
const rows = [], summaries = [];

for (const seed of seeds) {
  const student = saved.students.find(s => s.seed === seed);
  const trained = student.runs.find(r => r.arm === 'hybrid');
  if (!trained || !trained.after.weightsUnchanged || !trained.after.readoutUnchanged)
    throw Error(`Missing valid trained state ${seed}`);
  const flies = Object.fromEntries(['initial', 'trained'].map(arm => {
    const net = new Network(sub, wiring(sub, 'real', seed), { seed });
    const agent = new RpsRewardReadout(net, { seed, featureMean, featureStd, readoutLearn: false });
    agent.explore = false;
    if (arm === 'trained') {
      const plastic = new Uint8Array(net.m.length);
      for (const group of net.plastic) for (const i of group) plastic[i] = 1;
      const seen = new Set();
      for (const [i, value] of trained.changedWeights) {
        if (!Number.isInteger(i) || !plastic[i] || seen.has(i) || !Number.isFinite(value))
          throw Error(`Invalid saved KC-MBON weight ${seed}/${i}`);
        seen.add(i); net.m[i] = value;
      }
      const savedReadout = trained.finalReadout;
      if (savedReadout.W.length !== 3 || savedReadout.W.some(w => w.length !== 3 ||
        w.some(x => !Number.isFinite(x))) || savedReadout.b.length !== 3 ||
        savedReadout.b.some(x => !Number.isFinite(x))) throw Error(`Invalid saved readout ${seed}`);
      agent.W.forEach((w, a) => w.set(savedReadout.W[a]));
      agent.b.set(savedReadout.b);
      agent.baseline = savedReadout.baseline;
    }
    return [arm, { net, agent, frozenWeights: net.m.slice(),
      frozenReadout: { W: agent.W.map(w => [...w]), b: [...agent.b] } }];
  }));
  const studentRows = [];
  for (let repeat = 0; repeat < repeats; repeat++) for (let state = 0; state < 3; state++) {
    for (const arm of ['initial', 'trained']) {
      const { net, agent } = flies[arm];
      net.resetState();
      net.rand = rng(seed * 100000 + repeat + 1);
      agent.rand = rng(seed * 100000 + repeat + 70001);
      const input = net.laneVpn[state].slice(2);
      net.setDrive(input, 150);
      for (let t = 0; t < 240; t++) {
        if (t === 50) net.resetCounts();
        net.step();
      }
      net.setDrive(input, 0);
      const action = agent.decide(net);
      const rates = [...agent.rates], features = [...agent.f];
      const scores = score(agent, features), correctAction = (state + 2) % 3;
      const correctMargin = scores[correctAction] - Math.max(...scores.filter((_, a) => a !== correctAction));
      if (action !== scores.indexOf(Math.max(...scores))) throw Error('Readout decision mismatch');
      const row = { seed, repeat, state, arm, rates, features, scores, action,
        correctAction, correctMargin, paperMinusScissors: scores[1] - scores[2] };
      if (arm === 'initial') {
        const prior = initialProbe.trials.find(r => r.seed === seed && r.repeat === repeat && r.state === state);
        if (!prior || !prior.rates.every((x, i) => near(x, rates[i])) ||
          scores.some(x => x !== 0)) throw Error(`Initial probe mismatch ${seed}/${repeat}/${state}`);
        const learned = trained.finalReadout;
        row.trainedReadoutOnInitialScores = score(learned, features);
        row.trainedReadoutOnInitialAction = row.trainedReadoutOnInitialScores.indexOf(
          Math.max(...row.trainedReadoutOnInitialScores));
      }
      rows.push(row); studentRows.push(row);
    }
  }
  for (const arm of ['initial', 'trained']) {
    const { net, agent, frozenWeights, frozenReadout } = flies[arm];
    if (!net.m.every((x, i) => x === frozenWeights[i]) ||
      JSON.stringify({ W: agent.W.map(w => [...w]), b: [...agent.b] }) !== JSON.stringify(frozenReadout))
      throw Error(`Frozen state changed ${seed}/${arm}`);
  }
  const byArm = Object.fromEntries(['initial', 'trained'].map(arm => [arm,
    [0, 1, 2].map(state => {
      const xs = studentRows.filter(r => r.arm === arm && r.state === state);
      return { state, meanRates: [0, 1, 2].map(j => mean(xs.map(r => r.rates[j]))),
        meanFeatures: [0, 1, 2].map(j => mean(xs.map(r => r.features[j]))),
        meanScores: [0, 1, 2].map(j => mean(xs.map(r => r.scores[j]))),
        actionCounts: counts(xs.map(r => r.action)),
        correctTop: xs.filter(r => r.action === r.correctAction).length,
        meanCorrectMargin: mean(xs.map(r => r.correctMargin)),
        meanPaperMinusScissors: mean(xs.map(r => r.paperMinusScissors)),
        paperOverScissors: xs.filter(r => r.paperMinusScissors > 0).length,
        crossActionCounts: arm === 'initial' ? counts(xs.map(r => r.trainedReadoutOnInitialAction)) : undefined,
        crossMeanPaperMinusScissors: arm === 'initial' ?
          mean(xs.map(r => r.trainedReadoutOnInitialScores[1] - r.trainedReadoutOnInitialScores[2])) : undefined };
    })]));
  const gap = byArm.trained[2].meanPaperMinusScissors - byArm.trained[0].meanPaperMinusScissors;
  const globalPaper = byArm.trained.every(s => s.actionCounts[1] >= Math.max(s.actionCounts[0], s.actionCounts[2])) &&
    Math.abs(gap) < 0.25;
  summaries.push({ seed, byArm, state2MinusState0PaperScissorsGap: gap, globalPaper });
}
const result = { command: 'node tools/check_rps_hybrid_scores.mjs',
  source: 'saved valid hybrid sparse KC-MBON multipliers, readout weights/biases, calibration normalization',
  protocol: '12 matched reset probes per state and student; original VPN 150 Hz; counts reset 50 ms; decision 240 ms; no reward or learning',
  analysisOnlyCorrectAction: '(previousMove + 2) % 3',
  descriptiveGlobalPaperRule: 'paper modal in every state and |state2-state0 mean paper-minus-scissors score| < 0.25 for every student',
  summaries, allGlobalPaper: summaries.every(s => s.globalPaper), rows };
writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({ allGlobalPaper: result.allGlobalPaper,
  summaries: summaries.map(s => ({ seed: s.seed, trained: s.byArm.trained,
    initial: s.byArm.initial.map(x => ({ state: x.state, crossActionCounts: x.crossActionCounts,
      crossMeanPaperMinusScissors: x.crossMeanPaperMinusScissors })),
    state2MinusState0PaperScissorsGap: s.state2MinusState0PaperScissorsGap,
    globalPaper: s.globalPaper })) }, null, 2));
