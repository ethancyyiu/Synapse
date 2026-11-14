// One frozen, reward-free analysis of saved balanced-training weights.
// node tools/check_rps_paper_specificity.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { RpsBioRL } from '../js/rps.js';

const output = new URL('../data/rps_paper_specificity.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite saved paper-specificity result');
const saved = JSON.parse(readFileSync(new URL('../data/rps_balanced_training_corrected.json', import.meta.url)));
const earlier = JSON.parse(readFileSync(new URL('../data/rps_frozen_input.json', import.meta.url)));
const sub = JSON.parse(readFileSync(new URL('../data/subgraph.json', import.meta.url)));
if (!saved.allValid || saved.students.length !== 3 ||
  saved.students.some(s => ![651, 652, 653].includes(s.seed))) throw Error('Saved trained states failed provenance check');
const repeats = 12, arms = ['initial', 'autonomous', 'balanced'];
const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;
const near = (a, b) => Math.abs(a - b) < 1e-12;
const rows = [], summaries = [];

for (const student of saved.students) {
  const seed = student.seed;
  const flies = Object.fromEntries(arms.map(arm => {
    const net = new Network(sub, wiring(sub, 'real', seed), { seed });
    const agent = new RpsBioRL(net, { seed });
    if (arm !== 'initial') {
      const plastic = new Uint8Array(net.m.length);
      for (const group of net.plastic) for (const index of group) plastic[index] = 1;
      const changed = student.runs.find(r => r.arm === arm)?.changedWeights;
      if (!changed) throw Error(`Missing saved weights: ${seed}, ${arm}`);
      const seen = new Set();
      for (const [index, multiplier] of changed) {
        if (!Number.isInteger(index) || !plastic[index] || seen.has(index) ||
          !Number.isFinite(multiplier)) throw Error(`Invalid saved weight: ${seed}, ${arm}`);
        seen.add(index);
        net.m[index] = multiplier;
      }
    }
    return [arm, { net, agent, frozen: net.m.slice() }];
  }));
  const paperEdges = flies.initial.net.plastic[1];
  const paperPre = flies.initial.net.plasticPre[1];
  const paperMbonCount = flies.initial.net.groupMbon[1].length;
  const studentRows = [];
  for (let repeat = 0; repeat < repeats; repeat++) {
    for (let state = 0; state < 3; state++) {
      for (const arm of arms) {
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
        const rates = [...agent.rates];
        const activeKC = Array.from(net.idx.kc).filter(i => net.count[i] > 0);
        const paperInputProxy = paperEdges.reduce((sum, index, n) =>
          sum + net.w[index] * net.m[index] * net.trace[paperPre[n]], 0) / paperMbonCount;
        const row = { seed, repeat, state, arm, action, rates, activeKC,
          activeKCCounts: activeKC.map(i => net.count[i]),
          paperMinusScissors: rates[1] - rates[2],
          paperMinusBestOther: rates[1] - Math.max(rates[0], rates[2]),
          paperInputProxy, zeroRateTie: rates.every(x => x === 0) };
        studentRows.push(row);
        rows.push(row);
        if (arm === 'initial') {
          const old = earlier.trials.find(r => r.seed === seed && r.repeat === repeat && r.state === state);
          if (!old || action !== old.action || !old.rates.every((v, i) => near(v, rates[i])) ||
            JSON.stringify(activeKC) !== JSON.stringify(old.activeKC))
            throw Error(`Initial replay differs from archived KC probe: ${seed}/${repeat}/${state}`);
        }
      }
    }
  }
  for (const arm of arms) {
    const { net, frozen } = flies[arm];
    if (!net.m.every((m, i) => m === frozen[i])) throw Error(`Frozen weights changed: ${seed}/${arm}`);
  }
  const byArm = Object.fromEntries(arms.map(arm => [arm, [0, 1, 2].map(state => {
    const sample = studentRows.filter(r => r.arm === arm && r.state === state);
    return { state, meanActiveKC: mean(sample.map(r => r.activeKC.length)),
      meanRates: [0, 1, 2].map(a => mean(sample.map(r => r.rates[a]))),
      meanPaperMinusScissors: mean(sample.map(r => r.paperMinusScissors)),
      meanPaperMinusBestOther: mean(sample.map(r => r.paperMinusBestOther)),
      meanPaperInputProxy: mean(sample.map(r => r.paperInputProxy)),
      paperChosen: sample.filter(r => r.action === 1).length,
      zeroRateTies: sample.filter(r => r.zeroRateTie).length };
  })]));
  const balancedMinusInitial = [0, 1, 2].map(state => ({ state,
    paperRate: byArm.balanced[state].meanRates[1] - byArm.initial[state].meanRates[1],
    paperMinusBestOther: byArm.balanced[state].meanPaperMinusBestOther -
      byArm.initial[state].meanPaperMinusBestOther,
    paperInputProxy: byArm.balanced[state].meanPaperInputProxy -
      byArm.initial[state].meanPaperInputProxy }));
  const state2Specific = balancedMinusInitial[2].paperRate > 0 &&
    balancedMinusInitial[2].paperRate > Math.max(...balancedMinusInitial.slice(0, 2).map(x => x.paperRate)) &&
    balancedMinusInitial[2].paperMinusBestOther > 0 &&
    balancedMinusInitial[2].paperMinusBestOther >
      Math.max(...balancedMinusInitial.slice(0, 2).map(x => x.paperMinusBestOther));
  summaries.push({ seed, byArm, balancedMinusInitial, state2Specific });
}
const result = { command: 'node tools/check_rps_paper_specificity.mjs',
  source: 'corrected saved sparse trained weights; earlier initial KC probe',
  protocol: '12 matched stream keys per previous-move state; reset each trial; existing input at 150 Hz for 240 ms; count reset at 50 ms; original action readout; no reward or learning',
  operationalSpecificity: 'For every seed, state-2 balanced-minus-initial paper rate and paper-minus-best-other margin must both increase and exceed the corresponding changes in states 0 and 1',
  summaries, allState2Specific: summaries.every(s => s.state2Specific), rows };
writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({ allState2Specific: result.allState2Specific,
  summaries: summaries.map(s => ({ seed: s.seed, byArm: s.byArm,
    balancedMinusInitial: s.balancedMinusInitial, state2Specific: s.state2Specific })) }, null, 2));
