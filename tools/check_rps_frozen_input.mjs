// One preregistered, reward-free RPS input probe; see docs/lab-notebook.md.
// node tools/check_rps_frozen_input.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { RpsBioRL } from '../js/rps.js';

const output = new URL('../data/rps_frozen_input.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite frozen-input result');
const sub = JSON.parse(readFileSync(new URL('../data/subgraph.json', import.meta.url)));
const seeds = [651, 652, 653], repeats = 12;
const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;
const summary = [];
const trials = [];

for (const seed of seeds) {
  const net = new Network(sub, wiring(sub, 'real', seed), { seed });
  const agent = new RpsBioRL(net, { seed });
  const frozen = net.m.slice();
  const kc = Array.from(net.idx.kc);
  const rows = [];
  for (let repeat = 0; repeat < repeats; repeat++) {
    for (let state = 0; state < 3; state++) {
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
      const activeKC = kc.filter(i => net.count[i] > 0);
      const activeKCCounts = activeKC.map(i => net.count[i]);
      const best = Math.max(...rates);
      const row = { seed, repeat, state, activeKC, activeKCCounts, rates,
        paperMinusScissors: rates[1] - rates[2], action,
        paperStrictMax: Number(rates[1] > rates[0] && rates[1] > rates[2]),
        paperTieAdjustedMax: rates[1] === best ? 1 / rates.filter(x => x === best).length : 0,
        threeWayZeroTie: rates.every(x => x === 0) };
      rows.push(row);
      trials.push(row);
    }
  }
  if (net.m.some((weight, i) => weight !== frozen[i])) throw Error(`Weights changed: ${seed}`);

  const centroids = [0, 1, 2].map(state => {
    const calibration = rows.filter(r => r.state === state && r.repeat < 6);
    return kc.map(i => calibration.filter(r => r.activeKC.includes(i)).length / 6);
  });
  const classify = row => {
    const active = new Set(row.activeKC);
    const distances = centroids.map(c => c.reduce((d, p, j) => d + (Number(active.has(kc[j])) - p) ** 2, 0));
    return distances.indexOf(Math.min(...distances));
  };
  const heldout = rows.filter(r => r.repeat >= 6);
  const confusion = [0, 1, 2].map(state => [0, 1, 2].map(predicted =>
    heldout.filter(r => r.state === state && classify(r) === predicted).length));
  const byState = [0, 1, 2].map(state => {
    const subset = rows.filter(r => r.state === state);
    const frequencies = kc.map(i => [i, subset.filter(r => r.activeKC.includes(i)).length])
      .filter(([, n]) => n > 0).sort((a, b) => b[1] - a[1] || a[0] - b[0]);
    return { state, meanActiveKC: mean(subset.map(r => r.activeKC.length)),
      uniqueActiveKC: frequencies.length, mostFrequentKC: frequencies.slice(0, 12),
      meanRates: [0, 1, 2].map(a => mean(subset.map(r => r.rates[a]))),
      meanPaperMinusScissors: mean(subset.map(r => r.paperMinusScissors)),
      negativePaperMargin: subset.filter(r => r.paperMinusScissors < 0).length,
      paperChosen: subset.filter(r => r.action === 1).length,
      paperStrictMax: subset.reduce((n, r) => n + r.paperStrictMax, 0),
      paperTieAdjustedMax: subset.reduce((n, r) => n + r.paperTieAdjustedMax, 0),
      threeWayZeroTies: subset.filter(r => r.threeWayZeroTie).length,
      actionCounts: [0, 1, 2].map(a => subset.filter(r => r.action === a).length) };
  });
  const kcPass = confusion[2][2] >= 5 && confusion[0][2] + confusion[1][2] <= 2 &&
    byState[2].meanActiveKC >= 1;
  const competitionPass = byState[2].negativePaperMargin >= 10 && byState[2].paperChosen <= 2;
  summary.push({ seed, heldoutKCConfusion: confusion, byState, kcPass, competitionPass });
}

const result = { command: 'node tools/check_rps_frozen_input.mjs',
  design: { seeds, repeats, calibrationRepeats: 6, heldoutRepeats: 6,
    weights: 'reproducible initial, frozen; trained snapshots unavailable',
    neuralResetEachTrial: true, matchedNetworkAndTieStreamsBySeedRepeat: true,
    input: 'laneVpn[state].slice(2), 150 Hz, 0-240 ms', countResetMs: 50,
    decisionMs: 240, reward: false, teacher: false },
  gates: { eachSeedState2Recall: '>=5/6', eachSeedFalseState2: '<=2/12',
    eachSeedMeanActiveKC: '>=1', eachSeedNegativePaperMargin: '>=10/12',
    eachSeedPaperChosen: '<=2/12' },
  summary, allKcPass: summary.every(s => s.kcPass),
  allCompetitionPass: summary.every(s => s.competitionPass), trials };
writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({ summary, allKcPass: result.allKcPass,
  allCompetitionPass: result.allCompetitionPass }, null, 2));
