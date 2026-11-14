// Frozen, write-once routing prerequisite for the opt-in cue-specific value head.
// node tools/check_rps_teacher_guided_cue_head_route.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { Opponent, RpsContextualBandit } from '../js/rps.js';

// The first routing probe omitted the ordinary controller's MBON-noise setup and
// failed exact archive matching. Preserve it; this is the corrected replacement.
const output = new URL('../data/rps_teacher_guided_cue_head_route_corrected.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite cue-head routing preflight');
const read = path => JSON.parse(readFileSync(new URL(path, import.meta.url)));
const sub = read('../data/subgraph.json');
const prior = read('../data/rps_teacher_guided_reset_preflight.json');
const calibration = read('../data/rps_contextual_bandit.json').design;
const close = (a, b) => Math.abs(a - b) < 1e-10;
const students = [];
for (const seed of [831, 832]) {
  const net = new Network(sub, wiring(sub, 'real', seed), { seed });
  new RpsContextualBandit(net, { seed, featureMean: calibration.featureMean,
    featureStd: calibration.featureStd, readoutLearn: false });
  const game = new Opponent({ strategy: 'previous_opponent', offset: 2, noise: 0, seed: 41000 });
  const archived = prior.students.find(s => s.seed === seed).training.correct.rows;
  const rows = [];
  net.resetState();
  for (let round = 0; round < 24; round++) {
    if (round) net.resetState();
    const cue = game.observe().previousOpponent;
    const key = seed * 10000000 + round;
    net.rand = rng(key + 2);
    const vpn = cue === null ? [] : Array.from(net.laneVpn[cue].slice(2));
    if (vpn.length) net.setDrive(vpn, 150);
    let row;
    for (let t = 0; t < 680; t++) {
      if (t === 50) net.resetCounts();
      if (t === 240) {
        if (vpn.length) net.setDrive(vpn, 0);
        const laneCounts = net.laneVpn.map(ids => Array.from(ids.slice(2))
          .reduce((sum, i) => sum + net.count[i], 0));
        const maximum = Math.max(...laneCounts);
        const estimate = maximum === 0 ? null :
          laneCounts.filter(x => x === maximum).length === 1 ? laneCounts.indexOf(maximum) : 'tie';
        const rates = net.groupMbon.map(ids => Array.from(ids)
          .reduce((sum, i) => sum + net.count[i], 0) / ids.length);
        const features = rates.map((rate, j) =>
          (Math.log1p(rate) - calibration.featureMean[j]) / calibration.featureStd[j]);
        row = { round, cue, laneCounts, estimate, rates, features };
      }
      if (t === 280) {
        game.play(0);
        net.resetCounts();
      }
      net.step();
    }
    const old = archived[round];
    row.archivedMatched = old.cue === cue &&
      ratesEqual(row.rates, old.rates) && ratesEqual(row.features, old.features);
    rows.push(row);
  }
  const observed = rows.filter(r => r.cue !== null);
  students.push({ seed, rows, summary: {
    observedCorrect: observed.filter(r => r.estimate === r.cue).length,
    observedTotal: observed.length,
    cueFreeCorrect: rows[0].cue === null && rows[0].estimate === null,
    ties: rows.filter(r => r.estimate === 'tie').length,
    archivedMatched: rows.every(r => r.archivedMatched),
    byCue: [0, 1, 2].map(cue => {
      const x = observed.filter(r => r.cue === cue);
      return { cue, n: x.length, correct: x.filter(r => r.estimate === cue).length };
    }),
    synapsesUnchanged: net.m.every(v => v === 1)
  } });
}
function ratesEqual(a, b) {
  return a.length === b.length && a.every((v, i) => close(v, b[i]));
}
const gates = {
  provenance: prior.status === 'failed_retained_learning' && prior.stage === 4,
  route: students.every(s => s.summary.observedCorrect === 23 &&
    s.summary.observedTotal === 23 && s.summary.cueFreeCorrect && s.summary.ties === 0),
  archivedFeatures: students.every(s => s.summary.archivedMatched),
  synapses: students.every(s => s.summary.synapsesUnchanged)
};
const data = { protocol: {
  command: 'node tools/check_rps_teacher_guided_cue_head_route.mjs',
  students: [831, 832], seed: 41000, rounds: 24,
  rule: 'unique positive maximum of own VPN lane spike counts since 50ms reset at 240ms; all zero is cue-free',
  gates: '23/23 observed routes, cue-free first round, no ties, exact archived MBON features, frozen synapses'
}, gates, passed: Object.values(gates).every(Boolean), students };
writeFileSync(output, `${JSON.stringify(data, null, 2)}\n`);
console.log(JSON.stringify({ passed: data.passed, gates,
  students: students.map(s => ({ seed: s.seed, ...s.summary })) }, null, 2));
