// New receiver criterion for the actual VPN-routed student inputs; protocol locked in lab notebook.
import { readFile, open, access } from 'node:fs/promises';
import { Network, rng, wiring } from '../js/sim.js';
import { ChallengeCueHeadBandit, studentWeights } from '../js/rps_challenge_students.js';

const read = async name => JSON.parse(await readFile(new URL(`../data/${name}`, import.meta.url)));
const output = new URL('../data/rps_action_preference_receiver_v2.json', import.meta.url);
try { await access(output); throw Error('Write-once receiver result already exists'); }
catch (error) { if (error.code !== 'ENOENT') throw error; }
const sub = await read('subgraph.json');
const { featureMean, featureStd } = (await read('rps_contextual_bandit.json')).design;
const centroids = (await read('rps_readout_information.json')).centroids;
const seeds = [11003, 11004, 11005], DN = [3141, 3165];
const codes = [[1, 0], [0, 1], [1, 1]];
const protocol = { development: seeds[0], untouchedValidation: seeds.slice(1), cues: [0, 1, 2],
  repeats: 12, branches: [null, 0, 1, 2], codes, targets: DN, hz: 300,
  dnWindow: [240, 280], vpnWindow: [280, 520], vpnHz: 150, resetCounts: 330,
  decision: 520, roundEnd: 680,
  key: 'seed*10000000+(500+12*cue+repeat)*1000; same key across all four branches',
  streams: ['rng(key+2)', 'rng(key+501)'],
  feature: '(log1p(group MBON mean spike count)-fixed mean)/fixed std',
  gate: '12/12 measured code, intended DN positive/off zero, VPN cue route 12/12, finite features, median paired standardized MBON L2 <= median distinct-repeat no-message L2 for every seed/cue/code' };
function decode(counts) {
  const bits = counts.map(x => Number(x > 0));
  return codes.findIndex(code => code.every((value, i) => value === bits[i]));
}
const distance = (a, b) => Math.hypot(...a.map((x, i) => x - b[i]));
const median = xs => {
  const sorted = [...xs].sort((a, b) => a - b);
  return (sorted[(sorted.length - 1) >> 1] + sorted[sorted.length >> 1]) / 2;
};
function probe(seed, cue, repeat, message) {
  const net = new Network(sub, wiring(sub, 'real', seed), { seed });
  const agent = new ChallengeCueHeadBandit(net, seed, featureMean, featureStd);
  agent.explore = false; agent.readoutLearn = false;
  const initial = studentWeights(agent), key = seed * 10000000 +
    (500 + 12 * cue + repeat) * 1000;
  net.resetState(); net.rand = rng(key + 2); agent.rand = rng(key + 501);
  const counts = [0, 0], vpn = Array.from(net.laneVpn[cue].slice(2));
  for (let t = 0; t < 680; t++) {
    if (t === 240 && message !== null) DN.forEach((id, i) => {
      if (codes[message][i]) net.setDrive([id], 300);
    });
    if (t === 280) { net.setDrive(DN, 0); net.setDrive(vpn, 150); }
    if (t === 330) net.resetCounts();
    if (t === 520) { net.setDrive(vpn, 0); agent.decide(net); }
    net.step();
    if (t >= 240 && t < 280) for (let i = 0; i < net.nSpikes; i++) {
      if (net.spikes[i] === DN[0]) counts[0]++;
      else if (net.spikes[i] === DN[1]) counts[1]++;
    }
  }
  const f = agent.rates.map((rate, j) => (Math.log1p(rate) - featureMean[j]) / featureStd[j]);
  const oldF = agent.rates.map(Math.log1p);
  const oldD = centroids.map(c => c.reduce((sum, value, j) => sum + (value - oldF[j]) ** 2, 0));
  return { seed, cue, repeat, message, key, counts, decoded: decode(counts),
    laneCounts: [...agent.laneCounts], routedCue: agent.activeHead,
    rates: [...agent.rates], f, oldPooledCue: oldD.indexOf(Math.min(...oldD)),
    weightsUnchanged: studentWeights(agent) === initial && net.m.every(x => x === 1) };
}
function batch(seed) {
  const rows = [];
  for (let cue = 0; cue < 3; cue++) for (let repeat = 0; repeat < 12; repeat++)
    for (const message of [null, 0, 1, 2]) rows.push(probe(seed, cue, repeat, message));
  const conditions = [0, 1, 2].flatMap(cue => {
    const none = rows.filter(r => r.cue === cue && r.message === null);
    const natural = [];
    for (let i = 0; i < none.length; i++) for (let j = i + 1; j < none.length; j++)
      natural.push(distance(none[i].f, none[j].f));
    const naturalMedian = median(natural);
    return [null, 0, 1, 2].map(message => {
      const x = rows.filter(r => r.cue === cue && r.message === message);
      const paired = message === null ? [] : x.map(r =>
        distance(r.f, none.find(n => n.repeat === r.repeat).f));
      const decoded = x.filter(r => r.decoded === (message ?? -1)).length;
      const routed = x.filter(r => r.routedCue === cue && r.laneCounts[cue] >
        Math.max(...r.laneCounts.filter((_, i) => i !== cue))).length;
      const intended = x.every(r => message === null ? r.counts.every(c => c === 0) :
        codes[message].every((bit, i) => bit ? r.counts[i] > 0 : r.counts[i] === 0));
      const passed = x.length === 12 && decoded === 12 && routed === 12 && intended &&
        x.every(r => r.weightsUnchanged && r.f.every(Number.isFinite)) &&
        (message === null || median(paired) <= naturalMedian);
      return { cue, message, n: x.length, decoded, routed, intended,
        oldPooledCorrect: x.filter(r => r.oldPooledCue === cue).length,
        naturalMedian, pairedMedian: paired.length ? median(paired) : null,
        pairedDistances: paired, passed };
    });
  });
  return { seed, rows, conditions, passed: conditions.every(c => c.passed) };
}
const artifact = { protocol, batches: [], status: 'started' };
for (const seed of seeds) {
  const result = batch(seed);
  artifact.batches.push(result);
  if (!result.passed) break;
}
artifact.status = artifact.batches.length === seeds.length &&
  artifact.batches.every(b => b.passed) ? 'passed' : 'failed';
const file = await open(output, 'wx');
try { await file.writeFile(`${JSON.stringify(artifact, null, 2)}\n`); }
finally { await file.close(); }
console.log(JSON.stringify({ status: artifact.status, batches: artifact.batches.map(b => ({
  seed: b.seed, passed: b.passed, conditions: b.conditions.map(c => ({
    cue: c.cue, message: c.message, decoded: c.decoded, routed: c.routed,
    oldPooledCorrect: c.oldPooledCorrect, naturalMedian: c.naturalMedian,
    pairedMedian: c.pairedMedian, passed: c.passed })) })) }));
