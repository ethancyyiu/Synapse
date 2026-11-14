// Tests for js/engine.js. Part 1 checks the engine against closed-form answers on tiny hand-built networks.
// Part 2 (pass --compare) runs the real connectome on both engines and prints how their activity compares.
// node tools/test_engine.mjs [--compare]
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { Network, MODEL, rng, wiring } from '../js/engine.js';

const tiny = (roles, edges) => ({ neurons: { role: roles, tag: roles.map(() => -1) },
  edges: { pre: edges.map(e => e[0]), post: edges.map(e => e[1]), w: edges.map(e => e[2]) } });
const fireOnce = (net, i) => { net.setDrive([i], 1000); net.step(); net.setDrive([i], 0); }; // 1000 Hz = certain event
const near = (a, b, tolerance, label) => assert.ok(Math.abs(a - b) <= tolerance, `${label}: ${a} vs ${b}`);
let passed = 0;
const test = (name, body) => { body(); passed++; console.log(`ok - ${name}`); };

test('a synaptic input follows the closed-form potential after the 2 ms delay', () => {
  const net = new Network(tiny(['mid', 'mid'], [[0, 1, 10]]));
  fireOnce(net, 0); // neuron 0 fires in step 0
  const g0 = 10 * MODEL.SYNAPSE_MV, { TAU_MEMBRANE_MS: tm, TAU_SYNAPSE_MS: ts } = MODEL;
  const exact = ms => MODEL.REST_MV + g0 * ts / (ts - tm) * (Math.exp(-ms / ts) - Math.exp(-ms / tm));
  net.step(); // step 1: nothing has arrived yet
  near(net.potential(1), MODEL.REST_MV, 0, 'before arrival');
  for (let ms = 1; ms <= 60; ms++) { net.step(); near(net.potential(1), exact(ms), 1e-12, `potential ${ms} ms after arrival`); }
  assert.equal(net.count[1], 0);
});

test('a neuron fires above threshold, resets, and stays silent for the refractory period', () => {
  const net = new Network(tiny(['mid', 'mid'], [[0, 1, 400]]));
  net.setDrive([0], 1000); // neuron 0 is pushed on every step it is able to fire
  const fired = [[], []];
  for (let t = 0; t < 30; t++) { net.step(); for (let k = 0; k < net.nSpikes; k++) fired[net.spikes[k]].push(t); }
  assert.deepEqual(fired[0].slice(0, 4), [0, 3, 6, 9], 'driven neuron fires every 1 + refractory steps');
  assert.ok(fired[1].length > 0 && fired[1][0] >= MODEL.DELAY_STEPS, 'target fires only after the delay');
  for (let k = 1; k < fired[1].length; k++) assert.ok(fired[1][k] - fired[1][k - 1] > MODEL.REFRACTORY_STEPS, 'refractory gap');
  assert.equal(net.potential(0) <= MODEL.THRESHOLD_MV, true);
});

test('weak input below threshold never fires; counts and reset behave', () => {
  const net = new Network(tiny(['mid', 'mid'], [[0, 1, 5]]));
  net.setDrive([0], 1000);
  for (let t = 0; t < 200; t++) net.step();
  assert.equal(net.count[1], 0); // steady input of 1.375 mV every 3 ms cannot reach 7 mV above rest
  assert.ok(net.count[0] > 60);
  net.resetCounts(); assert.equal(net.count[0], 0);
  net.resetState(); assert.equal(net.t, 0); near(net.potential(1), MODEL.REST_MV, 0, 'reset potential');
  for (let t = 0; t < 20; t++) net.step();
  assert.equal(net.count[0], 0, 'resetState clears drive');
});

test('Poisson drive gives the expected firing rate', () => {
  const net = new Network(tiny(['mid'], []), undefined, { seed: 11 });
  net.setDrive([0], 150);
  const steps = 200000; for (let t = 0; t < steps; t++) net.step();
  const expected = 1 / (1 / 0.15 + MODEL.REFRACTORY_STEPS); // one event per 1/p steps, then the refractory steps
  near(net.count[0] === 65535 ? NaN : 0, 0, 0, 'count did not overflow in this test window');
  let total = 0; net.resetState(); net.setDrive([0], 150);
  for (let block = 0; block < 20; block++) { net.resetCounts(); for (let t = 0; t < 10000; t++) net.step(); total += net.count[0]; }
  near(total / 200000, expected, expected * 0.02, 'spikes per ms');
});

test('the spike trace decays with a 200 ms time constant', () => {
  const net = new Network(tiny(['mid'], []));
  fireOnce(net, 0);
  near(net.trace[0], 1, 0, 'trace at the spike');
  for (let t = 0; t < 100; t++) net.step();
  near(net.trace[0], Math.exp(-100 / MODEL.TRACE_TAU_MS), 1e-12, 'trace after 100 ms');
});

test('plastic synapses are the KC to MBON ones, grouped by MBON tag, and scale transmission', () => {
  const sub = tiny(['kc', 'kc', 'mbon', 'mbon', 'mid'], [[0, 2, 1], [1, 3, 1], [0, 4, 1], [1, 2, 1]]);
  sub.neurons.tag = [-1, -1, 0, 2, -1];
  const net = new Network(sub);
  assert.deepEqual(net.plastic.map(list => list.length), [2, 0, 1]);
  assert.deepEqual([...net.plasticPre[0]].sort(), [0, 1]);
  assert.deepEqual([...net.plasticPre[2]], [1]);
  assert.deepEqual([...net.groupMbon[0]], [2]); assert.deepEqual([...net.idx.kc], [0, 1]);
  net.m[net.plastic[2][0]] = 3; // triple the KC1 -> MBON3 synapse
  fireOnce(net, 1); net.step(); net.step();
  near(net.g[3], 3 * MODEL.SYNAPSE_MV * Math.exp(-1 / MODEL.TAU_SYNAPSE_MS), 1e-12, 'scaled input');
});

test('same seed, same spikes; different seed, different spikes', () => {
  const run = seed => { const net = new Network(tiny(['mid', 'mid', 'mid'], [[0, 1, 30], [1, 2, 30]]), undefined, { seed });
    net.setDrive([0], 80); const out = []; for (let t = 0; t < 400; t++) { net.step(); out.push(net.nSpikes); } return out.join(''); };
  assert.equal(run(5), run(5)); assert.notEqual(run(5), run(6));
  const a = rng(1), values = Array.from({ length: 20000 }, a);
  near(values.reduce((s, x) => s + x, 0) / values.length, 0.5, 0.01, 'generator mean');
  assert.ok(values.every(x => x >= 0 && x < 1));
});

console.log(`engine: ${passed} tests passed`);

if (process.argv.includes('--compare')) {
  // The schedule of one teaching round, on the real connectome, for each engine: cue on for 240 ms, spikes counted
  // from 50 ms, then one advice code driven on the two message neurons for 40 ms.
  const old = await import('../js/sim.js');
  const sub = JSON.parse(await readFile(new URL('../data/subgraph.json', import.meta.url)));
  const DN = [3141, 3165], CODES = [[1, 0], [0, 1], [1, 1]], TRIALS = 12;
  const measure = (make, name) => {
    const net = make();
    net.setNoise(net.idx.mbon, 400, 1.2); // the background noise the learners set
    const rows = [];
    let routed = 0, decoded = 0, total = 0;
    for (const cue of [0, 1, 2]) {
      const mbon = [0, 0, 0]; let kc = 0;
      for (let trial = 0; trial < TRIALS; trial++) {
        net.resetState(); net.rand = old.rng(880000 + cue * 1000 + trial);
        const vpn = Array.from(net.laneVpn[cue].slice(2));
        net.setDrive(vpn, 150);
        const dn = [0, 0];
        for (let t = 0; t < 280; t++) {
          if (t === 50) net.resetCounts();
          if (t === 240) {
            const lanes = net.laneVpn.map(ids => Array.from(ids.slice(2)).reduce((s, i) => s + net.count[i], 0));
            routed += Number(lanes.indexOf(Math.max(...lanes)) === cue && lanes.filter(v => v === Math.max(...lanes)).length === 1);
            net.groupMbon.forEach((ids, k) => { mbon[k] += ids.reduce((s, i) => s + net.count[i], 0) / ids.length / TRIALS; });
            kc += net.idx.kc.reduce((s, i) => s + Number(net.count[i] > 0), 0) / TRIALS;
            net.setDrive(vpn, 0);
            DN.forEach((id, j) => { if (CODES[trial % 3][j]) net.setDrive([id], 300); });
          }
          net.step();
          if (t >= 240) for (let k = 0; k < net.nSpikes; k++) { const j = DN.indexOf(net.spikes[k]); if (j >= 0) dn[j]++; }
        }
        decoded += Number(dn.every((c, j) => Number(c > 0) === CODES[trial % 3][j])); total++;
      }
      rows.push({ engine: name, cue, mbon0: mbon[0].toFixed(2), mbon1: mbon[1].toFixed(2), mbon2: mbon[2].toFixed(2), activeKcs: kc.toFixed(1) });
    }
    return { rows, routed: `${routed}/${total}`, decoded: `${decoded}/${total}` };
  };
  const a = measure(() => new old.Network(sub, old.wiring(sub, 'real', 1), { seed: 1 }), 'js/sim.js');
  const b = measure(() => new Network(sub, wiring(sub), { seed: 1 }), 'js/engine.js');
  console.table([...a.rows, ...b.rows]);
  console.log(`cue read correctly from the sensory lanes: sim.js ${a.routed}, engine.js ${b.routed}`);
  console.log(`advice code read correctly from the two message neurons: sim.js ${a.decoded}, engine.js ${b.decoded}`);
}
