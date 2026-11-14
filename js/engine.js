// Spiking-network engine written for this project (). It is an alternative to js/sim.js, not a port of it:
// js/sim.js (from Fly Hero) stays in place and every earlier result was produced with it.
//
// Model: leaky integrate-and-fire neurons whose synaptic input decays exponentially, with the parameters published
// by Shiu et al. 2024 (https://doi.org/10.1038/s41586-024-07763-9). In deviation-from-rest form, between spikes
//     du/dt = (g - u) / TAU_MEMBRANE        dg/dt = -g / TAU_SYNAPSE
// What this engine decides for itself:
//   - Exact integration. The pair of linear equations above has a closed-form solution over one 1 ms step, so the
//     step applies that solution (Rotter & Diesmann 1999) and has no integration error. js/sim.js uses forward Euler.
//   - Input lists. Poisson drive and background noise are applied by walking the lists of driven and noisy neurons,
//     so random numbers are consumed in a different order from js/sim.js.
// Consequently the two engines do not produce the same spike trains; they should agree statistically.
//
// It reads the same subgraph file and offers the members the existing learners and round schedules use
// (idx, laneVpn, groupMbon, count, trace, lastSpike, spikes, nSpikes, m, plastic, plasticPre, t, rand, setDrive,
// setNoise, resetState, resetCounts, step), so it can stand in for the Network of js/sim.js.

export const MODEL = Object.freeze({
  REST_MV: -52, THRESHOLD_MV: -45, TAU_MEMBRANE_MS: 20, TAU_SYNAPSE_MS: 5,
  REFRACTORY_STEPS: 2,   // 2.2 ms in the paper, on a 1 ms grid
  DELAY_STEPS: 2,        // 1.8 ms in the paper, on a 1 ms grid
  SYNAPSE_MV: 0.275,     // potential added per unit of connectome weight
  DRIVE_KICK_MV: 0.275 * 250, // one Poisson drive event, far above threshold as in the paper
  TRACE_TAU_MS: 200,     // spike trace read by the learners
});

// Closed-form one-step propagator for (u, g).
const DECAY_U = Math.exp(-1 / MODEL.TAU_MEMBRANE_MS);
const DECAY_G = Math.exp(-1 / MODEL.TAU_SYNAPSE_MS);
const G_TO_U = MODEL.TAU_SYNAPSE_MS / (MODEL.TAU_SYNAPSE_MS - MODEL.TAU_MEMBRANE_MS) * (DECAY_G - DECAY_U);
const DECAY_TRACE = Math.exp(-1 / MODEL.TRACE_TAU_MS);
const FIRE_AT = MODEL.THRESHOLD_MV - MODEL.REST_MV;

// Small seedable generator (sfc32), returning floats in [0, 1).
export function rng(seed) {
  let a = 0x9e3779b9, b = 0x243f6a88, c = seed >>> 0, d = 1;
  const next = () => {
    const t = (a + b | 0) + d | 0;
    d = d + 1 | 0;
    a = b ^ b >>> 9;
    b = c + (c << 3) | 0;
    c = (c << 21 | c >>> 11) + t | 0;
    return (t >>> 0) / 4294967296;
  };
  for (let i = 0; i < 12; i++) next();
  return next;
}

// The measured wiring, as typed arrays. (The scrambled-wiring controls of the rhythm game remain in js/sim.js.)
export function wiring(sub, variant = 'real') {
  if (variant !== 'real') throw Error('js/engine.js provides the measured wiring only');
  return { pre: Int32Array.from(sub.edges.pre), post: Int32Array.from(sub.edges.post), w: Float64Array.from(sub.edges.w) };
}

const ROLES = ['vpn', 'kc', 'apl', 'mbon', 'pam', 'ppl1', 'mid', 'dn', 'vnc'];

export class Network {
  constructor(sub, edges = wiring(sub), { seed = 1, kcMbonGain = 1 } = {}) {
    const role = sub.neurons.role, tag = sub.neurons.tag, n = this.N = role.length;
    this.sub = sub;
    this.rand = rng(seed);

    // Neuron groups the learners address by name.
    const members = Object.fromEntries(ROLES.map(r => [r, []]));
    role.forEach((r, i) => { if (members[r]) members[r].push(i); });
    this.idx = Object.fromEntries(ROLES.map(r => [r, Int32Array.from(members[r])]));
    const tagged = (list, value) => list.filter(i => tag[i] === value);
    this.laneVpn = [0, 1, 2].map(lane => tagged(this.idx.vpn, lane));
    this.groupMbon = [0, 1, 2].map(group => tagged(this.idx.mbon, group));

    // Outgoing synapses of each neuron, stored contiguously (counting sort by presynaptic neuron).
    const count = edges.pre.length, first = this.first = new Int32Array(n + 1);
    for (let e = 0; e < count; e++) first[edges.pre[e] + 1]++;
    for (let i = 0; i < n; i++) first[i + 1] += first[i];
    const cursor = first.slice(0, n);
    this.target = new Int32Array(count);
    this.weight = new Float64Array(count);
    this.m = new Float64Array(count).fill(1); // plasticity multiplier per synapse
    const plastic = [[], [], []], plasticPre = [[], [], []];
    for (let e = 0; e < count; e++) {
      const from = edges.pre[e], to = edges.post[e], slot = cursor[from]++;
      const learnable = role[from] === 'kc' && role[to] === 'mbon';
      this.target[slot] = to;
      this.weight[slot] = edges.w[e] * MODEL.SYNAPSE_MV * (learnable ? kcMbonGain : 1);
      if (learnable && tag[to] >= 0 && tag[to] < 3) { plastic[tag[to]].push(slot); plasticPre[tag[to]].push(from); }
    }
    this.plastic = plastic.map(list => Int32Array.from(list));
    this.plasticPre = plasticPre.map(list => Int32Array.from(list));

    this.u = new Float64Array(n);             // membrane potential minus rest, mV
    this.g = new Float64Array(n);             // synaptic input, mV
    this.hold = new Uint8Array(n);            // refractory steps left
    this.trace = new Float64Array(n);
    this.count = new Uint16Array(n);
    this.lastSpike = new Float64Array(n);
    this.spikes = new Int32Array(n);
    this.nSpikes = 0;
    this.arriving = Array.from({ length: MODEL.DELAY_STEPS + 1 }, () => new Float64Array(n));
    this.driveHz = new Float64Array(n); this.driven = [];
    this.noiseHz = new Float64Array(n); this.noiseMv = new Float64Array(n); this.noisy = [];
    this.resetState();
  }

  // Clears activity and drive. Background noise and synapse multipliers are kept.
  resetState() {
    this.t = 0;
    this.u.fill(0); this.g.fill(0); this.hold.fill(0); this.trace.fill(0);
    this.count.fill(0); this.lastSpike.fill(-1e9); this.nSpikes = 0;
    this.arriving.forEach(buffer => buffer.fill(0));
    this.driveHz.fill(0); this.driven = [];
  }

  resetCounts() { this.count.fill(0); }

  // Poisson drive in Hz: each event lifts the potential by DRIVE_KICK_MV.
  setDrive(indices, hz) {
    for (const i of indices) this.driveHz[i] = hz;
    this.driven = [];
    for (let i = 0; i < this.N; i++) if (this.driveHz[i] > 0) this.driven.push(i);
  }

  // Background noise in Hz: each event adds kickMv to the synaptic input.
  setNoise(indices, hz, kickMv) {
    for (const i of indices) { this.noiseHz[i] = hz; this.noiseMv[i] = kickMv; }
    this.noisy = [];
    for (let i = 0; i < this.N; i++) if (this.noiseHz[i] > 0) this.noisy.push(i);
  }

  potential(i) { return MODEL.REST_MV + this.u[i]; }

  step() {
    const { N, u, g, hold, trace, rand } = this;
    const slots = this.arriving.length, inbox = this.arriving[this.t % slots];

    // 1. Deliver arriving input, then advance every neuron that is not refractory by exactly 1 ms.
    for (let i = 0; i < N; i++) {
      g[i] += inbox[i]; inbox[i] = 0;
      trace[i] *= DECAY_TRACE;
      if (hold[i]) continue;
      u[i] = u[i] * DECAY_U + g[i] * G_TO_U;
      g[i] *= DECAY_G;
    }
    // 2. External events for this step. Refractory neurons ignore them.
    for (const i of this.noisy) if (!hold[i] && rand() < this.noiseHz[i] * 1e-3) g[i] += this.noiseMv[i];
    for (const i of this.driven) if (!hold[i] && rand() < this.driveHz[i] * 1e-3) u[i] += MODEL.DRIVE_KICK_MV;
    // 3. Threshold, reset, and the refractory countdown.
    let fired = 0;
    for (let i = 0; i < N; i++) {
      if (hold[i]) { hold[i]--; continue; }
      if (u[i] > FIRE_AT) {
        u[i] = 0; g[i] = 0; hold[i] = MODEL.REFRACTORY_STEPS;
        trace[i] += 1; this.count[i]++; this.lastSpike[i] = this.t;
        this.spikes[fired++] = i;
      }
    }
    // 4. Send each spike down its synapses, to arrive DELAY_STEPS later.
    const outbox = this.arriving[(this.t + MODEL.DELAY_STEPS) % slots];
    const { first, target, weight, m, spikes } = this;
    for (let s = 0; s < fired; s++)
      for (let k = first[spikes[s]], end = first[spikes[s] + 1]; k < end; k++) outbox[target[k]] += weight[k] * m[k];
    this.nSpikes = fired;
    this.t++;
  }
}
