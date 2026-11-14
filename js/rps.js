// Rock-Paper-Scissors on the existing connectome. Mirrors flyhero/rps*.py.
import { rng } from './sim.js';
import { BioRL, DEEP_DEFAULTS } from './agents.js';

export const MOVES = ['Rock', 'Paper', 'Scissors'];
export const ROUND_MS = 400;
export const T_COUNT = 50, T_SIGNAL = 180, T_DECIDE = 240, T_LEARN = 340;
export const MID_SOURCES = [2709, 2360, 2325];
const INPUT_HZ = 150;
const SIGNAL_HZ = 75;

export function result(player, opponent) {
  if (![player, opponent].every(x => Number.isInteger(x) && x >= 0 && x < 3)) throw Error('invalid move');
  return [0, 1, -1][(player - opponent + 3) % 3];
}

export class Opponent {
  constructor({ strategy = 'previous_opponent', seed = 1, sequence = [0, 1, 2], offset = 1,
    switchAfter = 12, secondOffset = 2, noise = 0 } = {}) {
    if (!['sequence', 'previous_opponent', 'previous_player', 'switch'].includes(strategy)) throw Error('unknown strategy');
    if (!sequence.length || sequence.some(x => !Number.isInteger(x) || x < 0 || x > 2)) throw Error('invalid sequence');
    if (switchAfter < 1) throw Error('invalid switchAfter');
    if (noise < 0 || noise > 1) throw Error('invalid noise');
    Object.assign(this, { strategy, sequence, offset, switchAfter, secondOffset, noise });
    this.rand = rng(seed);
    this.initial = Math.floor(this.rand() * 3);
    this.previousPlayer = this.previousOpponent = this.previousResult = null;
    this.round = 0;
  }
  observe() {
    return { previousPlayer: this.previousPlayer, previousOpponent: this.previousOpponent,
      previousResult: this.previousResult };
  }
  move() {
    if (this.strategy === 'sequence') return this.sequence[(this.round + this.initial) % this.sequence.length];
    const previous = this.strategy === 'previous_player' ? this.previousPlayer : this.previousOpponent;
    if (previous === null) return this.initial;
    const offset = this.strategy === 'switch' && this.round >= this.switchAfter ? this.secondOffset : this.offset;
    return (previous + offset) % 3;
  }
  play(player) {
    if (!Number.isInteger(player) || player < 0 || player > 2) throw Error('invalid player move');
    let opponent = this.move();
    if (this.noise && this.rand() < this.noise) opponent = Math.floor(this.rand() * 3);
    const reward = result(player, opponent);
    this.previousPlayer = player; this.previousOpponent = opponent; this.previousResult = reward;
    this.round++;
    return { opponent, reward };
  }
}

export class RpsBioRL extends BioRL {
  constructor(net, opts = {}) {
    super(net, opts);
    if (opts.rateBaseline !== undefined) {
      if (!Array.isArray(opts.rateBaseline) || opts.rateBaseline.length !== 3 ||
        opts.rateBaseline.some(x => !Number.isFinite(x))) throw Error('invalid RPS rate baseline');
      this.o.rateBaseline = Object.freeze([...opts.rateBaseline]);
    }
  }
  decide(net) {
    const rates = net.groupMbon.map(ids => ids.reduce((s, i) => s + net.count[i], 0) / ids.length);
    const scores = this.o.rateBaseline ? rates.map((rate, a) => rate - this.o.rateBaseline[a]) : rates;
    const best = Math.max(...scores), winners = [0, 1, 2].filter(k => scores[k] === best);
    this.rates = rates;
    this.scores = scores;
    this.action = winners[Math.floor(this.rand() * winners.length)];
    this.V = Math.tanh(rates[this.action] / this.o.cRef);
    this.elig = Float64Array.from(this.plasticPre[this.action], p => net.trace[p]);
    return this.action;
  }
}

// Opt-in hybrid: original dopamine KC->MBON plasticity plus reward-trained MBON-spike readout.
export class RpsRewardReadout extends RpsBioRL {
  constructor(net, opts = {}) {
    super(net, opts);
    for (const key of ['featureMean', 'featureStd'])
      if (!Array.isArray(opts[key]) || opts[key].length !== 3 ||
        opts[key].some(x => !Number.isFinite(x) || (key === 'featureStd' && x <= 0)))
        throw Error(`invalid ${key}`);
    this.featureMean = [...opts.featureMean];
    this.featureStd = [...opts.featureStd];
    this.W = [0, 1, 2].map(() => new Float64Array(3));
    this.b = new Float64Array(3);
    this.baseline = 0;
    this.explore = true;
    this.readoutLearn = opts.readoutLearn !== false;
    this.freezeSynapses = opts.freezeSynapses === true;
    this.kind = 'rps_reward_readout';
  }
  decide(net) {
    this.rates = net.groupMbon.map(ids => ids.reduce((s, i) => s + net.count[i], 0) / ids.length);
    this.f = this.rates.map((rate, j) => (Math.log1p(rate) - this.featureMean[j]) / this.featureStd[j]);
    const logits = this.scores = this.logits(this.f);
    const mx = Math.max(...logits), ex = logits.map(x => Math.exp(x - mx));
    const z = ex.reduce((s, x) => s + x, 0);
    this.p = ex.map(x => x / z);
    let action;
    if (this.explore) {
      let u = this.rand(); action = 0;
      while (action < 2 && u > this.p[action]) u -= this.p[action++];
    } else action = logits.indexOf(mx);
    this.action = action;
    this.V = Math.tanh(this.rates[action] / this.o.cRef);
    this.elig = Float64Array.from(this.plasticPre[action], p => net.trace[p]);
    return action;
  }
  logits(f) { return this.W.map((w, a) => w.reduce((s, x, j) => s + x * f[j], this.b[a])); }
  reward(net, r) { this.r = r; super.reward(net, r); }
  afterDopamine(net, learn) {
    super.afterDopamine(net, learn && !this.freezeSynapses);
    if (!learn || !this.readoutLearn) return;
    const adv = this.r - this.baseline;
    this.baseline += DEEP_DEFAULTS.baselineRate * (this.r - this.baseline);
    this.policyUpdate(adv);
  }
  policyUpdate(adv) {
    for (let a = 0; a < 3; a++) {
      const g = DEEP_DEFAULTS.lr * adv * ((a === this.action ? 1 : 0) - this.p[a]);
      this.b[a] += g;
      for (let j = 0; j < 3; j++) this.W[a][j] += g * this.f[j];
    }
  }
}

// Opt-in 3 -> 8 tanh -> 3 REINFORCE policy; the MBON feature, softmax, reward baseline,
// and frozen-synapse behavior above remain shared with the linear readout.
export class RpsNonlinearReadout extends RpsRewardReadout {
  constructor(net, opts = {}) {
    super(net, opts);
    const random = rng((opts.seed ?? 1) * 65537 + 17), scale = Math.sqrt(6 / 11);
    this.W1 = Array.from({ length: 8 }, () => Float64Array.from({ length: 3 },
      () => (2 * random() - 1) * scale));
    this.b1 = new Float64Array(8);
    this.W = Array.from({ length: 3 }, () => new Float64Array(8));
    this.kind = 'rps_nonlinear_readout';
  }
  logits(f) {
    this.h = this.W1.map((w, k) => Math.tanh(w.reduce((s, x, j) => s + x * f[j], this.b1[k])));
    return this.W.map((w, a) => w.reduce((s, x, k) => s + x * this.h[k], this.b[a]));
  }
  policyUpdate(adv) {
    const g = [0, 1, 2].map(a => DEEP_DEFAULTS.lr * adv * ((a === this.action ? 1 : 0) - this.p[a]));
    const hiddenGrad = this.h.map((h, k) => (1 - h * h) *
      g.reduce((s, value, a) => s + value * this.W[a][k], 0));
    for (let a = 0; a < 3; a++) {
      this.b[a] += g[a];
      for (let k = 0; k < 8; k++) this.W[a][k] += g[a] * this.h[k];
    }
    for (let k = 0; k < 8; k++) {
      this.b1[k] += hiddenGrad[k];
      for (let j = 0; j < 3; j++) this.W1[k][j] += hiddenGrad[k] * this.f[j];
    }
  }
}

// Opt-in contextual bandit: predict each action's reward from the same three MBON features.
// The selected action alone receives a reward-prediction update; neural synapses stay frozen.
export class RpsContextualBandit extends RpsRewardReadout {
  constructor(net, opts = {}) {
    super(net, { ...opts, freezeSynapses: true });
    this.alpha = 0.03;
    this.epsilon = 0.20;
    this.kind = 'rps_contextual_bandit';
    if (opts.dnCountScale !== undefined) {
      if (!Number.isFinite(opts.dnCountScale) || opts.dnCountScale <= 0)
        throw Error('invalid DN count scale');
      this.dnCountScale = opts.dnCountScale;
      this.dnWindowCounts = null;
      this.W = [0, 1, 2].map(() => new Float64Array(5));
      this.kind = 'rps_contextual_bandit_dn';
    }
  }
  setDnWindowCounts(counts) {
    if (!this.dnCountScale || !Array.isArray(counts) || counts.length !== 2 ||
      counts.some(x => !Number.isInteger(x) || x < 0)) throw Error('invalid measured DN counts');
    this.dnWindowCounts = [...counts];
  }
  decide(net) {
    this.rates = net.groupMbon.map(ids => ids.reduce((s, i) => s + net.count[i], 0) / ids.length);
    this.f = this.rates.map((rate, j) => (Math.log1p(rate) - this.featureMean[j]) / this.featureStd[j]);
    if (this.dnCountScale) {
      if (!this.dnWindowCounts) throw Error('missing measured DN counts');
      this.f.push(...this.dnWindowCounts.map(x => x / this.dnCountScale));
      this.dnWindowCounts = null;
    }
    this.scores = this.logits(this.f);
    const greedy = this.scores.indexOf(Math.max(...this.scores));
    const u = this.explore ? this.rand() : 1;
    const action = this.action = u < this.epsilon ? Math.floor(3 * u / this.epsilon) : greedy;
    this.V = Math.tanh(this.rates[action] / this.o.cRef);
    this.elig = Float64Array.from(this.plasticPre[action], p => net.trace[p]);
    return action;
  }
  afterDopamine(net, learn) {
    RpsBioRL.prototype.afterDopamine.call(this, net, false);
    if (!learn || !this.readoutLearn) return;
    const a = this.action, error = this.r - this.scores[a];
    this.b[a] += this.alpha * error;
    for (let j = 0; j < this.f.length; j++) this.W[a][j] += this.alpha * error * this.f[j];
  }
}

export class ActivityChannel {
  constructor(sub, net, bandwidth = 3, targetWidth = 2) {
    if (!Number.isInteger(bandwidth) || bandwidth < 1 || bandwidth > 3 || targetWidth < 1 || targetWidth > 2) throw Error('invalid bandwidth or target width');
    if (MID_SOURCES.some(i => sub.neurons.role[i] !== 'mid')) throw Error('connectome does not match channel sources');
    this.source = MID_SOURCES.slice(0, bandwidth);
    this.target = Array.from({ length: bandwidth }, (_, k) => Array.from(net.laneVpn[k].slice(0, targetWidth)));
    this.bandwidth = bandwidth;
    this.log = [];
  }
  sample(teacherNet) { return this.source.map(i => Math.min(2, teacherNet.count[i])); }
  inject(studentNet, signal) {
    if (!Array.isArray(signal) || signal.length !== this.bandwidth || signal.some(x => !Number.isInteger(x) || x < 0 || x > 2)) throw Error('invalid signal');
    this.target.forEach((ids, k) => studentNet.setDrive(ids, signal[k] * SIGNAL_HZ));
    this.log.push([...signal]);
  }
  clear(studentNet) { this.target.forEach(ids => studentNet.setDrive(ids, 0)); }
}

export class RpsEpisode {
  constructor(net, agent, opponent, { rounds = 24, learn = true, channel = null, incomingSignals = null,
    captureSignals = false, balancedActions = false, balancedCounts = null } = {}) {
    if (incomingSignals && (!channel || incomingSignals.length !== rounds)) throw Error('invalid incoming signals');
    Object.assign(this, { net, agent, opponent, rounds, learn, channel, incomingSignals, captureSignals, balancedActions });
    this.t = 0; this.events = []; this.signals = [];
    this.balancedCounts = balancedCounts ?? [0, 0, 0];
    net.resetState();
  }
  get done() { return this.t >= this.rounds * ROUND_MS; }
  get winRate() { return this.events.length ? this.events.filter(e => e.reward === 1).length / this.events.length : 0; }
  step() {
    const round = Math.floor(this.t / ROUND_MS), phase = this.t % ROUND_MS;
    if (phase === 0) this.startRound();
    else if (phase === T_COUNT) this.net.resetCounts();
    else if (phase === T_SIGNAL) this.exchangeSignal(round);
    else if (phase === T_DECIDE) this.playRound(round);
    else if (phase === T_LEARN) this.agent.afterDopamine(this.net, this.learn);
    this.net.step();
    this.t++;
  }
  startRound() {
    this.obs = this.opponent.observe();
    if (this.obs.previousOpponent !== null)
      this.net.setDrive(this.net.laneVpn[this.obs.previousOpponent].slice(2), INPUT_HZ);
  }
  exchangeSignal(round) {
    if (this.captureSignals) this.signals.push(this.channel.sample(this.net));
    if (this.incomingSignals) this.channel.inject(this.net, this.incomingSignals[round]);
  }
  playRound(round) {
    if (this.obs.previousOpponent !== null)
      this.net.setDrive(this.net.laneVpn[this.obs.previousOpponent].slice(2), 0);
    if (this.incomingSignals) this.channel.clear(this.net);

    const nativeAction = this.agent.decide(this.net);
    let action = nativeAction;
    if (this.balancedActions && this.obs.previousOpponent !== null) {
      action = this.balancedCounts[this.obs.previousOpponent]++ % 3;
      this.agent.action = action;
      this.agent.V = Math.tanh(this.agent.rates[action] / this.agent.o.cRef);
      this.agent.elig = Float64Array.from(this.agent.plasticPre[action], i => this.net.trace[i]);
    }
    const { opponent, reward } = this.opponent.play(action);
    this.events.push({ round, observation: this.obs, action, opponent, reward, t: this.t,
      rates: [...this.agent.rates], nativeAction, signal: this.incomingSignals ? this.incomingSignals[round] : null });
    this.agent.reward(this.net, reward);
    this.net.resetCounts();
  }
  run() { while (!this.done) this.step(); return this; }
}
