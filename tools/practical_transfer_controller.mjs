// Isolated synthetic recovery controller. Existing RPS agents are untouched.
import { RpsContextualBandit } from '../js/rps.js';

const COST = [0.06, 0.12, 0.44];
export const netReward = (success, action) => {
  if (![0, 1].includes(success) || ![0, 1, 2].includes(action)) throw Error('Invalid outcome');
  return (success ? 1 : -0.4) - COST[action];
};
export const decodeDn = counts => {
  if (!Array.isArray(counts) || counts.length !== 2 ||
      counts.some(n => !Number.isInteger(n) || n < 0)) throw Error('Invalid DN counts');
  const bits = counts.map(n => Number(n > 0));
  return [[1, 0], [0, 1], [1, 1]].findIndex(code => code.every((v, i) => v === bits[i]));
};
export const nearestCue = (f, centroids) => {
  if (f.length !== 3 || f.some(x => !Number.isFinite(x)) || centroids.length !== 3 ||
      centroids.some(row => row.length !== 3 || row.some(x => !Number.isFinite(x))))
    throw Error('Invalid MBON cue features');
  const d = centroids.map(row => row.reduce((s, x, j) => s + (x - f[j]) ** 2, 0));
  return d.indexOf(Math.min(...d));
};

export class RecoveryBayesCore {
  constructor() {
    this.alpha = Array.from({ length: 3 }, () => [1, 1, 1]);
    this.beta = Array.from({ length: 3 }, () => [1, 1, 1]);
    this.n = Array.from({ length: 3 }, () => [0, 0, 0]);
    this.last = null;
  }
  values(cue) {
    if (![0, 1, 2].includes(cue)) throw Error('Invalid decoded cue');
    const p = [0, 1, 2].map(a => this.alpha[cue][a] / (this.alpha[cue][a] + this.beta[cue][a]));
    const sd = [0, 1, 2].map(a => {
      const x = this.alpha[cue][a], y = this.beta[cue][a], z = x + y;
      return Math.sqrt(x * y / (z * z * (z + 1)));
    });
    return { p, sd,
      means: p.map((v, a) => 1.4 * v - .4 - COST[a]),
      optimistic: p.map((v, a) => 1.4 * Math.min(1, v + sd[a]) - .4 - COST[a]) };
  }
  choose(cue, training = true) {
    const value = this.values(cue);
    let action;
    if (training && this.n[cue].some(n => n < 6)) {
      const m = Math.min(...this.n[cue]);
      action = this.n[cue].indexOf(m);
    } else {
      const scores = training ? value.optimistic : value.means;
      action = scores.indexOf(Math.max(...scores));
    }
    this.last = { cue, action };
    return { action, cue, ...value, coverage: training && this.n[cue].some(n => n < 6) };
  }
  update(success, reward) {
    if (!this.last || ![0, 1].includes(success) ||
        !Number.isFinite(reward) || Math.abs(reward - netReward(success, this.last.action)) > 1e-12)
      throw Error('Missing decision or inconsistent selected reward');
    const { cue, action } = this.last;
    this.alpha[cue][action] += success;
    this.beta[cue][action] += 1 - success;
    this.n[cue][action]++;
  }
  imitate(decodedAction, enabled = true) {
    if (!enabled || decodedAction === -1) return 0;
    if (!this.last || ![0, 1, 2].includes(decodedAction)) throw Error('Invalid decoded preference');
    this.alpha[this.last.cue][decodedAction] += .25;
    return 1;
  }
  snapshot() { return { alpha: this.alpha.map(x => [...x]), beta: this.beta.map(x => [...x]),
    n: this.n.map(x => [...x]) }; }
  restore(state) {
    for (const key of ['alpha', 'beta', 'n'])
      if (state[key]?.length !== 3 || state[key].some(row => row.length !== 3 ||
          row.some(x => !Number.isFinite(x) || x < (key === 'n' ? 0 : 1))))
        throw Error('Invalid recovery controller state');
    this.alpha = state.alpha.map(x => [...x]);
    this.beta = state.beta.map(x => [...x]);
    this.n = state.n.map(x => [...x]);
    this.last = null;
  }
}

export class RecoveryBayesAgent extends RpsContextualBandit {
  constructor(net, opts) {
    super(net, { ...opts, freezeSynapses: true });
    this.core = new RecoveryBayesCore();
    this.centroids = opts.centroids;
    this.kind = 'practical_recovery_bayes_opt_in';
  }
  decide(net) {
    this.rates = net.groupMbon.map(ids => ids.reduce((s, id) => s + net.count[id], 0) / ids.length);
    this.f = this.rates.map((rate, j) =>
      (Math.log1p(rate) - this.featureMean[j]) / this.featureStd[j]);
    this.decodedCue = nearestCue(this.f, this.centroids);
    const d = this.core.choose(this.decodedCue, this.explore);
    this.scores = d.means;
    this.selectionScores = this.explore ? d.optimistic : d.means;
    this.coverage = d.coverage;
    this.action = d.action;
    this.V = Math.tanh(this.rates[this.action] / this.o.cRef);
    this.elig = Float64Array.from(this.plasticPre[this.action], p => net.trace[p]);
    return this.action;
  }
  afterDopamine(net, learn, success) {
    super.afterDopamine(net, false);
    if (learn && this.readoutLearn) this.core.update(success, this.r);
  }
  imitate(decoded, enabled) { return this.core.imitate(decoded, enabled); }
}
