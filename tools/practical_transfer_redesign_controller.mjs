// Isolated reward-only ridge/Thompson controller for the locked recovery redesign.
// Neither the RPS controllers nor the neural simulation import this file.
const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);
const matvec = (A, x) => A.map(row => dot(row, x));
const normal = random => {
  const u1 = Math.max(random(), 2 ** -32), u2 = random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
};

export class RecoveryRidge {
  constructor(d) {
    if (!Number.isInteger(d) || d < 2) throw Error('Invalid feature dimension');
    this.d = d;
    this.inv = Array.from({ length: 3 }, () => Array.from({ length: d }, (_, i) =>
      Array.from({ length: d }, (_, j) => i === j ? 1 / 4 : 0)));
    this.b = Array.from({ length: 3 }, () => Array(d).fill(0));
    this.n = [0, 0, 0];
    this.last = null;
  }
  predict(x) {
    if (x.length !== this.d || x.some(v => !Number.isFinite(v)))
      throw Error('Invalid features');
    const means = [], uncertainties = [];
    for (let a = 0; a < 3; a++) {
      const invX = matvec(this.inv[a], x);
      const variance = dot(x, invX);
      if (!Number.isFinite(variance) || variance < -1e-9)
        throw Error('Invalid ridge variance');
      means.push(dot(invX, this.b[a]));
      uncertainties.push(0.7 * Math.sqrt(Math.max(0, variance)));
    }
    return { means, uncertainties };
  }
  choose(x, { training = true, forcedAction = null, random = null } = {}) {
    const { means, uncertainties } = this.predict(x);
    let draws = null, action;
    if (!training) action = means.indexOf(Math.max(...means));
    else if (forcedAction !== null) {
      if (![0, 1, 2].includes(forcedAction)) throw Error('Invalid forced action');
      action = forcedAction;
    } else {
      if (typeof random !== 'function') throw Error('Missing Thompson RNG');
      draws = means.map((m, a) => m + uncertainties[a] * normal(random));
      action = draws.indexOf(Math.max(...draws));
    }
    this.last = training ? { x: [...x], action } : null;
    return { action, means, uncertainties, draws };
  }
  update(reward) {
    if (!this.last || !Number.isFinite(reward)) throw Error('Missing choice or invalid reward');
    const { x, action: a } = this.last;
    const invX = matvec(this.inv[a], x), denominator = 1 + dot(x, invX);
    if (!Number.isFinite(denominator) || denominator <= 0) throw Error('Invalid ridge update');
    for (let i = 0; i < this.d; i++) {
      this.b[a][i] += reward * x[i];
      for (let j = 0; j < this.d; j++)
        this.inv[a][i][j] -= invX[i] * invX[j] / denominator;
    }
    this.n[a]++;
    this.last = null;
  }
  snapshot() {
    return { d: this.d, inv: this.inv.map(A => A.map(row => [...row])),
      b: this.b.map(row => [...row]), n: [...this.n] };
  }
  restore(state) {
    if (state?.d !== this.d || state.inv?.length !== 3 || state.b?.length !== 3 ||
        state.n?.length !== 3 || state.inv.some(A => A.length !== this.d ||
        A.some(row => row.length !== this.d || row.some(v => !Number.isFinite(v)))) ||
        state.b.some(row => row.length !== this.d || row.some(v => !Number.isFinite(v))))
      throw Error('Invalid ridge snapshot');
    this.inv = state.inv.map(A => A.map(row => [...row]));
    this.b = state.b.map(row => [...row]);
    this.n = [...state.n];
    this.last = null;
  }
}
