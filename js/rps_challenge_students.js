// Demo-only live version of the archived cue-head/DNge175 teaching schedule.
// The student readout and selected-action update match the validated opt-in runner.
import { rng } from './sim.js';
import { RpsContextualBandit, result } from './rps.js';

const DN = [3141, 3165];
const head = () => ({ W: [0, 1, 2].map(() => new Float64Array(3)), b: new Float64Array(3) });
export const studentWeights = agent => JSON.stringify({
  heads: agent.heads.map(h => ({ W: h.W.map(w => [...w]), b: [...h.b] })), baseline: agent.baseline,
});

export class ChallengeCueHeadBandit extends RpsContextualBandit {
  constructor(net, seed, featureMean, featureStd) {
    super(net, { seed, featureMean, featureStd, readoutLearn: true });
    this.heads = [head(), head(), head(), head()]; // fourth head: no previous cue
    this.activeHead = null;
    this.laneCounts = null;
    this.kind = 'rps_contextual_bandit_cue_heads_opt_in';
  }
  decide(net) {
    this.laneCounts = net.laneVpn.map(ids => Array.from(ids.slice(2))
      .reduce((sum, i) => sum + net.count[i], 0));
    const maximum = Math.max(...this.laneCounts);
    if (maximum > 0 && this.laneCounts.filter(v => v === maximum).length !== 1)
      throw Error('Ambiguous student sensory-lane activity');
    this.activeHead = maximum === 0 ? 3 : this.laneCounts.indexOf(maximum);
    const h = this.heads[this.activeHead];
    this.W = h.W; this.b = h.b;
    return super.decide(net);
  }
}

export function comparisonRound(teacher, alone, guided, key, learn = true, messageRule = null) {
  const students = [alone, guided];
  const cue = guided.game.observe().previousOpponent;
  if (alone.game.observe().previousOpponent !== cue) throw Error('Student opponent cues diverged');
  const vpn = net => cue === null ? [] : Array.from(net.laneVpn[cue].slice(2));
  const teacherVpn = teacher ? vpn(teacher.net) : [];
  const studentVpn = students.map(s => vpn(s.net));
  const rows = students.map(() => ({ cue, candidate: null, action: null, opponent: null,
    reward: null, rates: null, scores: null, learnedScores: null, activeKcs: 0,
    selectedHead: null, laneCounts: null, updateStep: null }));
  const dnCounts = [0, 0];
  let target = null, advice = null, teacherRates = null, teacherScores = null;
  let t = 0;

  return {
    get done() { return t === 680; },
    get rows() { return rows; },
    get message() { return { advice, dnCounts: [...dnCounts], teacherRates, teacherScores }; },
    step() {
      if (t >= 680) throw Error('Round already complete');
      if (t === 0) {
        students.forEach((s, i) => {
          s.net.resetState();
          s.net.rand = rng(key + 2);
          s.agent.rand = rng(key + 501);
          if (studentVpn[i].length) s.net.setDrive(studentVpn[i], 150);
        });
        if (teacher) {
          teacher.net.rand = rng(teacher.key + 1);
          if (teacherVpn.length) teacher.net.setDrive(teacherVpn, 150);
        }
      }
      if (t === 50) {
        students.forEach(s => s.net.resetCounts());
        if (teacher) teacher.net.resetCounts();
      }
      if (t === 240) {
        students.forEach((s, i) => {
          if (studentVpn[i].length) s.net.setDrive(studentVpn[i], 0);
          const candidate = s.agent.decide(s.net);
          Object.assign(rows[i], { candidate, rates: [...s.agent.rates],
            scores: [...s.agent.scores], selectedHead: s.agent.activeHead,
            laneCounts: [...s.agent.laneCounts],
            activeKcs: s.net.idx.kc.reduce((n, k) => n + Number(s.net.count[k] > 0), 0) });
          if (s.agent.activeHead !== (cue === null ? 3 : cue))
            throw Error('Student cue route did not match observed VPN input');
        });
        if (teacher) {
          if (teacherVpn.length) teacher.net.setDrive(teacherVpn, 0);
          teacher.agent.decide(teacher.net);
          teacherRates = [...teacher.agent.rates];
          teacherScores = [...teacher.agent.scores];
        }
        if (cue !== null && (teacher || messageRule)) {
          const candidate = rows[1].candidate;
          advice = messageRule ? messageRule({ cue, candidate, teacherScores }) :
            Number(teacherScores[candidate] >
              Math.max(...teacherScores.filter((_, action) => action !== candidate)));
          if (advice !== 0 && advice !== 1) throw Error('Invalid opt-in advice bit');
          target = DN[advice];
          guided.net.setDrive([target], 300);
        }
      }
      if (t === 280) {
        if (target !== null) guided.net.setDrive([target], 0);
        const decoded = target === null ? null :
          dnCounts[0] === dnCounts[1] ? null : Number(dnCounts[1] > dnCounts[0]);
        if (target !== null && decoded !== advice) throw Error('Measured DN advice did not decode');
        rows[1].decodedBit = decoded;
        rows[1].dnCounts = [...dnCounts];
        rows[0].decodedBit = null;
        rows[0].dnCounts = [0, 0];
        const alternatives = [0, 1, 2].filter(a => a !== rows[1].candidate);
        const guidedAction = decoded === 0 ? alternatives[Math.floor(rng(key + 701)() * 2)] :
          rows[1].candidate;
        [rows[0].candidate, guidedAction].forEach((action, i) => {
          const s = students[i], row = rows[i];
          s.agent.action = action;
          s.agent.V = Math.tanh(s.agent.rates[action] / s.agent.o.cRef);
          s.agent.elig = Float64Array.from(s.agent.plasticPre[action], p => s.net.trace[p]);
          const played = s.game.play(action);
          s.agent.reward(s.net, played.reward);
          row.action = action; row.opponent = played.opponent; row.reward = played.reward;
          if (row.reward !== result(action, row.opponent)) throw Error('RPS reward mismatch');
          s.net.resetCounts();
        });
        if (rows[0].opponent !== rows[1].opponent)
          throw Error('Student opponent streams diverged');
      }
      if (t === 620) students.forEach((s, i) => {
        const row = rows[i];
        const before = studentWeights(s.agent);
        const selected = s.agent.heads[row.selectedHead];
        const oldW = [...selected.W[row.action]], oldB = selected.b[row.action];
        row.updateStep = learn ? s.agent.alpha * (row.reward - row.scores[row.action]) : 0;
        s.agent.afterDopamine(s.net, learn);
        const expectedW = oldW.map((value, j) => value + row.updateStep * s.agent.f[j]);
        const prior = JSON.parse(before), current = JSON.parse(studentWeights(s.agent));
        const unrelatedChanged = prior.heads.some((h, headIndex) =>
          h.W.some((w, action) => (headIndex !== row.selectedHead || action !== row.action) &&
            JSON.stringify(w) !== JSON.stringify(current.heads[headIndex].W[action])) ||
          h.b.some((value, action) => (headIndex !== row.selectedHead || action !== row.action) &&
            value !== current.heads[headIndex].b[action]));
        if (selected.W[row.action].some((value, j) => Math.abs(value - expectedW[j]) > 1e-10) ||
            Math.abs(selected.b[row.action] - oldB - row.updateStep) > 1e-10 ||
            unrelatedChanged || current.baseline !== prior.baseline ||
            (!learn && studentWeights(s.agent) !== before))
          throw Error('Student update did not match its own reward and selected cue head');
        row.learnedScores = s.agent.logits(s.agent.f);
      });
      if (teacher) teacher.net.step();
      students.forEach(s => s.net.step());
      if (t >= 240 && t < 280) {
        for (let i = 0; i < guided.net.nSpikes; i++) {
          if (guided.net.spikes[i] === DN[0]) dnCounts[0]++;
          else if (guided.net.spikes[i] === DN[1]) dnCounts[1]++;
        }
      }
      t++;
    },
  };
}

// Live copy of the archived action-preference protocol (docs/action-preference-strategy-b-protocol.md).
const PREFERENCE_CODES = [[1, 0], [0, 1], [1, 1]];
const decodePreference = counts => {
  const bits = counts.map(count => Number(count > 0));
  const decoded = PREFERENCE_CODES.findIndex(code => code.every((bit, i) => bit === bits[i]));
  return decoded < 0 ? null : decoded;
};
export class PreferenceStudent extends ChallengeCueHeadBandit {
  imitate(decodedAction, rate = .03) {
    if (this.activeHead === 3 || decodedAction === null) return 0;
    const h = this.heads[this.activeHead], q = this.logits(this.f);
    const mx = Math.max(...q), ex = q.map(x => Math.exp(x - mx));
    const z = ex.reduce((a, b) => a + b, 0);
    for (let a = 0; a < 3; a++) {
      const g = rate * (Number(a === decodedAction) - ex[a] / z);
      h.b[a] += g;
      for (let j = 0; j < 3; j++) h.W[a][j] += g * this.f[j];
    }
    return 1;
  }
}
// Opt-in (docs/trust-gate-pilot-protocol.md): imitation scaled by a trust value the student
// learns from its own reward on rounds where it had already played the advised move.
export class TrustGatedStudent extends PreferenceStudent {
  trust = 1;
  imitate(decodedAction) {
    if (this.activeHead === 3 || decodedAction === null) return 0;
    if (this.action === decodedAction) this.trust += .2 * (Number(this.r === 1) - this.trust);
    return super.imitate(decodedAction, .03 * this.trust);
  }
}
export function preferenceRound(teacher, alone, guided, key, random = null, adviceEnabled = true) {
  const students = random ? [alone, guided, random] : [alone, guided];
  const cue = guided.game.observe().previousOpponent;
  if (alone.game.observe().previousOpponent !== cue) throw Error('Student opponent cues diverged');
  const vpn = net => cue === null ? [] : Array.from(net.laneVpn[cue].slice(2));
  const teacherVpn = vpn(teacher.net), studentVpn = students.map(s => vpn(s.net));
  const rows = students.map(() => ({ cue, candidate: null, action: null, opponent: null, reward: null,
    rates: null, scores: null, learnedScores: null, activeKcs: 0, selectedHead: null, updateStep: null,
    imitation: 0 }));
  const counts = students.map(() => [0, 0]), sent = students.map(() => null);
  const dnCounts = counts[1];
  let preference = null, decoded = null, t = 0;
  return {
    get done() { return t === 680; },
    get rows() { return rows; },
    get message() { return { preference, decoded, dnCounts: [...dnCounts] }; },
    step() {
      if (t >= 680) throw Error('Round already complete');
      if (t === 0) {
        students.forEach((s, i) => {
          s.net.resetState(); s.net.rand = rng(key + 2); s.agent.rand = rng(key + 501);
          if (studentVpn[i].length) s.net.setDrive(studentVpn[i], 150);
        });
        teacher.net.rand = rng(teacher.key + 1);
        if (teacherVpn.length) teacher.net.setDrive(teacherVpn, 150);
      }
      if (t === 50) { students.forEach(s => s.net.resetCounts()); teacher.net.resetCounts(); }
      if (t === 240) {
        students.forEach((s, i) => {
          if (studentVpn[i].length) s.net.setDrive(studentVpn[i], 0);
          const candidate = s.agent.decide(s.net);
          Object.assign(rows[i], { candidate, action: candidate, rates: [...s.agent.rates],
            scores: [...s.agent.scores], selectedHead: s.agent.activeHead,
            activeKcs: s.net.idx.kc.reduce((n, k) => n + Number(s.net.count[k] > 0), 0) });
          if (s.agent.activeHead !== (cue === null ? 3 : cue))
            throw Error('Student cue route did not match observed VPN input');
        });
        if (teacherVpn.length) teacher.net.setDrive(teacherVpn, 0);
        teacher.agent.decide(teacher.net);
        if (cue !== null && adviceEnabled) {
          const scores = [...teacher.agent.scores];
          preference = scores.indexOf(Math.max(...scores));
          sent[1] = preference;
          if (random) sent[2] = Math.floor(rng(key + 901)() * 3);
          students.forEach((s, i) => {
            if (sent[i] !== null) DN.forEach((id, j) => { if (PREFERENCE_CODES[sent[i]][j]) s.net.setDrive([id], 300); });
          });
        }
      }
      if (t === 280) {
        students.forEach((s, i) => {
          if (i) s.net.setDrive(DN, 0);
          const got = i ? decodePreference(counts[i]) : null;
          if (got !== sent[i]) throw Error('Measured DN preference did not decode');
          rows[i].decodedBit = got; rows[i].dnCounts = [...counts[i]];
        });
        decoded = rows[1].decodedBit;
        students.forEach((s, i) => {
          const played = s.game.play(rows[i].action);
          s.agent.reward(s.net, played.reward);
          rows[i].opponent = played.opponent; rows[i].reward = played.reward;
          if (rows[i].reward !== result(rows[i].action, rows[i].opponent)) throw Error('RPS reward mismatch');
          s.net.resetCounts();
        });
        if (rows[0].opponent !== rows[1].opponent) throw Error('Student opponent streams diverged');
      }
      if (t === 620) {
        students.forEach((s, i) => {
          rows[i].updateStep = s.agent.alpha * (rows[i].reward - rows[i].scores[rows[i].action]);
          s.agent.afterDopamine(s.net, true);
        });
        students.forEach((s, i) => { if (i) rows[i].imitation = s.agent.imitate(rows[i].decodedBit); });
        students.forEach((s, i) => { rows[i].learnedScores = s.agent.logits(s.agent.f); });
      }
      teacher.net.step();
      students.forEach(s => s.net.step());
      if (t >= 240 && t < 280) students.forEach((s, n) => {
        if (n) for (let i = 0; i < s.net.nSpikes; i++) {
          if (s.net.spikes[i] === DN[0]) counts[n][0]++;
          else if (s.net.spikes[i] === DN[1]) counts[n][1]++;
        }
      });
      t++;
    },
  };
}
