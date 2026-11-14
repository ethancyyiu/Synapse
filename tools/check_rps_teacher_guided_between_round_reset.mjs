// Frozen, write-once student-only between-round reset feasibility probe.
// node tools/check_rps_teacher_guided_between_round_reset.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { RpsContextualBandit } from '../js/rps.js';

const output = new URL('../data/rps_teacher_guided_between_round_reset.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite between-round reset result');
const read = path => JSON.parse(readFileSync(new URL(path, import.meta.url)));
const prior = read('../data/rps_teacher_guided_exploration_preflight.json');
const sub = read('../data/subgraph.json');
const context = read('../data/rps_contextual_bandit.json');
const calibration = read('../data/rps_readout_information.json');
if (prior.stage !== 2 || prior.status !== 'stopped_receiver' ||
    prior.receiver.rows.length !== 216 || !context.allValid || !calibration.passed)
  throw Error('Archived receiving or decoder data changed');
const { featureMean, featureStd } = context.design;
const targets = [3141, 3165], seeds = [831, 832];
const branches = ['none', 'bit0', 'bit1'];
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const snapshot = a => ({ W: a.W.map(w => [...w]), b: [...a.b], baseline: a.baseline });
const decodeCue = rates => {
  const x = rates.map(Math.log1p);
  const d = calibration.centroids.map(c => c.reduce((s, v, j) => s + (x[j] - v) ** 2, 0));
  return d.indexOf(Math.min(...d));
};
const makeFly = (seed, trained = null) => {
  const net = new Network(sub, wiring(sub, 'real', seed), { seed });
  const agent = new RpsContextualBandit(net,
    { seed, featureMean, featureStd, readoutLearn: true });
  if (trained) {
    agent.W.forEach((w, a) => w.set(trained.W[a]));
    agent.b.set(trained.b); agent.baseline = trained.baseline;
  }
  agent.explore = false; net.resetState();
  return { net, agent, initial: snapshot(agent) };
};
function receiveRound(fly, cue, key, bit) {
  const { net, agent } = fly, vpn = Array.from(net.laneVpn[cue].slice(2));
  if (!vpn.length || vpn.some(i => targets.includes(i))) throw Error('Invalid VPN cue');
  net.rand = rng(key + 2); agent.rand = rng(key + 501);
  net.setDrive(vpn, 150);
  const dnCounts = [0, 0];
  let row, target = null;
  for (let t = 0; t < 680; t++) {
    if (t === 50) net.resetCounts();
    if (t === 240) {
      if (!vpn.every(i => net.drive[i] === 150)) throw Error('Cue drive missing');
      net.setDrive(vpn, 0);
      const candidate = agent.decide(net);
      if (bit !== null) { target = targets[bit]; net.setDrive([target], 300); }
      row = { cue, key, candidate, rates: [...agent.rates],
        features: [...agent.f], cueGuess: decodeCue(agent.rates),
        dnCounts: null, decodedBit: null, action: null };
    }
    if (t === 280) {
      if (target !== null) net.setDrive([target], 0);
      const decodedBit = target === null || dnCounts[0] === dnCounts[1] ? null :
        Number(dnCounts[1] > dnCounts[0]);
      const alternatives = [0, 1, 2].filter(a => a !== row.candidate);
      const action = decodedBit === 0 ?
        alternatives[Math.floor(rng(key + 701)() * 2)] : row.candidate;
      agent.action = action;
      agent.V = Math.tanh(agent.rates[action] / agent.o.cRef);
      agent.elig = Float64Array.from(agent.plasticPre[action], p => net.trace[p]);
      row.dnCounts = [...dnCounts]; row.decodedBit = decodedBit; row.action = action;
      net.resetCounts();
    }
    net.step();
    if (t >= 240 && t < 280) for (let k = 0; k < net.nSpikes; k++) {
      if (net.spikes[k] === targets[0]) dnCounts[0]++;
      else if (net.spikes[k] === targets[1]) dnCounts[1]++;
    }
  }
  if (!row || net.t !== 680 && net.t !== 1360 ||
      !same(snapshot(agent), fly.initial) || net.m.some(v => v !== 1))
    throw Error('Frozen student round invalid');
  return row;
}
function teacherTrace(cue, episodeKey, candidate) {
  const fly = makeFly(830, prior.teacher.final);
  const { net, agent } = fly, rows = [];
  for (let round = 0; round < 2; round++) {
    const state = round === 0 ? cue : (cue + 2) % 3;
    const key = 830 * 10000000 + episodeKey * 1000 + round;
    const vpn = Array.from(net.laneVpn[state].slice(2));
    net.rand = rng(key + 1); net.setDrive(vpn, 150);
    let row;
    for (let t = 0; t < 680; t++) {
      if (t === 50) net.resetCounts();
      if (t === 240) {
        net.setDrive(vpn, 0);
        agent.decide(net);
        const scores = [...agent.scores];
        row = { cue: state, key, rates: [...agent.rates], scores,
          approval: round === 0 ? Number(scores[candidate] >
            Math.max(...scores.filter((_, a) => a !== candidate))) : null };
      }
      if (t === 280) net.resetCounts();
      net.step();
    }
    row.netTimeAfterRound = net.t;
    rows.push(row);
  }
  return { rows, stateContinuous: rows[0].netTimeAfterRound === 680 &&
    rows[1].netTimeAfterRound === 1360,
  weightsUnchanged: same(snapshot(agent), fly.initial) && net.m.every(v => v === 1) };
}
const archived = prior.receiver.rows, rows = [];
for (const seed of seeds) for (let cue = 0; cue < 3; cue++)
  for (let repeat = 0; repeat < 12; repeat++) {
    const episodeKey = 200 + cue * 12 + repeat;
    const key = seed * 10000000 + episodeKey * 1000;
    const teacher = teacherTrace(cue, episodeKey,
      archived.find(r => r.seed === seed && r.cue === cue && r.repeat === repeat &&
        r.branch === 'none').first.candidate);
    for (const branch of branches) {
      const old = archived.find(r => r.seed === seed && r.cue === cue &&
        r.repeat === repeat && r.branch === branch);
      if (!old) throw Error('Missing archived matched condition');
      const fly = makeFly(seed);
      const bit = branch === 'none' ? null : Number(branch === 'bit1');
      const first = receiveRound(fly, cue, key, bit);
      const firstMatch = ['candidate', 'rates', 'features', 'cueGuess',
        'dnCounts', 'decodedBit', 'action'].every(k => same(first[k], old.first[k]));
      const retained = { candidate: first.candidate, action: first.action,
        decodedBit: first.decodedBit, dnCounts: [...first.dnCounts],
        teacherApproval: teacher.rows[0].approval };
      fly.net.resetState(); // Only new intervention: after first 680 ms, before next cue.
      const bufferPreserved = same(retained, { candidate: first.candidate,
        action: first.action, decodedBit: first.decodedBit,
        dnCounts: first.dnCounts, teacherApproval: teacher.rows[0].approval });
      const nextCue = (cue + 2) % 3;
      const second = receiveRound(fly, nextCue, key + 1, null);
      rows.push({ seed, cue, repeat, branch, teacher, first, second,
        firstMatch, bufferPreserved, resetApplied: true,
        weightsUnchanged: same(snapshot(fly.agent), fly.initial) && fly.net.m.every(v => v === 1) });
    }
  }
const mean = xs => xs.reduce((s, v) => s + v, 0) / xs.length;
const summary = [];
for (const seed of seeds) for (let cue = 0; cue < 3; cue++) {
  const grouped = branches.map(branch => rows.filter(r => r.seed === seed &&
    r.cue === cue && r.branch === branch));
  const old = branches.map(branch => archived.filter(r => r.seed === seed &&
    r.cue === cue && r.branch === branch));
  const base = grouped[0];
  summary.push({ seed, cue, nextCue: (cue + 2) % 3,
    branches: branches.map((branch, i) => {
      const x = grouped[i], bit = i === 0 ? null : i - 1;
      return { branch, n: x.length,
        continuousFirst: old[i].filter(r => r.first.cueGuess === cue).length,
        continuousSecond: old[i].filter(r => r.second.cueGuess === (cue + 2) % 3).length,
        resetFirst: x.filter(r => r.first.cueGuess === cue).length,
        resetSecond: x.filter(r => r.second.cueGuess === (cue + 2) % 3).length,
        gainedSecond: x.filter(r => r.second.cueGuess === (cue + 2) % 3 &&
          old[i].find(v => v.repeat === r.repeat).second.cueGuess !== (cue + 2) % 3).length,
        lostSecond: x.filter(r => r.second.cueGuess !== (cue + 2) % 3 &&
          old[i].find(v => v.repeat === r.repeat).second.cueGuess === (cue + 2) % 3).length,
        decoded: bit === null ? null : x.filter(r => r.first.decodedBit === bit).length,
        intendedMean: bit === null ? null : mean(x.map(r => r.first.dnCounts[bit])),
        noMessageTargetMean: bit === null ? null : mean(base.map(r => r.first.dnCounts[bit])),
        targetDominates: bit === null ? null : x.filter(r =>
          r.first.dnCounts[bit] > r.first.dnCounts[1 - bit]).length };
    }) });
}
const gates = {
  sampleIntegrity: rows.length === 216 && rows.every(r => r.resetApplied &&
    r.weightsUnchanged && r.firstMatch && r.bufferPreserved),
  recognitionFloor: summary.every(s => s.branches.every(b =>
    b.n === 12 && b.resetFirst >= 10 && b.resetSecond >= 10)),
  noMessageLoss: summary.every(s => s.branches.slice(1).every(b =>
    b.resetFirst >= s.branches[0].resetFirst &&
    b.resetSecond >= s.branches[0].resetSecond)),
  bitDecoding: summary.every(s => s.branches.slice(1).every(b => b.decoded === 12)),
  dnStimulation: summary.every(s => s.branches.slice(1).every(b =>
    b.targetDominates === 12 && b.intendedMean - b.noMessageTargetMean >= 1)),
  firstActionAndAdvice: rows.every(r => r.firstMatch && r.bufferPreserved &&
    (r.branch === 'none' || r.first.decodedBit === Number(r.branch === 'bit1'))),
  teacherContinuityAndWeights: rows.every(r => r.teacher.stateContinuous &&
    r.teacher.weightsUnchanged && r.weightsUnchanged)
};
const result = { protocol: { seeds, transitions: [[0, 2], [1, 0], [2, 1]], repeats: 12,
  branches, reset: 'student Network.resetState() after first 680ms round, before second 0ms cue',
  source: 'archived continuous-state receiver rows; frozen teacher 830 traced for source preservation',
  command: 'node tools/check_rps_teacher_guided_between_round_reset.mjs',
  gates: '>=10/12 first and second recognition; no branch count lower than no-message; 12/12 bit decoding/dominance; >=1 intended mean spike gain; buffer/proposal/teacher integrity' },
  rows, summary, gates, passed: Object.values(gates).every(Boolean) };
writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({ summary, gates, passed: result.passed }, null, 2));
