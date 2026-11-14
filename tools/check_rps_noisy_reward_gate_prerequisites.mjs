// One prospectively locked, staged prerequisite test. No teacher-assisted learning.
// node tools/check_rps_noisy_reward_gate_prerequisites.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { Opponent, RpsContextualBandit, result } from '../js/rps.js';

const output = new URL('../data/rps_noisy_reward_gate_prerequisites.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite noisy prerequisite result');
const read = path => JSON.parse(readFileSync(new URL(path, import.meta.url)));
const sub = read('../data/subgraph.json');
const { featureMean, featureStd } = read('../data/rps_contextual_bandit.json').design;
const centroids = read('../data/rps_readout_information.json').centroids;
const students = [863, 864], teachers = [861, 862], targets = [3141, 3165];
const rounds = 24, episodes = 12, traceRounds = 288;
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const avg = xs => xs.reduce((a, b) => a + b, 0) / xs.length;
const count = (xs, value) => xs.filter(x => x === value).length;
const optimal = cue => cue; // Offset-2 opponent: cue beats (cue+2)%3.
const approve = (scores, a) => Number(scores[a] > Math.max(...scores.filter((_, i) => i !== a)));
const game = seed => new Opponent({ strategy: 'previous_opponent', offset: 2, noise: .5, seed });
const wires = new Map();
const netFor = seed => {
  if (!wires.has(seed)) wires.set(seed, wiring(sub, 'real', seed));
  return new Network(sub, wires.get(seed), { seed });
};
const makeHead = () => ({ W: [0, 1, 2].map(() => new Float64Array(3)), b: new Float64Array(3) });
class CueBandit extends RpsContextualBandit {
  constructor(net, seed) {
    super(net, { seed, featureMean, featureStd, readoutLearn: true });
    this.heads = [makeHead(), makeHead(), makeHead(), makeHead()];
    this.activeHead = null;
    this.laneCounts = null;
  }
  decide(net) {
    this.laneCounts = net.laneVpn.map(ids => Array.from(ids.slice(2))
      .reduce((sum, i) => sum + net.count[i], 0));
    const m = Math.max(...this.laneCounts);
    if (m > 0 && this.laneCounts.filter(x => x === m).length !== 1)
      throw Error('Ambiguous VPN-derived cue head');
    this.activeHead = m === 0 ? 3 : this.laneCounts.indexOf(m);
    this.W = this.heads[this.activeHead].W;
    this.b = this.heads[this.activeHead].b;
    return super.decide(net);
  }
}
const snapshot = a => ({ heads: a.heads.map(h => ({ W: h.W.map(w => [...w]), b: [...h.b] })),
  baseline: a.baseline });
function load(a, p) {
  for (let h = 0; h < 4; h++) for (let action = 0; action < 3; action++)
    a.heads[h].W[action].set(p.heads[h].W[action]);
  for (let h = 0; h < 4; h++) a.heads[h].b.set(p.heads[h].b);
  a.baseline = p.baseline;
}
function permutation(seed, cue) {
  const x = [0, 1, 2], rand = rng(73000 + 1000 * seed + cue);
  for (let i = 2; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [x[i], x[j]] = [x[j], x[i]];
  }
  return x;
}
const trainingSeed = (s, e) => 41000 + 1000 * s + e;
const evaluationSeed = (s, i) => 52000 + 1000 * s + 3 * i;
const sourceSeed = s => 63000 + 1000 * s;
function plannedMoves(seed, n) {
  const opponent = game(seed), out = [];
  for (let i = 0; i < n; i++) {
    const cue = opponent.observe().previousOpponent;
    const played = opponent.play(0); // Player action cannot affect previous-opponent process.
    out.push({ cue, opponent: played.opponent,
      base: cue === null ? opponent.initial : (cue + 2) % 3 });
  }
  return out;
}
function summarize(rows) {
  const observed = rows.filter(r => r.cue !== null);
  return { n: rows.length, wins: count(rows.map(r => r.reward), 1),
    observed: observed.length, expectedOptimal: count(observed.map(r => r.action === optimal(r.cue)), true),
    actions: [0, 1, 2].map(a => count(observed.map(r => r.action), a)),
    cueRouting: count(observed.map(r => r.estimatedCue === r.cue), true),
    byCue: [0, 1, 2].map(cue => {
      const x = observed.filter(r => r.cue === cue);
      return { cue, n: x.length, wins: count(x.map(r => r.reward), 1),
        optimal: count(x.map(r => r.action === optimal(cue)), true),
        actions: [0, 1, 2].map(a => count(x.map(r => r.action), a)),
        meanOptimalMargin: x.length ? avg(x.map(r => r.scores[optimal(cue)] -
          Math.max(...r.scores.filter((_, a) => a !== optimal(cue))))) : null };
    }) };
}
function runRound(net, agent, opponent, seed, episodeKey, round, learn, balanced, counters,
  candidatePermutations = null, planned = null) {
  const cue = opponent.observe().previousOpponent;
  const vpn = cue === null ? [] : Array.from(net.laneVpn[cue].slice(2));
  const key = seed * 10000000 + episodeKey * 1000 + round;
  net.rand = rng(key + 2); agent.rand = rng(key + 501);
  if (vpn.length) net.setDrive(vpn, 150);
  let row;
  for (let t = 0; t < 680; t++) {
    if (t === 50) net.resetCounts();
    if (t === 240) {
      if (vpn.length) net.setDrive(vpn, 0);
      const provisional = agent.decide(net), estimatedCue = agent.activeHead;
      let action = provisional, candidate = null, bit = null;
      if (balanced && estimatedCue !== 3) action = counters[estimatedCue]++ % 3;
      if (candidatePermutations && estimatedCue !== 3) {
        candidate = candidatePermutations[estimatedCue][counters[estimatedCue]++ % 3];
        bit = approve(agent.scores, candidate);
      }
      agent.action = action;
      agent.V = Math.tanh(agent.rates[action] / agent.o.cRef);
      agent.elig = Float64Array.from(agent.plasticPre[action], p => net.trace[p]);
      row = { round, cue, estimatedCue, key, provisional, action, candidate, bit,
        rates: [...agent.rates], features: [...agent.f], scores: [...agent.scores] };
    }
    if (t === 280) {
      const played = opponent.play(row.action);
      agent.reward(net, played.reward);
      row.opponent = played.opponent; row.reward = played.reward;
      if (planned && (planned.cue !== cue || planned.opponent !== row.opponent))
        throw Error('Causal opponent stream changed after action selection');
      if (row.candidate !== null) {
        row.candidateReward = result(row.candidate, row.opponent);
        row.expectedOptimalCandidate = row.candidate === optimal(cue);
      }
      net.resetCounts();
    }
    if (t === 620) agent.afterDopamine(net, learn);
    net.step();
  }
  if (net.m.some(x => x !== 1) || (cue !== null && vpn.length === 0))
    throw Error('Synapse integrity or VPN sensory pathway failed');
  return row;
}
function episode(seed, params, opponentSeed, episodeKey, n, learn, balanced, studentReset,
  permutations = null, planned = null) {
  const net = netFor(seed), agent = new CueBandit(net, seed);
  if (params) load(agent, params);
  agent.explore = learn; net.resetState();
  const before = snapshot(agent), opponent = game(opponentSeed), counters = [0, 0, 0], rows = [];
  for (let round = 0; round < n; round++) {
    if (studentReset && round) net.resetState();
    rows.push(runRound(net, agent, opponent, seed, episodeKey, round, learn, balanced,
      counters, permutations, planned?.[round]));
  }
  const after = snapshot(agent);
  if (!learn && !same(before, after)) throw Error('Frozen controller weights changed');
  return { before, after, rows, summary: summarize(rows), synapsesUnchanged: net.m.every(x => x === 1),
    finalNeuralTime: net.t, resets: studentReset ? n - 1 : 0 };
}
function frozen(seed, params, studentReset, plans) {
  const sequences = [0, 1, 2].map(i => episode(seed, params, evaluationSeed(seed, i),
    100 + i, rounds, false, false, studentReset, null, plans[i]));
  const rows = sequences.flatMap((e, i) => e.rows.map(r => ({ evalIndex: i, ...r })));
  return { rows, summary: summarize(rows), weightsUnchanged: sequences.every(e =>
    same(e.before, e.after) && e.synapsesUnchanged) };
}
const data = { protocol: { source: 'docs/lab-notebook.md locked completion before simulation',
  command: 'node tools/check_rps_noisy_reward_gate_prerequisites.mjs', students, teachers,
  opponent: 'previous_opponent offset 2 noise 0.5', training: '12x24', evaluation: '3x24',
  sourceTrace: 288, streamFormulas: { trainingSeed: '41000+1000*s+e',
    evaluationSeed: '52000+1000*s+3*i', sourceSeed: '63000+1000*s',
    neuralKey: 's*10000000+episodeKey*1000+round+2',
    actionKey: 's*10000000+episodeKey*1000+round+501',
    candidatePermutation: 'rng(73000+1000*s+cue)', shuffle: 'rng(74000+1000*s+cue)',
    receiver: 's*10000000+(300+12*cue+repeat)*1000' } },
  environment: null, students: [], teachers: [], source: [], receiver: null,
  status: 'not_started', stage: 0, gates: {} };
function stop(stage, status, gates) {
  data.stage = stage; data.status = status; data.gates = gates;
  writeFileSync(output, `${JSON.stringify(data, null, 2)}\n`, { flag: 'wx' });
  console.log(JSON.stringify({ stage, status, gates,
    students: data.students.map(x => ({ seed: x.seed, initial: x.initial?.summary,
      final: x.final?.summary })),
    teachers: data.teachers.map(x => ({ seed: x.seed, final: x.final?.summary })),
    source: data.source.map(x => ({ seed: x.seed, byCue: x.byCue })) }, null, 2));
  process.exit(0);
}

// Stage 1: inspect only the fixed opponent streams. No fly is simulated or trained.
const plans = new Map(), environment = [];
for (const seed of [...students, ...teachers]) {
  const training = Array.from({ length: episodes }, (_, e) => plannedMoves(trainingSeed(seed, e), rounds));
  const evaluation = [0, 1, 2].map(i => plannedMoves(evaluationSeed(seed, i), rounds));
  const source = teachers.includes(seed) ? plannedMoves(sourceSeed(seed), traceRounds) : null;
  plans.set(seed, { training, evaluation, source });
  const allTrain = training.flat(), allEval = evaluation.flat();
  environment.push({ seed, trainingCueCounts: [0, 1, 2].map(c => count(allTrain.map(r => r.cue), c)),
    evaluationCueCounts: [0, 1, 2].map(c => count(allEval.map(r => r.cue), c)),
    trainingNoiseDeviations: count(allTrain.map(r => r.opponent !== r.base), true),
    evaluationNoiseDeviations: count(allEval.map(r => r.opponent !== r.base), true),
    sourceCueCounts: source && [0, 1, 2].map(c => count(source.map(r => r.cue), c)) });
}
data.environment = { rows: environment, stationaryRule: true,
  causalCheck: 'Opponent.play samples after action; its actual move becomes next previous-opponent cue',
  theoreticalSuboptimalWinProbability: 1 / 6 };
const environmentGate = environment.every(x => x.trainingCueCounts.every(n => n > 0) &&
  x.evaluationCueCounts.every(n => n >= 15) && x.trainingNoiseDeviations > 0 &&
  x.evaluationNoiseDeviations > 0 && (!x.sourceCueCounts || x.sourceCueCounts.every(n => n > 0)));
if (!environmentGate) stop(1, 'stopped_environment', { environment: false });

// Stage 2: two independent no-message students, fixed budget and frozen streams.
for (const seed of students) {
  const plan = plans.get(seed), blank = new CueBandit(netFor(seed), seed);
  let params = snapshot(blank);
  const initial = frozen(seed, params, true, plan.evaluation), training = [];
  for (let e = 0; e < episodes; e++) {
    const run = episode(seed, params, trainingSeed(seed, e), e, rounds, true, true,
      true, null, plan.training[e]);
    params = run.after;
    training.push({ episode: e, summary: run.summary, rows: run.rows,
      synapsesUnchanged: run.synapsesUnchanged, resets: run.resets });
  }
  const final = frozen(seed, params, true, plan.evaluation);
  const initialRate = initial.summary.expectedOptimal / initial.summary.observed;
  const finalRate = final.summary.expectedOptimal / final.summary.observed;
  const gates = { coverage: final.summary.byCue.every(c => c.n >= 15),
    perCue: final.summary.byCue.every(c => c.optimal / c.n >= .70),
    gain: finalRate - initialRate >= .20,
    frozenWeights: initial.weightsUnchanged && final.weightsUnchanged,
    synapses: training.every(e => e.synapsesUnchanged),
    resets: training.every(e => e.resets === 23) };
  data.students.push({ seed, initial, training, final, gain: finalRate - initialRate, gates });
}
if (!data.students.every(x => Object.values(x.gates).every(Boolean)))
  stop(2, 'stopped_student_learning', { environment: true,
    student: data.students.map(x => ({ seed: x.seed, ...x.gates })) });

// Stage 3: train independently, then score frozen teacher policies.
for (const seed of teachers) {
  const plan = plans.get(seed), blank = new CueBandit(netFor(seed), seed);
  let params = snapshot(blank), training = [];
  for (let e = 0; e < episodes; e++) {
    const run = episode(seed, params, trainingSeed(seed, e), e, rounds, true, true,
      false, null, plan.training[e]);
    params = run.after;
    training.push({ episode: e, summary: run.summary, rows: run.rows,
      synapsesUnchanged: run.synapsesUnchanged, finalNeuralTime: run.finalNeuralTime });
  }
  const final = frozen(seed, params, false, plan.evaluation);
  const gates = { coverage: final.summary.byCue.every(c => c.n >= 15),
    perCue: final.summary.byCue.every(c => c.optimal / c.n >= .75),
    frozenWeights: final.weightsUnchanged,
    synapses: training.every(e => e.synapsesUnchanged),
    continuous: training.every(e => e.finalNeuralTime === rounds * 680) };
  data.teachers.push({ seed, training, final, params, gates });
}
if (!data.teachers.every(x => Object.values(x.gates).every(Boolean)))
  stop(3, 'stopped_teacher_learning', { environment: true, student: true,
    teacher: data.teachers.map(x => ({ seed: x.seed, ...x.gates })) });

function balanced(rows) {
  const positives = rows.filter(r => r.expectedOptimalCandidate),
    negatives = rows.filter(r => !r.expectedOptimalCandidate);
  if (!positives.length || !negatives.length) return null;
  return (count(positives.map(r => r.bit), 1) / positives.length +
    count(negatives.map(r => r.bit), 0) / negatives.length) / 2;
}
function shuffleP95(rows, seed, cue) {
  const rand = rng(74000 + 1000 * seed + cue), scores = [];
  for (let repeat = 0; repeat < 1000; repeat++) {
    const bits = rows.map(r => r.bit);
    for (let i = bits.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [bits[i], bits[j]] = [bits[j], bits[i]];
    }
    scores.push(balanced(rows.map((r, i) => ({ ...r, bit: bits[i] }))));
  }
  scores.sort((a, b) => a - b);
  return scores[949];
}

// Stage 4/5: one held-out, continuous-neural-state trace per frozen teacher.
for (const teacher of data.teachers) {
  const seed = teacher.seed, perms = [0, 1, 2].map(c => permutation(seed, c));
  const source = episode(seed, teacher.params, sourceSeed(seed), 200, traceRounds,
    false, false, false, perms, plans.get(seed).source);
  const byCue = [0, 1, 2].map(cue => {
    const x = source.rows.filter(r => r.cue === cue && r.candidate !== null);
    const lucky = x.filter(r => !r.expectedOptimalCandidate && r.candidateReward === 1);
    const good = x.filter(r => r.expectedOptimalCandidate && r.candidateReward === 1);
    const accuracy = balanced(x), shuffledP95 = shuffleP95(x, seed, cue);
    return { cue, n: x.length, candidates: [0, 1, 2].map(a => count(x.map(r => r.candidate), a)),
      bits: [0, 1].map(b => count(x.map(r => r.bit), b)), accuracy,
      shuffledP95, untrainedAccuracy: .5, lucky: lucky.length,
      rejectedLucky: count(lucky.map(r => r.bit), 0), good: good.length,
      approvedGood: count(good.map(r => r.bit), 1),
      sourceGate: accuracy !== null && accuracy >= .80 &&
        accuracy - shuffledP95 >= .15 && accuracy - .5 >= .15,
      opportunityGate: lucky.length >= 4 && good.length > 0 &&
        count(lucky.map(r => r.bit), 0) / lucky.length >= .75 &&
        count(good.map(r => r.bit), 1) / good.length >= .75 };
  });
  data.source.push({ seed, permutations: perms, rows: source.rows, byCue,
    weightsUnchanged: same(source.before, source.after) && source.synapsesUnchanged,
    continuous: source.finalNeuralTime === traceRounds * 680 });
}
if (!data.source.every(x => x.weightsUnchanged && x.continuous &&
    x.byCue.every(c => c.sourceGate && c.opportunityGate)))
  stop(5, 'stopped_source_or_opportunity', { environment: true, student: true,
    teacher: true, source: data.source.map(x => ({ seed: x.seed,
      cues: x.byCue.map(c => ({ cue: c.cue, source: c.sourceGate,
        opportunity: c.opportunityGate })) })) });

// Stage 6: frozen matched forced-bit two-round probes, with the prior decoder.
const decodeCue = rates => {
  const v = rates.map(Math.log1p);
  return centroids.map(c => c.reduce((s, x, j) => s + (v[j] - x) ** 2, 0))
    .indexOf(Math.min(...centroids.map(c => c.reduce((s, x, j) =>
      s + (v[j] - x) ** 2, 0))));
};
function receiverRound(net, agent, cue, key, bit) {
  const vpn = Array.from(net.laneVpn[cue].slice(2)), dnCounts = [0, 0];
  net.rand = rng(key + 2); agent.rand = rng(key + 501);
  net.setDrive(vpn, 150);
  let row;
  for (let t = 0; t < 680; t++) {
    if (t === 50) net.resetCounts();
    if (t === 240) {
      net.setDrive(vpn, 0); agent.decide(net);
      row = { cue, estimatedCue: agent.activeHead, rates: [...agent.rates],
        recognized: decodeCue(agent.rates) === cue, bit, dnCounts: null, decodedBit: null };
      if (bit !== null) net.setDrive([targets[bit]], 300);
    }
    if (t === 280) {
      if (bit !== null) net.setDrive([targets[bit]], 0);
      row.dnCounts = [...dnCounts];
      row.decodedBit = dnCounts[0] === dnCounts[1] ? null : Number(dnCounts[1] > dnCounts[0]);
      net.resetCounts();
    }
    net.step();
    if (t >= 240 && t < 280) for (let j = 0; j < net.nSpikes; j++) {
      if (net.spikes[j] === targets[0]) dnCounts[0]++;
      else if (net.spikes[j] === targets[1]) dnCounts[1]++;
    }
  }
  return row;
}
const probes = [];
for (const seed of students) for (let cue = 0; cue < 3; cue++)
  for (let repeat = 0; repeat < 12; repeat++) for (const bit of [null, 0, 1]) {
    const net = netFor(seed), agent = new CueBandit(net, seed), before = snapshot(agent);
    agent.explore = false; net.resetState();
    const key = seed * 10000000 + (300 + 12 * cue + repeat) * 1000;
    const first = receiverRound(net, agent, cue, key, bit);
    const retained = [...first.dnCounts];
    net.resetState();
    const bufferPreserved = same(first.dnCounts, retained);
    const second = receiverRound(net, agent, (cue + 2) % 3, key + 1, null);
    probes.push({ seed, cue, repeat, bit, first, second, bufferPreserved,
      weightsUnchanged: same(before, snapshot(agent)) && net.m.every(x => x === 1) });
  }
const receiverSummary = students.flatMap(seed => [0, 1, 2].map(cue => {
  const x = probes.filter(p => p.seed === seed && p.cue === cue);
  const base = x.filter(p => p.bit === null);
  return { seed, cue, branches: [null, 0, 1].map(bit => {
    const y = x.filter(p => p.bit === bit);
    return { bit, n: y.length, firstRecognized: count(y.map(p => p.first.recognized), true),
      secondRecognized: count(y.map(p => p.second.recognized), true),
      decoded: bit === null ? null : count(y.map(p => p.first.decodedBit === bit), true),
      intendedMeanGain: bit === null ? null : avg(y.map(p => p.first.dnCounts[bit])) -
        avg(base.map(p => p.first.dnCounts[bit])),
      targetDominance: bit === null ? null : count(y.map(p =>
        p.first.dnCounts[bit] > p.first.dnCounts[1 - bit]), true) };
  }) };
}));
const receiverGates = { probes: probes.length === 216 &&
    probes.every(p => p.bufferPreserved && p.weightsUnchanged),
  recognition: receiverSummary.every(s => s.branches.every(b =>
    b.n === 12 && b.firstRecognized >= 10 && b.secondRecognized >= 10)),
  noLargeLoss: receiverSummary.every(s => s.branches.slice(1).every(b =>
    b.firstRecognized >= s.branches[0].firstRecognized - 2 &&
    b.secondRecognized >= s.branches[0].secondRecognized - 2)),
  bit: receiverSummary.every(s => s.branches.slice(1).every(b =>
    b.decoded === 12 && b.targetDominance === 12 && b.intendedMeanGain >= 1)) };
data.receiver = { probes, summary: receiverSummary, gates: receiverGates };
stop(6, Object.values(receiverGates).every(Boolean) ? 'passed' : 'stopped_receiver',
  { environment: true, student: true, teacher: true, source: true, receiver: receiverGates });
