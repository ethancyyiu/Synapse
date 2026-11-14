// Locked, source-only RPS information study. No training or receiver.
// node tools/check_rps_teacher_source_information.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { RpsContextualBandit } from '../js/rps.js';

const output = new URL('../data/rps_teacher_source_information.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite teacher-source result');
const read = path => JSON.parse(readFileSync(new URL(path, import.meta.url)));
const saved = read('../data/rps_contextual_bandit.json');
const calibration = read('../data/rps_readout_information.json');
const sub = read('../data/subgraph.json');
const seeds = [712, 713, 714], states = [0, 1, 2], repeats = 12;
if (!saved.allValid || saved.passed || !calibration.passed ||
  JSON.stringify(calibration.design.calibrationSeeds) !== JSON.stringify([651, 652, 653]) ||
  JSON.stringify(saved.design.seeds) !== JSON.stringify([711, ...seeds]))
  throw Error('Archived protocol or calibration differs from locked source study');
const centroids = calibration.centroids;
const feature = rates => rates.map(Math.log1p);
const predict = rates => {
  const f = feature(rates);
  const distances = centroids.map(c => c.reduce((sum, x, j) => sum + (f[j] - x) ** 2, 0));
  return distances.indexOf(Math.min(...distances));
};
const countCorrect = rows => rows.filter(r => r.predictedPrevious === r.previous).length;
const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const sourceTrials = [], probes = [], perSeed = [];
for (const seed of seeds) {
  const student = saved.students.find(s => s.seed === seed);
  if (!student || !Object.values(student.validity).every(Boolean)) throw Error(`Invalid saved teacher ${seed}`);
  const trained = student.runs.find(r => r.arm === 'contextual_bandit');
  const untrained = student.runs.find(r => r.arm === 'contextual_bandit_no_learning');
  if (!trained || !untrained || trained.kcMbonChangedWeights || untrained.kcMbonChangedWeights ||
    !same(untrained.initialController, untrained.finalController) ||
    same(trained.initialController, trained.finalController)) throw Error(`Invalid source controls ${seed}`);
  const history = [trained, untrained].map((run, trainedFlag) => {
    const observed = run.after.events.filter(e => e.previous !== null);
    if (observed.length !== 69 || states.some(s => observed.filter(e => e.previous === s).length !== [23, 24, 22][s]))
      throw Error(`Invalid held-out schedule ${seed}`);
    return observed.map(e => {
      if (e.opponent !== (e.previous + 1) % 3 || e.rates.length !== 3 ||
        e.rates.some(x => !Number.isFinite(x) || x < 0)) throw Error('Invalid source trial');
      const predictedPrevious = predict(e.rates);
      const row = { seed, trained: Boolean(!trainedFlag), sequenceSeed: e.sequenceSeed, round: e.round,
        previous: e.previous, nextMove: e.opponent, rates: e.rates, predictedPrevious,
        predictedNextMove: (predictedPrevious + 1) % 3 };
      sourceTrials.push(row);
      return row;
    });
  });
  const historyDifference = history[0].filter((r, i) =>
    r.rates.some((x, j) => Math.abs(x - history[1][i].rates[j]) > 1e-9)).length;

  const copies = [trained, untrained].map(run => {
    const net = new Network(sub, wiring(sub, 'real', seed), { seed });
    const agent = new RpsContextualBandit(net, { seed,
      featureMean: saved.design.featureMean, featureStd: saved.design.featureStd,
      readoutLearn: false });
    const params = run.finalController;
    agent.W.forEach((row, a) => row.set(params.W[a]));
    agent.b.set(params.b); agent.baseline = params.baseline;
    agent.explore = false;
    return { net, agent, params };
  });
  for (const state of states) for (let repeat = 0; repeat < repeats; repeat++) {
    const responses = copies.map(({ net, agent }) => {
      net.resetState();
      net.rand = rng(seed * 100000 + repeat + 1);
      const input = net.laneVpn[state].slice(2);
      net.setDrive(input, 150);
      for (let t = 0; t < 240; t++) {
        if (t === 50) net.resetCounts();
        net.step();
      }
      net.setDrive(input, 0);
      const rates = net.groupMbon.map(ids => ids.reduce((sum, i) => sum + net.count[i], 0) / ids.length);
      const action = agent.decide(net);
      if (!same(rates, agent.rates)) throw Error('Source and controller features disagree');
      return { rates, action };
    });
    probes.push({ seed, state, repeat, trained: responses[0], untrained: responses[1],
      different: responses[0].rates.some((x, j) => Math.abs(x - responses[1].rates[j]) > 1e-9),
      actionsDiffer: responses[0].action !== responses[1].action });
  }
  if (copies.some(({ net, agent, params }) => !net.m.every(x => x === 1) ||
    !same({ W: agent.W.map(row => [...row]), b: [...agent.b], baseline: agent.baseline }, params)))
    throw Error(`Frozen probe changed parameters ${seed}`);
  perSeed.push({ seed, trainedCorrect: countCorrect(history[0]),
    untrainedCorrect: countCorrect(history[1]), historyDifference,
    probeDifferencesByState: states.map(state => probes.filter(p => p.seed === seed &&
      p.state === state && p.different).length),
    probeActionDifferencesByState: states.map(state => probes.filter(p => p.seed === seed &&
      p.state === state && p.actionsDiffer).length) });
}
const trainedRows = sourceTrials.filter(r => r.trained);
const untrainedRows = sourceTrials.filter(r => !r.trained);
const byState = states.map(state => {
  const t = trainedRows.filter(r => r.previous === state);
  const u = untrainedRows.filter(r => r.previous === state);
  return { state, total: t.length, trainedCorrect: countCorrect(t), untrainedCorrect: countCorrect(u) };
});
const shuffleRand = rng(20261104), shuffledScores = [];
for (let iteration = 0; iteration < 1000; iteration++) {
  let correct = 0;
  for (const seed of seeds) {
    const rows = trainedRows.filter(r => r.seed === seed);
    const labels = rows.map(r => r.previous);
    for (let i = labels.length - 1; i > 0; i--) {
      const j = Math.floor(shuffleRand() * (i + 1));
      [labels[i], labels[j]] = [labels[j], labels[i]];
    }
    correct += rows.filter((r, i) => r.predictedPrevious === labels[i]).length;
  }
  shuffledScores.push(correct);
}
const sorted = shuffledScores.toSorted((a, b) => a - b);
const shuffled95 = sorted[949];
const trainedCorrect = countCorrect(trainedRows), untrainedCorrect = countCorrect(untrainedRows);
const gates = { information: trainedCorrect / trainedRows.length >= 0.8 &&
    byState.every(x => x.trainedCorrect / x.total >= 0.7) && trainedCorrect > shuffled95,
  trainingDependent: perSeed.every(x => x.probeDifferencesByState.every(n => n >= 6)) };
const result = { command: 'node tools/check_rps_teacher_source_information.mjs',
  design: { source: 'three action-group MBON mean spike counts at 240 ms, reset at 50 ms',
    calibrationSeeds: [651, 652, 653], heldoutTeacherSeeds: seeds,
    opponent: 'previous_opponent offset 1 noise 0', excludedFirstRounds: 3,
    balancedChance: 69, shuffledPermutations: 1000, shuffleSeed: 20261104,
    informationGate: 'trained >=80% overall, >=70% each state, >95th-percentile shuffled score',
    trainingDependenceGate: '>=6/12 distinct matched predecision probes per state and teacher, tolerance 1e-9' },
  centroids, trainedCorrect, untrainedCorrect, byState, perSeed,
  shuffledMean: mean(shuffledScores), shuffled95,
  previousMoveLookupCorrect: trainedRows.length,
  conditionalNextMoveInformationBits: 0,
  gates, passed: gates.information && gates.trainingDependent,
  sourceTrials, probes };
writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({ trainedCorrect, untrainedCorrect, byState, perSeed,
  shuffledMean: result.shuffledMean, shuffled95, previousMoveLookupCorrect: result.previousMoveLookupCorrect,
  gates, passed: result.passed }, null, 2));
