// Frozen comparison of the three archived nonlinear RPS students; no training.
// node tools/analyze_rps_nonlinear_seeds.mjs
import assert from 'node:assert/strict';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { Opponent, RpsEpisode, RpsNonlinearReadout, T_DECIDE } from '../js/rps.js';

const output = new URL('../data/rps_nonlinear_seed_comparison.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite frozen seed comparison');
const read = path => JSON.parse(readFileSync(new URL(path, import.meta.url)));
const sub = read('../data/subgraph.json');
const saved = read('../data/rps_nonlinear_readout.json');
const information = read('../data/rps_readout_information.json');
assert.equal(saved.allValid, true);
assert.deepEqual(saved.design.seeds, [681, 682, 683]);
assert.deepEqual(saved.design.heldoutSeeds, [51000, 51001, 51002]);
assert.equal(information.passed, true);
const { featureMean, featureStd } = saved.design;
const mean = xs => xs.reduce((s, x) => s + x, 0) / xs.length;
const sd = xs => Math.sqrt(mean(xs.map(x => (x - mean(xs)) ** 2)));
const vector = (rows, key) => [0, 1, 2].map(j => ({ mean: mean(rows.map(r => r[key][j])),
  sd: sd(rows.map(r => r[key][j])) }));
const correctAction = previous => [2, 0, 1][previous];
const softmax = scores => {
  const exp = scores.map(x => Math.exp(x - Math.max(...scores)));
  return exp.map(x => x / exp.reduce((s, v) => s + v, 0));
};
const margin = row => row.scores[correctAction(row.previous)] -
  Math.max(...row.scores.filter((_, a) => a !== correctAction(row.previous)));
const classify = rates => {
  const x = rates.map(Math.log1p);
  const distances = information.centroids.map(c => c.reduce((s, v, j) => s + (x[j] - v) ** 2, 0));
  return distances.indexOf(Math.min(...distances));
};
const controller = agent => ({ W: agent.W.map(row => [...row]), b: [...agent.b],
  W1: agent.W1.map(row => [...row]), b1: [...agent.b1], baseline: agent.baseline });
const scoredAction = (params, features) => {
  const h = params.W1.map((row, k) => Math.tanh(row.reduce((s, x, j) => s + x * features[j], params.b1[k])));
  const scores = params.W.map((row, a) => row.reduce((s, x, k) => s + x * h[k], params.b[a]));
  return scores.indexOf(Math.max(...scores));
};

const replay = saved.students.map(student => {
  const { seed } = student;
  const run = student.runs.find(x => x.arm === 'nonlinear_frozen');
  assert.ok(run && run.changedWeights.length === 0 && run.after.events.length === 72);
  const net = new Network(sub, wiring(sub, 'real', seed), { seed });
  const agent = new RpsNonlinearReadout(net, { seed, featureMean, featureStd,
    freezeSynapses: true, readoutLearn: false });
  agent.W.forEach((row, a) => row.set(run.finalController.W[a]));
  agent.b.set(run.finalController.b);
  agent.W1.forEach((row, k) => row.set(run.finalController.W1[k]));
  agent.b1.set(run.finalController.b1);
  agent.baseline = run.finalController.baseline;
  agent.explore = false;
  const initial = controller(agent);
  assert.deepEqual(initial, run.finalController);
  assert.ok(net.m.every(x => x === 1));
  const rows = [];
  saved.design.heldoutSeeds.forEach((sequenceSeed, i) => {
    const ep = new RpsEpisode(net, agent, new Opponent({ strategy: 'previous_opponent',
      offset: 1, noise: 0, seed: sequenceSeed }), { rounds: 24, learn: false });
    const decisions = [];
    while (!ep.done) {
      const phase = ep.t % 400;
      if (phase === 0) {
        const round = Math.floor(ep.t / 400);
        const key = seed * 10000000 + (100 + i) * 1000 + round;
        net.rand = rng(key + 1); agent.rand = rng(key + 501);
      }
      ep.step();
      if (phase === T_DECIDE) decisions.push({ features: [...agent.f], scores: [...agent.scores] });
    }
    const archive = run.after.events.filter(x => x.sequenceSeed === sequenceSeed);
    ep.events.forEach((event, round) => {
      const expected = archive[round], decision = decisions[round];
      assert.deepEqual([event.observation.previousOpponent, event.opponent, event.action, event.reward],
        [expected.previous, expected.opponent, expected.action, expected.reward]);
      assert.deepEqual(event.rates, expected.rates);
      assert.deepEqual(decision.features, expected.features);
      assert.deepEqual(decision.scores, expected.scores);
      assert.equal(event.action, decision.scores.indexOf(Math.max(...decision.scores)));
      rows.push({ previous: expected.previous, action: event.action, reward: event.reward,
        rates: event.rates, features: decision.features, scores: decision.scores });
    });
  });
  assert.deepEqual(controller(agent), initial);
  assert.ok(net.m.every(x => x === 1));
  assert.equal(rows.filter(r => r.reward === 1).length, run.after.wins);
  return { seed, params: initial, rows };
});

const students = replay.map(({ seed, rows }) => ({ seed,
  wins: rows.filter(r => r.reward === 1).length,
  states: [0, 1, 2].map(previous => {
    const xs = rows.filter(r => r.previous === previous);
    const wrong = xs.filter(r => r.action !== correctAction(previous));
    return { previous, trials: xs.length, accuracy: xs.filter(r => r.reward === 1).length,
      feature: vector(xs, 'features'), logits: vector(xs, 'scores'),
      correctMargin: { mean: mean(xs.map(margin)), sd: sd(xs.map(margin)) },
      correctProbability: { mean: mean(xs.map(r => softmax(r.scores)[correctAction(previous)])),
        sd: sd(xs.map(r => softmax(r.scores)[correctAction(previous)])) },
      selected: [0, 1, 2].map(a => xs.filter(r => r.action === a).length),
      fixedCentroidCorrect: xs.filter(r => classify(r.rates) === previous).length,
      fixedCentroidPredictions: [0, 1, 2].map(a => xs.filter(r => classify(r.rates) === a).length),
      wrong: { count: wrong.length, selectedProbabilityMean: wrong.length ?
        mean(wrong.map(r => softmax(r.scores)[r.action])) : null,
        correctProbabilityMean: wrong.length ?
          mean(wrong.map(r => softmax(r.scores)[correctAction(previous)])) : null,
        selectedProbabilityAtLeast07: wrong.filter(r => softmax(r.scores)[r.action] >= 0.7).length },
    };
  }),
}));
const crossPolicyState0 = replay.map(({ seed, rows }) => ({ neuralSeed: seed,
  controllers: replay.map(({ seed: controllerSeed, params }) => ({ controllerSeed,
    scissorsChoices: rows.filter(r => r.previous === 0 && scoredAction(params, r.features) === 2).length })) }));
const result = { command: 'node tools/analyze_rps_nonlinear_seeds.mjs',
  source: 'data/rps_nonlinear_readout.json',
  method: 'Exact frozen 72-round replay per seed; population SD; fixed pre-existing centroid; correct action used only for analysis',
  validation: { archivedTrialsMatched: 216, frozenWeightsAndControllerUnchanged: true },
  students, crossPolicyState0 };
writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({ students, crossPolicyState0 }, null, 2));
