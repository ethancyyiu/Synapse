import assert from 'node:assert/strict';
import { rng } from '../js/sim.js';
import { RecoveryRidge } from './practical_transfer_redesign_controller.mjs';

const x = [1, .2, -.3, .4], c = new RecoveryRidge(4);
const before = c.predict(x);
assert.deepEqual(before.means, [0, 0, 0]);
assert.ok(before.uncertainties.every(v => v > 0 && Number.isFinite(v)));
const forced = c.choose(x, { training: true, forcedAction: 2,
  random: () => { throw Error('Forced choice consumed RNG'); } });
assert.equal(forced.action, 2);
assert.equal(forced.draws, null);
c.update(.69);
assert.deepEqual(c.n, [0, 0, 1]);
const after = c.predict(x);
assert.ok(after.uncertainties[2] < before.uncertainties[2]);
assert.equal(after.uncertainties[0], before.uncertainties[0]);
assert.equal(after.uncertainties[1], before.uncertainties[1]);
assert.ok(after.means[2] > 0);
assert.deepEqual(after.means.slice(0, 2), [0, 0]);
// Exact ridge posterior for one sample: x'x=1.29, mu=.69*1.29/(4+1.29).
assert.ok(Math.abs(after.means[2] - .69 * 1.29 / 5.29) < 1e-12);
const saved = JSON.stringify(c.snapshot());
const frozen = c.choose(x, { training: false,
  random: () => { throw Error('Frozen choice consumed RNG'); } });
assert.equal(frozen.action, 2);
assert.equal(frozen.draws, null);
assert.equal(JSON.stringify(c.snapshot()), saved);
assert.throws(() => c.update(.1));
const restored = new RecoveryRidge(4); restored.restore(JSON.parse(saved));
assert.deepEqual(restored.predict(x), c.predict(x));
function replay() {
  const agent = new RecoveryRidge(4), random = rng(20261109118), actions = [];
  for (let i = 0; i < 15; i++) {
    const decision = agent.choose([1, i / 15, -.2, .3], { training: true, random });
    actions.push(decision.action); agent.update((i % 3) / 3 - .3);
  }
  return { actions, parameters: agent.snapshot() };
}
assert.deepEqual(replay(), replay());
assert.throws(() => new RecoveryRidge(0));
assert.throws(() => restored.choose([1, 2], { training: false }));
console.log('Ridge controller checks passed: posterior, uncertainty, selected-arm update, forced coverage, deterministic replay, frozen choice');
