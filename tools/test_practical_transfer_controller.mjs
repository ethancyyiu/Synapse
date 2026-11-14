import assert from 'node:assert/strict';
import { RecoveryBayesCore, decodeDn, nearestCue, netReward } from './practical_transfer_controller.mjs';

assert.deepEqual([0, 1, 2].map(a => netReward(1, a)), [.94, .88, .56]);
assert.equal(decodeDn([8, 0]), 0);
assert.equal(decodeDn([0, 8]), 1);
assert.equal(decodeDn([8, 7]), 2);
assert.equal(decodeDn([0, 0]), -1);
assert.equal(nearestCue([0, 1, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]), 1);
const c = new RecoveryBayesCore();
const initial = JSON.stringify(c.snapshot());
for (let i = 0; i < 18; i++) {
  const d = c.choose(2, true);
  assert.equal(d.action, i % 3);
  assert.equal(d.coverage, true);
  c.update(0, netReward(0, d.action));
}
assert.deepEqual(c.n[2], [6, 6, 6]);
assert.deepEqual(c.n[0], [0, 0, 0]);
assert.ok(c.values(2).sd[0] < new RecoveryBayesCore().values(2).sd[0]);
c.choose(2, true);
assert.throws(() => c.update(1, 99), /inconsistent/);
const before = JSON.stringify(c.snapshot());
c.imitate(1, false);
assert.equal(JSON.stringify(c.snapshot()), before);
c.imitate(1, true);
assert.equal(c.alpha[2][1], JSON.parse(before).alpha[2][1] + .25);
assert.equal(c.n[2][1], 6);
const restored = new RecoveryBayesCore();
restored.restore(c.snapshot());
assert.deepEqual(restored.snapshot(), c.snapshot());
const snapshot = JSON.stringify(restored.snapshot());
for (let i = 0; i < 10; i++) restored.choose(2, false);
assert.equal(JSON.stringify(restored.snapshot()), snapshot);
assert.equal(JSON.stringify(new RecoveryBayesCore().snapshot()), initial);
console.log('Practical transfer controller correctness passed.');
