// Focused smoke for the live six-round pause; writes no research artifact.
// node tools/check_rps_six_round_demo.mjs
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { inlineCheckpoint, runStudents } from '../js/rps_student_engine.js';

const data = async name => JSON.parse(await readFile(new URL(`../data/${name}`, import.meta.url)));
const [sub, teacher, calibration] = await Promise.all([
  data('subgraph.json'),
  data('rps_action_preference_strategy_b_initial.json'),
  data('rps_contextual_bandit.json'),
]);
const control = { paused: false, slow: false, continueAfterSix: false };
const stream = runStudents({
  sub, seed: teacher.protocol.teacherSeed, config: { mode: 'preset', offset: 2 },
  teacherState: JSON.parse(teacher.teacher.finalWeights),
  featureMean: calibration.design.featureMean, featureStd: calibration.design.featureStd,
  pauseAtSix: true, control, runCheckpoint: inlineCheckpoint,
});
let rounds = 0, messages = 0, ready = false;
const points = [];
for await (const event of stream) {
  if (event.type === 'round') {
    rounds++;
    if (event.message.decoded !== null) {
      messages++;
      assert.equal(event.message.decoded, event.message.preference);
      assert.ok(event.message.dnCounts.some(n => n > 0));
    }
  }
  if (event.type === 'checkpoint') points.push(event.point);
  if (event.type === 'six_ready') {
    ready = true;
    assert.equal(rounds, 6);
    assert.deepEqual(points.map(p => p.rounds).sort((a, b) => a - b), [0, 6]);
    for (const point of points) {
      assert.equal(point.disconnected, true);
      assert.equal(point.weightsUnchanged, true);
      assert.equal(point.wins.length, 3);
      assert.equal(point.byCue.length, 2);
    }
    control.continueAfterSix = true;
  }
  if (ready && rounds === 7) {
    await stream.return();
    break;
  }
}
assert.ok(ready && messages > 0);
assert.equal(rounds, 7, 'the same run must resume at round seven');
console.log(`Six-round pause: 6 matched rounds, ${messages} measured DN messages, 0/6 frozen 72-round checks, resumed at round 7`);
