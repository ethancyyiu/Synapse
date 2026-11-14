// One focused live-engine smoke with archived teacher weights; no result artifact is written.
// node tools/check_rps_fast_preview.mjs [48]
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { runStudents } from '../js/rps_student_engine.js';

const data = async name => JSON.parse(await readFile(new URL(`../data/${name}`, import.meta.url)));
const [sub, teacher, calibration] = await Promise.all([
  data('subgraph.json'),
  data('rps_action_preference_strategy_b_initial.json'),
  data('rps_contextual_bandit.json'),
]);
const rounds = process.argv[2] === '48' ? 48 : 24;
const started = performance.now();
const seen = { round: 0, episode: 0, checkpoint: 0, final: 0, done: 0, message: 0 };
for await (const event of runStudents({
  sub, seed: teacher.protocol.teacherSeed, config: { mode: 'preset', offset: 2 },
  teacherState: JSON.parse(teacher.teacher.finalWeights),
  featureMean: calibration.design.featureMean, featureStd: calibration.design.featureStd,
  preview: true, previewEpisodes: rounds / 24, control: { paused: false, slow: false },
  runCheckpoint: () => { throw Error('Fast preview requested an expensive checkpoint'); },
})) {
  if (event.type in seen) seen[event.type]++;
  if (event.type === 'round' && event.message.decoded !== null) {
    seen.message++;
    assert.equal(event.message.decoded, event.message.preference);
    assert.ok(event.message.dnCounts.some(n => n > 0));
  }
  if (event.type === 'done') {
    assert.equal(event.totals.length, 2);
    for (const wins of event.totals) assert.ok(wins >= 0 && wins <= 72);
    for (const cues of event.byCue) assert.equal(cues.reduce((sum, cue) => sum + cue.n, 0), 69);
  }
}
assert.deepEqual({ ...seen, message: undefined },
  { round: rounds, episode: rounds / 24, checkpoint: 0, final: 1, done: 1, message: undefined });
assert.ok(seen.message > 0, 'actual DN messages were received');
console.log(`Fast preview: ${rounds} neural training rounds, ${seen.message} decoded messages, 72 disconnected evaluation rounds per student, no checkpoints; ${(performance.now() - started) / 1000}s`);
