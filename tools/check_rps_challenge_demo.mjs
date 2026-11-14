// Focused DOM and archived-data smoke; one live Fly A run is a UI check, not a research result.
// node tools/check_rps_challenge_demo.mjs
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Opponent } from '../js/rps.js';

const root = join(fileURLToPath(new URL('..', import.meta.url)));
const html = await readFile(join(root, 'index.html'), 'utf8');
const legacy = await readFile(join(root, 'legacy.html'), 'utf8');
assert.doesNotMatch(html, /id="game-title"|id="live-title"|id="findings-title"|src="js\/rps_app.js"/);
assert.match(html, /id="research-evidence"/);
assert.match(html, /id="brain-stage"/);
assert.ok(html.indexOf('Step 2 - Train Fly A') < html.indexOf('id="challenge-teacher-brain"') &&
  html.indexOf('id="challenge-teacher-brain"') < html.indexOf('Step 3 - Compare Fly B students') &&
  html.indexOf('Step 3 - Compare Fly B students') < html.indexOf('id="guided-brain"') &&
  html.indexOf('Step 3 - Compare Fly B students') < html.indexOf('id="alone-brain"'));
assert.doesNotMatch(html, /id="alone-brain-view"[^>]*hidden/);
assert.match(html, /<option value="noisy-a">Noisy circle/);
assert.match(html, /<option value="six">6 rounds, then continue to 288/);
assert.match(html, /id="challenge-fast"/);
assert.match(html, /measured 3D cell-body positions/);
assert.match(html, /id="brain-reset"/);
assert.match(legacy, /id="game-title"/);
assert.match(legacy, /id="live-title"/);
assert.match(html, /Step 1 - Set up the game.*?[\s\S]*Step 2 - Train Fly A.*?[\s\S]*Step 3 - Compare Fly B students.*?[\s\S]*Step 4 - Disconnect and compare/);
assert.match(html, /id="teach-dn-counts"/);
assert.match(html, /id="teach-update-detail"/);
const archive = await readFile(join(root, 'archive.html'), 'utf8');
const ids = [...(html + archive).matchAll(/\bid="([^"]+)"/g)].map(match => match[1]);
assert.equal(new Set(ids).size, ids.length, 'unique DOM ids');
class Element {
  constructor(tag = 'div') { this.tagName = tag.toUpperCase(); this.children = []; this.style = {}; this.hidden = false; this.textContent = ''; this.width = 320; this.height = 437; this.drawn = 0; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = [...children]; }
  getContext() { return this.context ??= { clearRect() {}, fillRect() {}, drawImage: () => { this.drawn++; } }; }
}
const elements = Object.fromEntries(ids.map(id => [id, new Element()]));
const $ = id => elements[id];
globalThis.document = { documentElement: {}, getElementById: id => {
  assert.ok($(id), `missing ${id}`); return $(id);
}, createElement: tag => new Element(tag) };
globalThis.getComputedStyle = () => ({ getPropertyValue: name => ({
  '--paper': '#15181c', '--ink': '#e6e8eb', '--graphite': '#a3abb4',
  '--gfp': '#5cc17a', '--magenta': '#e466b8',
})[name] });
globalThis.fetch = async path => {
  const bytes = await readFile(join(root, path));
  return { ok: true, json: async () => JSON.parse(bytes.toString('utf8')),
    arrayBuffer: async () => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) };
};
globalThis.requestAnimationFrame = callback => queueMicrotask(() => callback(performance.now()));
$('challenge-opponent').value = 'a';
$('challenge-seed').value = '830';
$('replay-seed').value = '851';
$('replay-arm').value = 'correct';
$('replay-checkpoint').value = '6';
$('ablation-evidence-seed').value = '11017';
const { analyzePattern, challengeOpponent, loadSpeedEvidence,
  loadActionPreferenceEvidence, loadAblationEvidence } = await import('../js/rps_challenge.js');
await loadSpeedEvidence();
assert.equal($('evidence-speed-table').children.length, 7);
assert.match($('evidence-speed-chart').innerHTML, /<svg/);
assert.match($('evidence-speed-table').children[2].children[1].textContent, /69\/72/);
assert.match($('evidence-speed-table').children[2].children[2].textContent, /24\/72/);
assert.match($('evidence-speed-status').textContent, /gate: failed/);
assert.match(html, /all four reported trained students reached at least 58\/72 frozen wins by round 6/);
assert.match(html, /This is a separate engineered hybrid imitation protocol/);
const preferenceFiles = {
  'a-11007': 'rps_action_preference_teaching_v3.json',
  'a-11008': 'rps_action_preference_teaching_v3.json',
  'a-11011': 'rps_action_preference_confirmation_first.json',
  'a-11012': 'rps_action_preference_confirmation_second.json',
  'b-11015': 'rps_action_preference_strategy_b_initial.json',
  'b-11016': 'rps_action_preference_strategy_b_confirmation.json',
};
for (const [selection, file] of Object.entries(preferenceFiles)) {
  $('preference-evidence-seed').value = selection;
  await loadActionPreferenceEvidence();
  const archive = JSON.parse(await readFile(join(root, 'data', file)));
  const seed = Number(selection.slice(2));
  const pair = archive.pilot.pairs?.find(x => x.seed === seed) ?? archive.pilot;
  assert.equal($('preference-evidence-table').children.length, 7,
    `${selection}: ${$('preference-evidence-status').textContent}`);
  assert.match($('preference-evidence-chart').innerHTML, /<svg/, selection);
  for (let i = 0; i < 7; i++) {
    const row = $('preference-evidence-table').children[i].children;
    assert.equal(row[0].textContent, pair.byArm.trained.checkpoints[i].rounds);
    for (const [j, arm] of ['trained', 'alone', 'untrained', 'shuffled', 'generic'].entries())
      assert.equal(row[j + 1].textContent, `${pair.byArm[arm].checkpoints[i].wins}/72`,
        `${selection}, round ${i}, ${arm}`);
  }
  assert.match($('preference-evidence-status').textContent, new RegExp(`student ${seed}`));
  const firstMessage = pair.byArm.trained.training.find(row => row.decoded >= 0);
  assert.match($('preference-event-dn').textContent,
    new RegExp(`neuron 3141 = ${firstMessage.dnCounts[0]}, neuron 3165 = ${firstMessage.dnCounts[1]}`));
  assert.match($('preference-event-update').textContent,
    new RegExp(`value step was ${firstMessage.rewardStep.toFixed(4)}`));
  assert.match($('preference-event-status').textContent, new RegExp(`student ${seed}`));
}
$('preference-evidence-seed').value = 'a-11007';
await Promise.all([loadActionPreferenceEvidence(), loadActionPreferenceEvidence()]);
assert.equal($('preference-evidence-table').children.length, 7,
  'overlapping archive loads render seven rows once');

const ablation = JSON.parse(await readFile(join(root, 'data',
  'rps_connectome_contribution_ablation.json')));
for (const seed of [11017, 11018]) {
  $('ablation-evidence-seed').value = String(seed);
  await loadAblationEvidence();
  const pair = ablation.pairs.find(p => p.studentSeed === seed);
  assert.equal($('ablation-evidence-table').children.length, 7,
    `${seed}: ${$('ablation-evidence-status').textContent}`);
  assert.equal(($('ablation-evidence-chart').innerHTML.match(/<path d=/g) ?? []).length, 5);
  assert.match($('ablation-evidence-chart').innerHTML, /stroke-dasharray="5 5"/);
  for (let i = 0; i < 7; i++) {
    const row = $('ablation-evidence-table').children[i].children;
    assert.equal(row[0].textContent, pair.byArm.neural.checkpoints[i].rounds);
    for (const [j, arm] of ['neural', 'digital', 'silent', 'scrambled'].entries())
      assert.equal(row[j + 1].textContent, `${pair.byArm[arm].checkpoints[i].wins}/72`);
  }
  assert.match($('ablation-evidence-status').textContent, new RegExp(`Student ${seed}`));
}
assert.match(html, /552\/552 trained neural messages decoded correctly/);
assert.match(html, /neural-spike delivery and direct digital delivery equally effective/);

for (const offset of [1, 2]) {
  const opponent = new Opponent({ strategy: 'previous_opponent', offset, noise: 0, seed: 41000 });
  const first = opponent.play(0).opponent;
  for (let n = 0, previous = first; n < 6; n++) {
    const next = opponent.play(0).opponent;
    assert.equal(next, (previous + offset) % 3);
    previous = next;
  }
}
assert.doesNotMatch(html, /<option value="0">Repeat/);
assert.match($('challenge-rule').textContent, /Strategy A: .*after Rock it plays Paper/);
assert.match($('challenge-pattern-preview').textContent, /^Rock → Paper → Scissors → Rock/);
assert.match($('challenge-answer-preview').textContent, /^Paper → Scissors → Rock → Paper/);
$('challenge-opponent').value = 'b'; $('challenge-opponent').onchange();
assert.match($('challenge-rule').textContent, /Strategy B: .*after Rock it plays Scissors/);
assert.match($('challenge-pattern-preview').textContent, /^Rock → Scissors → Paper/);
$('challenge-opponent').value = 'noisy-a'; $('challenge-opponent').onchange();
assert.match($('challenge-rule').textContent, /80% probability/);
assert.match($('challenge-rule').textContent, /teacher-specific benefit\) is unconfirmed/);
const noisyPair = [challengeOpponent({ mode: 'preset', offset: 1, noise: 0.3 }, 41000),
  challengeOpponent({ mode: 'preset', offset: 1, noise: 0.3 }, 41000)];
for (let round = 0; round < 72; round++)
  assert.equal(noisyPair[0].play(round % 3).opponent, noisyPair[1].play((round + 1) % 3).opponent,
    'noisy opponent streams stay matched across student actions');
for (const sequence of [[0], [0, 1], [0, 1, 2], [0, 1, 0, 2], [2, 0, 1, 2, 1]]) {
  for (const seed of [41000, 51001]) {
    const opponent = challengeOpponent({ mode: 'custom', sequence }, seed);
    for (let round = 0; round < 288; round++) {
      if (round) assert.equal(opponent.observe().previousOpponent,
        sequence[(round - 1) % sequence.length], `previous move at round ${round}`);
      assert.equal(opponent.play(0).opponent, sequence[round % sequence.length],
        `pattern length ${sequence.length}, seed ${seed}, round ${round}`);
    }
    const restarted = challengeOpponent({ mode: 'custom', sequence }, seed);
    assert.equal(restarted.play(0).opponent, sequence[0], 'reset restarts custom pattern');
    const pair = [challengeOpponent({ mode: 'custom', sequence }, seed),
      challengeOpponent({ mode: 'custom', sequence }, seed)];
    for (let round = 0; round < 30; round++)
      assert.equal(pair[0].play(round % 3).opponent, pair[1].play((round + 1) % 3).opponent,
        'students face the same rule regardless of their own action');
  }
}
assert.equal(analyzePattern([0, 1, 2]).distinguishable, true);
assert.deepEqual(analyzePattern([0, 1, 0, 2]).ambiguous, [0]);
assert.equal(analyzePattern([0, 1, 0, 2]).successors[0].includes(2), true,
  'wraparound transition is classified');
assert.equal(analyzePattern([0]).coverage.length, 1);
assert.throws(() => analyzePattern([]), /1–5/);
assert.throws(() => analyzePattern([0, 1, 2, 0, 1, 2]), /1–5/);
$('challenge-opponent').value = 'custom'; $('challenge-opponent').onchange();
assert.equal($('challenge-pattern-wrap').hidden, false);
assert.equal($('challenge-pattern-list').children.length, 3);
$('challenge-pattern-list').children[2].children[3].onclick(); // remove Scissors
$('challenge-pattern-list').children[1].children[3].onclick(); // remove Paper
assert.equal($('challenge-pattern-list').children.length, 1);
assert.match($('challenge-pattern-warning').textContent, /Only 1 of the 3 moves ever appear/);
$('challenge-add-paper').onclick();
$('challenge-add-rock').onclick();
$('challenge-add-scissors').onclick();
assert.equal($('challenge-pattern-list').children.length, 4);
assert.match($('challenge-pattern-warning').textContent, /after Rock it sometimes plays Paper and sometimes Scissors/);
assert.match($('challenge-pattern-warning').textContent, /Hard to predict/);
$('challenge-add-paper').onclick();
assert.equal($('challenge-pattern-list').children.length, 5);
assert.equal($('challenge-add-rock').disabled, true);
$('challenge-pattern-list').children[4].children[1].onclick(); // move last Paper up
assert.match($('challenge-pattern-preview').textContent, /Rock → Paper → Rock → Paper → Scissors/);
$('challenge-pattern-list').children[3].children[3].onclick(); // remove Paper, leave a length-4 ambiguous challenge
assert.equal($('challenge-pattern-list').children.length, 4);
assert.match($('challenge-rule').textContent, /Your pattern/);
console.log('preset rules, custom lengths 1–5, exact repetition, editor, and information warning: pass');

await $('replay-load').onclick();
assert.match($('replay-status').textContent, /archived replay loaded/i);
assert.equal($('replay-details').hidden, false);
assert.equal($('replay-controls').children.length, 5);
assert.equal($('replay-teacher-bars').children.length, 3);
assert.equal($('replay-student-bars').children.length, 3);
assert.match($('replay-dn').textContent, /3141 \(bit 0\) = 9, neuron 3165 \(bit 1\) = 0/);
assert.match($('replay-final').textContent, /Final choice: Paper.*Opponent: Paper.*0 tie/);
assert.match($('replay-frozen-detail').textContent, /70\/72 frozen disconnected wins/);
$('replay-checkpoint').value = '12'; $('replay-checkpoint').onchange();
assert.match($('replay-frozen-detail').textContent, /mean Q Rock/);
$('replay-arm').value = 'disconnected'; $('replay-arm').onchange(); $('replay-next').onclick();
assert.match($('replay-dn').textContent, /3141 \(bit 0\) = 0, neuron 3165 \(bit 1\) = 0/);
$('replay-seed').value = '852'; $('replay-seed').onchange();
assert.match($('replay-conclusion').textContent, /failed/);
assert.match($('replay-conclusion').textContent, /7 versus 4/);
console.log('archived teacher, student, DN, reward, frozen controls, and failed gates: pass');

assert.equal($('compare-start').disabled, true);
assert.equal($('alone-brain-view').hidden, false);
assert.equal($('guided-brain-view').hidden, false);
assert.match($('alone-status').textContent, /Idle/);
assert.match($('guided-status').textContent, /Idle/);
const teacherRun = $('challenge-train').onclick();
assert.equal($('compare-start').disabled, true);
assert.match($('alone-status').textContent, /Idle/);
assert.match($('guided-status').textContent, /Idle/);
await teacherRun;
assert.match($('challenge-teacher-progress').textContent, /288 \/ 288 training rounds/,
  $('challenge-status').textContent);
assert.match($('challenge-teacher-eval').textContent, /Test score: won \d+ of 72 new games with learning switched off/);
assert.match($('challenge-teacher-eval').textContent, /After each opponent move/);
assert.equal($('teacher-details').hidden, false);
assert.match($('challenge-status').textContent, /training and frozen evaluation complete/);
assert.equal($('challenge-teacher-bars').children.length, 3);
assert.ok($('challenge-teacher-brain').drawn > 1);
assert.match($('challenge-teacher-brain-header').textContent, /neurons simulated/);
assert.match($('challenge-teacher-q').textContent, /Rock .*Paper .*Scissors/);
assert.equal($('compare-start').disabled, false);
assert.match($('teacher-learning-chart').innerHTML, /<svg/);
assert.match($('teacher-learning-note').textContent, /After 288 rounds, Fly A won \d+\/72/);
assert.equal($('runs-table').children.length, 1);
await $('compare-start').onclick();
assert.doesNotMatch($('compare-status').textContent, /Comparison stopped/);
assert.match($('compare-round').textContent, /288 \/ 288/);
assert.match($('guided-dn').textContent, /3141 \d+.*3165 \d+.*decoded (Rock|Paper|Scissors)/);
assert.match($('guided-advice').textContent, /Fly A preference: (Rock|Paper|Scissors)/);
assert.match($('alone-q').textContent, /Own-reward update/);
assert.match($('guided-q').textContent, /Own-reward update/);
for (const id of ['alone', 'guided']) {
  assert.equal($(`${id}-brain-view`).hidden, false);
  assert.ok($(`${id}-brain`).drawn > 1);
  assert.match($(`${id}-brain-header`).textContent, /neurons simulated/);
}
assert.match($('compare-result').textContent, /Fly A disconnected.*No advice, exploration, or learning/);
assert.equal($('compare-cards').hidden, false);
assert.match($('compare-alone-score').textContent, /\d+\/72 wins/);
assert.match($('compare-guided-score').textContent, /\d+\/72 wins/);
assert.match($('compare-difference').textContent, /[+-]?\d+ wins/);
assert.match($('compare-alone-cues').textContent, /By previous move/);
assert.match($('compare-guided-cues').textContent, /By previous move/);
assert.equal($('compare-learning').hidden, false);
for (const round of [0, 6, 12, 24, 48, 96, 288])
  assert.match($('compare-checkpoints').textContent, new RegExp(`${round} rounds: Alone \\d+\\/72`));
assert.match($('compare-frozen-chart').innerHTML, /<svg/);
assert.match($('compare-training-chart').innerHTML, /<svg/);
assert.match($('compare-interpretation').textContent, /single-run early-advantage criterion/);
assert.match($('runs-table').children[0].children[6].textContent, /\d+\/72/);
assert.match($('runs-chart').innerHTML, /<svg/);
console.log('matched live students, measured DN advice, own-reward updates, disconnected frozen 72-round comparison: pass');
$('challenge-reset').onclick();
assert.match($('challenge-teacher-progress').textContent, /0 \/ 288/);
assert.equal($('challenge-train').disabled, false);
assert.equal($('compare-start').disabled, true);
assert.equal($('compare-cards').hidden, true);
assert.equal($('alone-brain-view').hidden, false);
assert.equal($('guided-brain-view').hidden, false);
assert.match($('alone-status').textContent, /Idle/);
assert.match($('guided-status').textContent, /Idle/);
assert.match(html, /href="rhythm.html"/);
assert.match(await readFile(join(root, 'rhythm.html'), 'utf8'), /js\/app.js/);
console.log('live custom-pattern Fly A training, frozen held-out check, reset, and rhythm link: pass');
