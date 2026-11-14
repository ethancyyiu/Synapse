// DOM-level smoke check for the browser controller when a browser is unavailable.
// Run: node tools/check_rps_demo.mjs
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { result, MOVES, Opponent, RpsEpisode, T_DECIDE } from '../js/rps.js';
import { rng } from '../js/sim.js';

const root = join(fileURLToPath(new URL('..', import.meta.url)));
const html = await readFile(join(root, 'legacy.html'), 'utf8');
const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(x => x[1]);
assert.equal(new Set(ids).size, ids.length, 'duplicate HTML ids');
const ctx = new Proxy({}, { get: (_, key) => key === 'canvas' ? null : () => {} });
class Element {
  constructor(tag = 'div') {
    this.tag = tag; this.children = []; this.style = {}; this.dataset = {};
    this.classList = { toggle() {} }; this.textContent = ''; this.hidden = false;
    this.width = 800; this.height = 600;
  }
  getContext() { return ctx; }
  append(...children) { this.children.push(...children); }
  prepend(...children) { this.children.unshift(...children); }
  replaceChildren(...children) { this.children = [...children]; }
  createTHead() { const child = new Element('thead'); this.append(child); return child; }
  createTBody() { const child = new Element('tbody'); this.append(child); return child; }
  insertRow() { const child = new Element('tr'); this.append(child); return child; }
  insertCell() { const child = new Element('td'); this.append(child); return child; }
  querySelectorAll(selector) {
    if (selector !== '[data-move]') return [];
    return this.moveButtons ??= [0, 1, 2].map(n => {
      const button = new Element('button'); button.dataset.move = String(n); return button;
    });
  }
}
const elements = Object.fromEntries(ids.map(id => [id, new Element()]));
elements['game-model'].value = 'bio';
elements['game-model-b'].value = 'bio';
elements['game-seed-a'].value = '681';
elements['game-seed-b'].value = '682';
elements['game-opponent'].value = 'human';
elements.speed.value = 'max';
globalThis.document = {
  documentElement: new Element(),
  getElementById(id) { assert.ok(elements[id], `missing element ${id}`); return elements[id]; },
  createElement(tag) { return new Element(tag); },
};
globalThis.getComputedStyle = () => ({ getPropertyValue(key) {
  return ({ '--paper': '#15181c', '--ink': '#e6e8eb', '--graphite': '#a3abb4',
    '--gfp': '#5cc17a', '--magenta': '#e466b8' })[key];
} });
globalThis.fetch = async url => {
  const data = await readFile(join(root, url));
  return { ok: true, json: async () => JSON.parse(data.toString('utf8')),
    arrayBuffer: async () => data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) };
};
let nextFrame;
globalThis.requestAnimationFrame = fn => { nextFrame = fn; };
const app = await import('../js/rps_app.js');
const archive = JSON.parse(await readFile(join(root, 'data/rps_nonlinear_readout.json')));
assert.equal(archive.allValid, true);
assert.equal(archive.passed, false);
assert.match(html, /locked all-student success criterion failed/);
for (const student of archive.students) {
  const { seed } = student;
  const run = student.runs.find(x => x.arm === 'nonlinear_frozen');
  const { net, agent } = app.gameFly(seed, 'nonlinear');
  assert.equal(agent.kind, 'rps_nonlinear_readout');
  assert.equal(agent.explore, false);
  assert.equal(agent.readoutLearn, false);
  assert.equal(agent.freezeSynapses, true);
  assert.ok(net.m.every(x => x === 1), `saved ${seed} KC-to-MBON multipliers`);
  assert.deepEqual(agent.W.map(row => [...row]), run.finalController.W);
  assert.deepEqual(agent.b, Float64Array.from(run.finalController.b));
  assert.deepEqual(agent.W1.map(row => [...row]), run.finalController.W1);
  assert.deepEqual(agent.b1, Float64Array.from(run.finalController.b1));
  assert.equal(agent.baseline, run.finalController.baseline);
  assert.deepEqual(agent.featureMean, archive.design.featureMean);
  assert.deepEqual(agent.featureStd, archive.design.featureStd);
  for (let sequenceIndex = 0; sequenceIndex < archive.design.heldoutSeeds.length; sequenceIndex++) {
    const sequenceSeed = archive.design.heldoutSeeds[sequenceIndex];
    const ep = new RpsEpisode(net, agent, new Opponent({
      strategy: 'previous_opponent', offset: 1, noise: 0, seed: sequenceSeed,
    }), { rounds: 24, learn: false });
    const decisions = [];
    while (!ep.done) {
      const phase = ep.t % 400;
      if (phase === 0) {
        const round = Math.floor(ep.t / 400);
        const key = seed * 10000000 + (100 + sequenceIndex) * 1000 + round;
        net.rand = rng(key + 1); agent.rand = rng(key + 501);
      }
      ep.step();
      if (phase === T_DECIDE) decisions.push({ features: [...agent.f], scores: [...agent.scores] });
    }
    const saved = run.after.events.filter(e => e.sequenceSeed === sequenceSeed);
    assert.equal(ep.events.length, saved.length);
    ep.events.forEach((event, round) => {
      const expected = saved[round], decision = decisions[round];
      assert.deepEqual([event.observation.previousOpponent, event.opponent, event.action, event.reward],
        [expected.previous, expected.opponent, expected.action, expected.reward]);
      assert.deepEqual(event.rates, expected.rates, `saved ${seed} MBON rates`);
      assert.deepEqual(decision.features, expected.features, `saved ${seed} neural features`);
      assert.deepEqual(decision.scores, expected.scores, `saved ${seed} three action scores`);
      assert.equal(event.action, decision.scores.indexOf(Math.max(...decision.scores)));
    });
  }
  assert.ok(net.m.every(x => x === 1), `frozen ${seed} neural weights`);
  assert.deepEqual(agent.W.map(row => [...row]), run.finalController.W);
  assert.deepEqual(agent.W1.map(row => [...row]), run.finalController.W1);
  assert.deepEqual(agent.b, Float64Array.from(run.finalController.b));
  assert.deepEqual(agent.b1, Float64Array.from(run.finalController.b1));
  assert.equal(agent.baseline, run.finalController.baseline);
}
console.log('nonlinear 681-683 restore and exact archived frozen MBON/action replay: pass');
const $ = id => elements[id];
let now = performance.now();
function tick() { const frame = nextFrame; assert.equal(typeof frame, 'function'); frame(now += 16); }
function until(rows) {
  for (let i = 0; i < 2000 && $('game-history').children.length < rows; i++) tick();
  assert.equal($('game-history').children.length, rows, 'round did not finish');
}
function verifyRows() {
  for (const row of $('game-history').children) {
    const [, , flyMove, opponentMove, outcome] = row.children.map(cell => cell.textContent);
    assert.equal(outcome, result(MOVES.indexOf(flyMove), MOVES.indexOf(opponentMove)) > 0 ? 'win' :
      result(MOVES.indexOf(flyMove), MOVES.indexOf(opponentMove)) < 0 ? 'loss' : 'tie');
  }
}
assert.match($('game-status').textContent, /Choose a move/);
assert.equal(app.game[0].agent.kind, 'bio');
assert.equal(app.game[1].agent.kind, 'bio');
const [rockButton] = $('game-human').querySelectorAll('[data-move]');
rockButton.onclick();
until(2);
verifyRows();
assert.ok($('game-history').children.every(row => row.children[3].textContent === 'Rock'));
for (const id of ['a', 'b']) {
  assert.match($(`game-${id}-header`).textContent, /neurons simulated/);
  assert.match($(`game-${id}-kc`).textContent, /^\d+$/);
  for (let action = 0; action < 3; action++) assert.ok(Number.isFinite(Number($(`game-${id}-mbon-${action}`).textContent)));
}
console.log('human round, live KC/MBON display, outcomes: pass');

$('game-opponent').value = 'pattern'; $('game-opponent').onchange();
assert.equal($('game-human').hidden, true);
assert.equal($('game-pattern').hidden, false);
$('game-next').onclick(); until(2); verifyRows();
$('game-watch').onclick();
for (let i = 0; i < 2000 && $('game-history').children.length < 4; i++) tick();
assert.ok($('game-history').children.length >= 4, 'watch mode did not advance');
$('game-watch').onclick();
console.log('patterned next/watch, two flies, outcomes: pass');

$('game-model').value = 'hybrid'; $('game-model').onchange();
assert.match($('game-model-note').textContent, /Experimental linear hybrid/);
assert.equal(app.game[0].agent.kind, 'rps_reward_readout');
assert.equal(app.game[1].agent.kind, 'bio');
assert.equal($('game-history').children.length, 0);
const linearWeights = app.game.map(fly => fly.net.m.slice());
const linearReadout = JSON.stringify(app.game[0].agent.W.map(row => [...row]));
const linearBaseline = app.game[0].agent.baseline;
$('game-next').onclick(); until(2); verifyRows();
app.game.forEach((fly, i) => assert.deepEqual(fly.net.m, linearWeights[i]));
assert.equal(JSON.stringify(app.game[0].agent.W.map(row => [...row])), linearReadout);
assert.equal(app.game[0].agent.baseline, linearBaseline);
$('game-reset').onclick();
assert.equal($('game-history').children.length, 0);
assert.equal($('game-a-score').textContent, '0 wins / 0 rounds');
assert.match(html, /href="rhythm.html"/);
assert.match(await readFile(join(root, 'rhythm.html'), 'utf8'), /js\/app.js/);
console.log('independent original/linear switch, reset, rhythm link: pass');

$('game-model').value = 'nonlinear'; $('game-model').onchange();
assert.equal($('game-seed-a-wrap').hidden, false);
assert.equal($('game-seed-b-wrap').hidden, true);
assert.equal(app.game[0].seed, 681);
assert.equal(app.game[0].agent.kind, 'rps_nonlinear_readout');
assert.equal(app.game[1].agent.kind, 'bio');
assert.match($('game-model-note').textContent, /KC→MBON weights were frozen/);
$('game-opponent').value = 'human'; $('game-opponent').onchange();
const humanWeights = app.game.map(fly => fly.net.m.slice());
const humanReadout = JSON.stringify(app.game[0].agent.W.map(row => [...row]));
const humanBaseline = app.game[0].agent.baseline;
rockButton.onclick(); until(2); verifyRows();
app.game.forEach((fly, i) => assert.deepEqual(fly.net.m, humanWeights[i]));
assert.equal(JSON.stringify(app.game[0].agent.W.map(row => [...row])), humanReadout);
assert.equal(app.game[0].agent.baseline, humanBaseline);
assert.equal(app.game[0].agent.scores.length, 3);
$('game-opponent').value = 'pattern'; $('game-opponent').onchange();
$('game-model-b').value = 'nonlinear'; $('game-model-b').onchange();
assert.equal(app.game[0].seed, 681);
assert.equal(app.game[1].seed, 682);
assert.equal(app.game[1].agent.kind, 'rps_nonlinear_readout');
assert.throws(() => app.gameFly(684, 'nonlinear'), /Incompatible saved nonlinear student/);
$('game-seed-b').value = '684'; $('game-seed-b').onchange();
assert.equal(app.game, null, 'invalid archive selection must not keep an old controller active');
assert.match($('game-status').textContent, /Selected saved model unavailable/);
$('game-seed-b').value = '682'; $('game-seed-b').onchange();
assert.match($('game-a-model').textContent, /681/);
assert.match($('game-b-model').textContent, /682/);
const patternWeights = app.game.map(fly => fly.net.m.slice());
$('game-next').onclick(); until(2); verifyRows();
app.game.forEach((fly, i) => assert.deepEqual(fly.net.m, patternWeights[i]));
$('game-watch').onclick();
for (let i = 0; i < 2000 && $('game-history').children.length < 4; i++) tick();
assert.ok($('game-history').children.length >= 4);
$('game-watch').onclick();
$('game-seed-b').value = '683'; $('game-seed-b').onchange();
assert.equal(app.game[0].seed, 681);
assert.equal(app.game[1].seed, 683);
assert.equal($('game-history').children.length, 0);
$('game-reset').onclick();
assert.equal($('game-history').children.length, 0);
assert.equal(app.game[0].agent.kind, 'rps_nonlinear_readout');
assert.equal(app.game[1].agent.kind, 'rps_nonlinear_readout');
console.log('nonlinear human/pattern/two-fly play, independent seeds, switching, reset, frozen weights: pass');

$('play').onclick();
for (let i = 0; i < 2000 && !MOVES.includes($('student-move').textContent); i++) tick();
assert.ok(MOVES.includes($('student-move').textContent), 'experimental student did not decide');
assert.notEqual($('signal-values').textContent, 'No signal yet', 'teacher message was not displayed');
$('play').onclick();
$('reset').onclick();
assert.equal($('student-training').textContent, '0');
console.log('existing experimental message demo start/reset: pass');
