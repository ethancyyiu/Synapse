// Records one real full run for the demo's "Play recorded run" button. Writes display data only.
// node tools/record_demo_replay.mjs
import { readFile, writeFile } from 'node:fs/promises';
import { Network, rng, wiring } from '../js/sim.js';
import { RpsContextualBandit, RpsEpisode, T_DECIDE } from '../js/rps.js';
import { inlineCheckpoint, runStudents } from '../js/rps_student_engine.js';
import { challengeOpponent } from '../js/rps_challenge_opponent.js';

const seed = 830, config = { mode: 'preset', offset: 1 };
const base = new URL('../data/', import.meta.url);
const read = async name => JSON.parse(await readFile(new URL(name, base)));
const sub = await read('subgraph.json');
const { featureMean, featureStd } = (await read('rps_contextual_bandit.json')).design;

// Background firing that the page hides by default is left out to keep the file small.
const recorded = sub.neurons.role.map(role => role !== 'vnc' && role !== 'mid');
const chunks = [];
let offset = 0;
function sparse(net, since) {
  const fired = [];
  for (let i = 0; i < recorded.length; i++)
    if (recorded[i] && net.lastSpike[i] > since && net.lastSpike[i] <= net.t) fired.push(i);
  const start = offset;
  chunks.push(Uint16Array.from(fired));
  offset += fired.length;
  return { offset: start, count: fired.length };
}

const events = [];
const net = new Network(sub, wiring(sub, 'real', seed), { seed });
const agent = new RpsContextualBandit(net, { seed, featureMean, featureStd, readoutLearn: true });
let completed = 0, since = -1, wins = 0;
async function episode(index, training) {
  const ep = new RpsEpisode(net, agent, challengeOpponent(config, (training ? 41000 : 51000) + (training ? index : index - 100)),
    { rounds: 24, learn: training });
  while (!ep.done) {
    const phase = ep.t % 400;
    if (phase === 0) {
      const key = seed * 10000000 + index * 1000 + Math.floor(ep.t / 400);
      net.rand = rng(key + 1); agent.rand = rng(key + 501);
    }
    ep.step();
    if (phase === T_DECIDE) {
      if (training) completed++;
      events.push({ type: 'teacher', frozen: !training, progress: training ? completed / 288 : 1,
        story: training ? `Round ${completed}/288 · learning from wins and losses` : 'No learning, no help',
        nets: [{ view: 0, t: net.t, ...sparse(net, since) }] });
      since = net.t;
    }
  }
  if (!training) wins += ep.events.filter(event => event.reward === 1).length;
}
for (let i = 0; i < 12; i++) await episode(i, true);
const teacherState = JSON.parse(JSON.stringify({ W: agent.W.map(row => [...row]), b: [...agent.b], baseline: agent.baseline }));
agent.explore = false;
for (let i = 0; i < 3; i++) await episode(100 + i, false);
events.push({ type: 'teacherDone', wins });
console.log('Fly A frozen wins', wins);

const last = [-1, -1, -1];
for await (const event of runStudents({ sub, seed, config, teacherState, featureMean, featureStd,
  control: { paused: false, slow: false }, runCheckpoint: inlineCheckpoint })) {
  if (event.type === 'frame') {
    const nets = event.nets.map((state, view) => {
      const reset = view > 0 && state.t < last[view];
      const entry = { view, t: state.t, reset, ...sparse(state, reset ? -1 : last[view]) };
      last[view] = state.t;
      return entry;
    });
    events.push({ type: 'frame', nets });
  } else {
    events.push(event);
    if (event.type === 'checkpoint') console.log(event.point.rounds, event.point.wins.join(' '));
  }
}
const spikes = new Uint16Array(offset);
let cursor = 0;
for (const chunk of chunks) { spikes.set(chunk, cursor); cursor += chunk.length; }
await writeFile(new URL('demo_replay_strategy_a_830.bin', base), Buffer.from(spikes.buffer));
await writeFile(new URL('demo_replay_strategy_a_830.json', base), JSON.stringify({
  label: 'Strategy A · Fly A seed 830 · VNC and Mid firing not recorded', events }));
console.log('events', events.length, 'spikes', offset);
