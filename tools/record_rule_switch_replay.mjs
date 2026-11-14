// Records run s=5 of the locked rule-switch cohort for the demo's rule-change replay. Display data only.
// It repeats tools/check_rule_switch_cohort.mjs for s=5 (alone, trained, and random arms) and checks that
// every frozen checkpoint matches data/rule_switch_cohort/seed_5.json.
// node tools/record_rule_switch_replay.mjs
import { readFile, writeFile } from 'node:fs/promises';
import { Network, rng, wiring } from '../js/sim.js';
import { RpsContextualBandit, RpsEpisode, T_DECIDE } from '../js/rps.js';
import { PreferenceStudent, comparisonRound, preferenceRound, studentWeights } from '../js/rps_challenge_students.js';
import { frozenStudentCheckpoint } from '../js/rps_challenge_checkpoints.js';
import { challengeOpponent } from '../js/rps_challenge_opponent.js';

const s = 5, NAME = 'demo_replay_rule_switch_s5';
const base = new URL('../data/', import.meta.url);
const read = async name => JSON.parse(await readFile(new URL(name, base)));
const sub = await read('subgraph.json');
const { featureMean, featureStd } = (await read('rps_contextual_bandit.json')).design;
const saved = (await read(`rule_switch_cohort/seed_${s}.json`)).points;
const A = { mode: 'preset', offset: 1 }, B = { mode: 'preset', offset: 2 };
const teacherSeed = 26001 + 2 * s, studentSeed = 26002 + 2 * s;
const makeNet = seed => new Network(sub, wiring(sub, 'real', seed), { seed });
const student = weights => {
  const net = makeNet(studentSeed), agent = new PreferenceStudent(net, studentSeed, featureMean, featureStd);
  if (weights) {
    const w = JSON.parse(weights);
    w.heads.forEach((h, i) => { h.W.forEach((row, a) => agent.heads[i].W[a].set(row)); agent.heads[i].b.set(h.b); });
    agent.baseline = w.baseline;
  }
  agent.explore = true;
  return { net, agent, game: null };
};

// Spikes of the classes the page shows by default, stored sparsely (as in record_demo_replay.mjs).
const recorded = sub.neurons.role.map(role => role !== 'vnc' && role !== 'mid');
const chunks = []; let offset = 0;
function sparse(net, since) {
  const fired = [];
  for (let i = 0; i < recorded.length; i++)
    if (recorded[i] && net.lastSpike[i] > since && net.lastSpike[i] <= net.t) fired.push(i);
  chunks.push(Uint16Array.from(fired));
  const start = offset; offset += fired.length;
  return { offset: start, count: fired.length };
}
const events = [];

// 1. Fly A learns the new rule (Strategy B), exactly as in the cohort runner.
const teacherNet = makeNet(teacherSeed);
const teacherAgent = new RpsContextualBandit(teacherNet, { seed: teacherSeed, featureMean, featureStd });
let since = -1, completed = 0, teacherWins = 0;
for (let e = 0; e < 15; e++) {
  const training = e < 12;
  if (!training) teacherAgent.explore = false;
  const ep = new RpsEpisode(teacherNet, teacherAgent,
    challengeOpponent(B, training ? 110000 + 100 * s + e : 170000 + 10 * s + (e - 12)), { rounds: 24, learn: training });
  while (!ep.done) {
    const phase = ep.t % 400;
    if (phase === 0) {
      const k = teacherSeed * 1e7 + (training ? e : 100 + e - 12) * 1000 + ep.t / 400;
      teacherNet.rand = rng(k + 1); teacherAgent.rand = rng(k + 501);
    }
    ep.step();
    if (phase === T_DECIDE) {
      if (training) completed++;
      events.push({ type: 'teacher', frozen: !training, progress: training ? completed / 288 : 1,
        story: training ? `Round ${completed}/288 · Fly A learns the NEW rule from wins and losses` : 'Fly A test: no learning, no help',
        nets: [{ view: 0, t: teacherNet.t, ...sparse(teacherNet, since) }] });
      since = teacherNet.t;
    }
  }
  if (!training) teacherWins += ep.events.filter(event => event.reward === 1).length;
}
if (teacherWins < 58) throw Error('Teacher gate failed; this should match the cohort');
events.push({ type: 'teacherDone', wins: teacherWins });

// 2. Fly B learns the OLD rule (Strategy A), 96 rounds, no messages.
const shared = student(), twin = student();
for (let e = 0; e < 4; e++) {
  const games = [0, 1].map(() => challengeOpponent(A, 130000 + 100 * s + e));
  shared.game = games[0]; twin.game = games[1];
  for (let r = 0; r < 24; r++) {
    const round = comparisonRound(null, shared, twin, studentSeed * 1e7 + e * 1000 + r, true);
    while (!round.done) round.step();
  }
}
const start = studentWeights(shared.agent);
const alone = student(start), trained = student(start), random = student(start);
const makeOpponent = seed => challengeOpponent(B, 170000 + 10 * s + (seed - 51000));
const check = other => frozenStudentCheckpoint({ sub, students: [alone, other], studentSeed, featureMean, featureStd, makeOpponent });
let last = null;
async function measure(rounds) {
  const [t, r] = [await check(trained), await check(random)];
  const want = saved.find(p => p.rounds === rounds);
  if (t.wins[0] !== r.wins[0] || t.wins[0] !== want.alone || t.wins[1] !== want.trained || r.wins[1] !== want.random)
    throw Error(`Checkpoint ${rounds} does not match the locked cohort result`);
  last = { t, r };
  events.push({ type: 'checkpoint', point: { rounds, wins: [t.wins[0], t.wins[1], r.wins[1]] } });
  console.log(rounds, t.wins[0], t.wins[1], r.wins[1]);
}
events.push({ type: 'stage', step: 3, progress: 0, pause: 2200, title: 'Every Fly B learned the OLD rule',
  story: '96 rounds on Strategy A, then copied into three identical Fly Bs' });
await measure(0);
events.push({ type: 'stage', step: 3, progress: 0, pause: 2600, title: 'The opponent just changed its rule',
  story: `Old habit: all three Fly Bs now win ${saved[0].alone}/72. Fly A already knows the new rule.` });

// 3. After the switch: 96 rounds with the same preferenceRound as the cohort.
teacherNet.resetState();
const teacher = { net: teacherNet, agent: teacherAgent, key: 0 };
const lastT = [-1, -1, -1];
let done = 0, shownPhase = -1;
for (let e = 0; e < 4; e++) {
  for (const x of [alone, trained, random]) x.game = challengeOpponent(B, 150000 + 100 * s + e);
  const episodeWins = [0, 0, 0];
  for (let r = 0; r < 24; r++) {
    teacher.key = teacherSeed * 1e7 + (200 + e) * 1000 + r;
    const round = preferenceRound(teacher, alone, trained, studentSeed * 1e7 + (200 + e) * 1000 + r, random);
    let time = 0;
    while (!round.done) {
      round.step(); time++;
      const phase = time < 240 ? 0 : time < 280 ? 1 : time < 620 ? 2 : 3;
      if (phase !== shownPhase) { shownPhase = phase; events.push({ type: 'phase', phase }); }
      if (time % 40 === 0 || time === 280) {
        events.push({ type: 'frame', nets: [teacherNet, alone.net, trained.net].map((net, view) => {
          const reset = view > 0 && net.t < lastT[view];
          const entry = { view, t: net.t, reset, ...sparse(net, reset ? -1 : lastT[view]) };
          lastT[view] = net.t;
          return entry;
        }) });
      }
    }
    done++;
    round.rows.forEach((row, i) => { episodeWins[i] += Number(row.reward === 1); });
    events.push({ type: 'round', completed: done, total: 96, episode: e, rows: round.rows, message: round.message });
    if ([24, 48, 96].includes(done)) await measure(done);
  }
  events.push({ type: 'episode', point: { episode: e + 1, wins: episodeWins } });
}
events.push({ type: 'final' });
events.push({ type: 'done', totals: [last.t.wins[0], last.t.wins[1]], byCue: last.t.byCue });

const spikes = new Uint16Array(offset);
let cursor = 0;
for (const chunk of chunks) { spikes.set(chunk, cursor); cursor += chunk.length; }
await writeFile(new URL(`${NAME}.bin`, base), Buffer.from(spikes.buffer));
await writeFile(new URL(`${NAME}.json`, base), JSON.stringify({
  label: 'Rule change A → B · run 5 of 15, a stronger run (study average in Result) · VNC and Mid not recorded', events }));
console.log('Fly A', teacherWins, 'events', events.length, 'spikes', offset);
