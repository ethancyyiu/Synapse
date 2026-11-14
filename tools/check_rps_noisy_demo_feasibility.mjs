// One prospectively locked, write-once exploratory demo pilot.
// node tools/check_rps_noisy_demo_feasibility.mjs
import { readFile, writeFile, access } from 'node:fs/promises';
import { Network, rng, wiring } from '../js/sim.js';
import { RpsContextualBandit, RpsEpisode } from '../js/rps.js';
import { challengeOpponent } from '../js/rps_challenge_opponent.js';
import { runStudents } from '../js/rps_student_engine.js';

const output = new URL('../data/rps_noisy_demo_feasibility.json', import.meta.url);
try { await access(output); throw Error('Refusing to overwrite noisy demo feasibility result'); }
catch (error) { if (error.code !== 'ENOENT') throw error; }
const read = async name => JSON.parse(await readFile(new URL(`../data/${name}`, import.meta.url)));
const [sub, calibration] = await Promise.all([read('subgraph.json'), read('rps_contextual_bandit.json')]);
const seed = 931, config = { mode: 'preset', offset: 1, noise: 0.3 };
const featureMean = calibration.design.featureMean, featureStd = calibration.design.featureStd;
const result = { protocol: { seed, studentSeed: seed + 10000, config, teacherRounds: 288,
  studentRounds: 24, frozenRounds: 72, teacherMinWins: 45, teacherMinCueFraction: 0.5,
  teacherMinActionCount: 6, studentMinAdvantage: 8 }, teacher: null, student: null,
  gates: {}, conclusion: null };
// Existing opponent's noise draw must remain matched across agents and depend only on its seed.
for (const opponentSeed of [41000, 51000, 51001, 51002]) {
  const pair = [challengeOpponent(config, opponentSeed), challengeOpponent(config, opponentSeed)];
  for (let round = 0; round < 72; round++) {
    if (pair[0].observe().previousOpponent !== pair[1].observe().previousOpponent)
      throw Error('Matched opponent cues diverged');
    if (pair[0].play(round % 3).opponent !== pair[1].play((round + 1) % 3).opponent)
      throw Error('Opponent draws depend on student action');
  }
}
result.gates.matchedOpponent = true;
const net = new Network(sub, wiring(sub, 'real', seed), { seed });
const agent = new RpsContextualBandit(net, { seed, featureMean, featureStd, readoutLearn: true });
const snapshot = () => JSON.stringify({ W: agent.W.map(row => [...row]), b: [...agent.b], baseline: agent.baseline });
function episode(opponentSeed, episodeKey, learn) {
  const ep = new RpsEpisode(net, agent, challengeOpponent(config, opponentSeed), { rounds: 24, learn });
  while (!ep.done) {
    if (ep.t % 400 === 0) {
      const round = Math.floor(ep.t / 400), key = seed * 10000000 + episodeKey * 1000 + round;
      net.rand = rng(key + 1); agent.rand = rng(key + 501);
    }
    ep.step();
  }
  return ep.events.map(e => ({ cue: e.observation.previousOpponent, action: e.action,
    opponent: e.opponent, reward: e.reward }));
}
const training = [];
for (let episodeIndex = 0; episodeIndex < 12; episodeIndex++)
  training.push(...episode(41000 + episodeIndex, episodeIndex, true));
const trained = snapshot();
agent.explore = false;
const frozen = [];
for (let i = 0; i < 3; i++) frozen.push(...episode(51000 + i, 100 + i, false));
if (snapshot() !== trained || !net.m.every(value => value === 1))
  throw Error('Teacher frozen check changed weights');
const byCue = [0, 1, 2].map(cue => {
  const rows = frozen.filter(row => row.cue === cue);
  return { cue, n: rows.length, wins: rows.filter(row => row.reward === 1).length };
});
const actions = [0, 1, 2].map(action => frozen.filter(row => row.cue !== null && row.action === action).length);
const wins = frozen.filter(row => row.reward === 1).length;
result.teacher = { wins, byCue, actions, trainingWins: Array.from({ length: 12 }, (_, episodeIndex) =>
  training.slice(episodeIndex * 24, (episodeIndex + 1) * 24).filter(row => row.reward === 1).length) };
result.gates.teacher = wins >= 45 && byCue.every(row => row.n > 0 && row.wins / row.n >= 0.5) &&
  actions.every(n => n >= 6);
if (result.gates.teacher) {
  const messages = [];
  let final = null, rounds = 0;
  for await (const event of runStudents({ sub, seed, config, teacherState: JSON.parse(trained),
    featureMean, featureStd, control: { paused: false, slow: false }, preview: true,
    runCheckpoint: () => { throw Error('Preview requested a checkpoint'); } })) {
    if (event.type === 'round') { rounds++; messages.push(event.message); }
    if (event.type === 'done') final = { wins: event.totals, byCue: event.byCue };
  }
  result.student = { rounds, messages: messages.map(({ preference, decoded, dnCounts }) =>
    ({ preference, decoded, dnCounts })), final };
  result.gates.receiver = rounds === 24 && messages.filter(m => m.preference !== null).length === 23 &&
    messages.every(m => m.preference === m.decoded &&
      (m.preference === null || m.dnCounts.some(count => count > 0)));
  result.gates.display = result.gates.receiver && final.wins[1] - final.wins[0] >= 8;
}
result.conclusion = !result.gates.teacher ? 'stop: teacher prerequisite failed' :
  !result.gates.receiver ? 'stop: receiver prerequisite failed' :
  result.gates.display ? 'opt-in exploratory demo candidate; no teacher-specific claim' :
  'stop: no clear 24-round live teaching contrast';
console.log(JSON.stringify({ teacher: result.teacher, studentWins: result.student?.final?.wins,
  gates: result.gates, conclusion: result.conclusion }));
await writeFile(output, JSON.stringify(result, null, 2), { flag: 'wx' });
