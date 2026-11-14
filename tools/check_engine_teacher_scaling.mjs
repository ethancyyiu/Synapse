// Teacher learning on js/engine.js with the js/sim.js scaling constants and with constants re-measured on
// js/engine.js; see docs/engine-teacher-scaling-protocol.md. Write-once per seed.
// node tools/check_engine_teacher_scaling.mjs SEED_INDEX
import { readFile, open, access, mkdir } from 'node:fs/promises';
import { Network, rng, wiring } from '../js/engine.js';
import { RpsContextualBandit, RpsEpisode } from '../js/rps.js';
import { challengeOpponent } from '../js/rps_challenge_opponent.js';

const s = Number(process.argv[2]);
if (!Number.isInteger(s) || s < 0 || s > 14) throw Error('Use seed index 0-14');
const dir = new URL('../data/engine_teacher_scaling/', import.meta.url);
const output = new URL(`seed_${s}.json`, dir);
try { await access(output); throw Error('Write-once result already exists'); }
catch (error) { if (error.code !== 'ENOENT') throw error; }
const read = async name => JSON.parse(await readFile(new URL('../data/' + name, import.meta.url)));
const sub = await read('subgraph.json');
const simConstants = (({ featureMean, featureStd }) => ({ featureMean, featureStd }))((await read('rps_contextual_bandit.json')).design);
const B = { mode: 'preset', offset: 2 };
const makeNet = seed => new Network(sub, wiring(sub), { seed });
const play = (net, agent, opponent, learn, key) => {
  const ep = new RpsEpisode(net, agent, opponent, { rounds: 24, learn });
  while (!ep.done) {
    if (ep.t % 400 === 0) { const k = key + ep.t / 400; net.rand = rng(k + 1); agent.rand = rng(k + 501); }
    ep.step();
  }
  return ep;
};

// Re-measure the two constants on this engine: no reward, outcome, or label is used.
const CAL = 32999;
const calNet = makeNet(CAL);
const calAgent = new RpsContextualBandit(calNet, { seed: CAL, ...simConstants, readoutLearn: false });
const logs = [];
for (let e = 0; e < 6; e++) {
  const ep = play(calNet, calAgent, challengeOpponent(B, 890000 + e), false, CAL * 1e7 + e * 1000);
  for (const event of ep.events) if (event.observation.previousOpponent !== null) logs.push(event.rates.map(rate => Math.log1p(rate)));
}
const featureMean = [0, 1, 2].map(j => logs.reduce((sum, row) => sum + row[j], 0) / logs.length);
const featureStd = [0, 1, 2].map(j => Math.sqrt(logs.reduce((sum, row) => sum + (row[j] - featureMean[j]) ** 2, 0) / logs.length));
if (featureStd.some(v => !(v > 0))) throw Error('A re-measured scale is zero');
const engineConstants = { featureMean, featureStd };

const teacherSeed = 32001 + 2 * s;
const teacherScore = constants => {
  const net = makeNet(teacherSeed);
  const agent = new RpsContextualBandit(net, { seed: teacherSeed, ...constants });
  for (let e = 0; e < 12; e++) play(net, agent, challengeOpponent(B, 910000 + 100 * s + e), true, teacherSeed * 1e7 + e * 1000);
  agent.explore = false;
  let wins = 0;
  for (let e = 0; e < 3; e++)
    wins += play(net, agent, challengeOpponent(B, 970000 + 10 * s + e), false, teacherSeed * 1e7 + (100 + e) * 1000)
      .events.filter(event => event.reward === 1).length;
  if (!net.m.every(v => v === 1)) throw Error('Synapses changed');
  return wins;
};
const result = { engine: 'js/engine.js', teacherSeed, calibrationRounds: logs.length, simConstants, engineConstants,
  wins: { simConstants: teacherScore(simConstants), engineConstants: teacherScore(engineConstants) } };
await mkdir(dir, { recursive: true });
const file = await open(output, 'wx');
try { await file.writeFile(JSON.stringify(result, null, 1) + '\n'); } finally { await file.close(); }
console.log(JSON.stringify({ s, ...result.wins }));
