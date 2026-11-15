// Study 1, part E diagnostic: does Fly A's measured neural activity carry its learned advice?
// Protocol: docs/connectome-contribution-2-protocol.md. Write-once per seed.
// For one network seed, a trained Fly A (rule B, existing procedure) and an untrained Fly A (same network,
// no training) see the same 144 rule-B rounds with identical noise. We record every neuron's spike count
// at each decision and the trained Fly A's chosen action (its advice). No student, no teaching.
// node tools/check_endogenous_signal.mjs SEED_INDEX   (0-4)
import { readFile, open, access, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { Network, rng, wiring } from '../js/sim.js';
import { RpsContextualBandit, RpsEpisode } from '../js/rps.js';
import { challengeOpponent } from '../js/rps_challenge_opponent.js';

const s = Number(process.argv[2]);
if (!Number.isInteger(s) || s < 0 || s > 4) throw Error('Use seed index 0-4');
// v2: v1 saved empty individual-MBON counts (groupMbon holds typed arrays, which .flat() does not unpack).
// v1 files are kept unchanged in data/endogenous_signal/; this fixed recorder writes to a new folder.
const dir = new URL('../data/endogenous_signal_v2/', import.meta.url);
const output = new URL(`seed_${s}.json`, dir);
try { await access(output); throw Error('Write-once result already exists'); }
catch (error) { if (error.code !== 'ENOENT') throw error; }
const read = async name => JSON.parse(await readFile(new URL('../data/' + name, import.meta.url)));
const sub = await read('subgraph.json');
const { featureMean, featureStd } = (await read('rps_contextual_bandit.json')).design;
const B = { mode: 'preset', offset: 2 };
const teacherSeed = 35001 + 2 * s;
const makeNet = () => new Network(sub, wiring(sub, 'real', teacherSeed), { seed: teacherSeed });
const run = (net, agent, opponentSeed, keyBase, learn, capture) => {
  const ep = new RpsEpisode(net, agent, challengeOpponent(B, opponentSeed), { rounds: 24, learn });
  const rows = [];
  while (!ep.done) {
    if (ep.t % 400 === 0) { const k = teacherSeed * 1e7 + keyBase + ep.t / 400; net.rand = rng(k + 1); agent.rand = rng(k + 501); }
    const before = ep.events.length;
    ep.step();
    if (capture && ep.events.length > before) {
      const ev = ep.events.at(-1);
      const counts = Array.from(net.count);
      rows.push({ cue: ev.observation.previousOpponent, action: ev.action, reward: ev.reward,
        groupRates: ev.rates, mbon: net.groupMbon.flatMap(g => [...g]).map(i => counts[i]),
        countsHash: createHash('sha256').update(Int32Array.from(counts).buffer ? Buffer.from(Int32Array.from(counts).buffer) : JSON.stringify(counts)).digest('hex') });
    }
  }
  return { ep, rows };
};

// Trained Fly A: the existing teacher procedure (12 rule-B episodes, then gate).
const tNet = makeNet(), trained = new RpsContextualBandit(tNet, { seed: teacherSeed, featureMean, featureStd });
for (let e = 0; e < 12; e++) run(tNet, trained, 1210000 + 100 * s + e, e * 1000, true, false);
trained.explore = false;
let gate = 0;
for (let e = 0; e < 3; e++) gate += run(tNet, trained, 1220000 + 10 * s + e, (100 + e) * 1000, false, false).ep.events.filter(x => x.reward === 1).length;
// Untrained Fly A: same network seed, never trained.
const uNet = makeNet(), untrained = new RpsContextualBandit(uNet, { seed: teacherSeed, featureMean, featureStd });
untrained.explore = false;
// Same 144 evaluation rounds, same noise keys, for both.
const rows = [];
for (let e = 0; e < 6; e++) {
  tNet.resetState(); uNet.resetState();
  const t = run(tNet, trained, 1230000 + 100 * s + e, (300 + e) * 1000, false, true).rows;
  const u = run(uNet, untrained, 1230000 + 100 * s + e, (300 + e) * 1000, false, true).rows;
  t.forEach((row, i) => rows.push({ episode: e, cue: row.cue, advice: row.action, adviceWins: row.reward === 1,
    untrainedAction: u[i].action, trainedMbon: row.mbon, untrainedMbon: u[i].mbon,
    trainedGroupRates: row.groupRates, untrainedGroupRates: u[i].groupRates,
    identicalAllNeurons: row.countsHash === u[i].countsHash }));
}
await mkdir(dir, { recursive: true });
const file = await open(output, 'wx');
try { await file.writeFile(JSON.stringify({ seed: s, teacherSeed, teacherGateWins: gate, gatePassed: gate >= 58, rows }, null, 1) + '\n'); }
finally { await file.close(); }
console.log(JSON.stringify({ s, gate, rounds: rows.length, identical: rows.filter(r => r.identicalAllNeurons).length,
  adviceWinRate: rows.filter(r => r.adviceWins).length / rows.length }));
