// Replay the saved seed-1 teacher and retain raw source spikes before channel clipping.
// Run: node tools/audit_rps_channel_sources.mjs
import { readFileSync, writeFileSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { ActivityChannel, Opponent, RpsBioRL, RpsEpisode } from '../js/rps.js';

const sub = JSON.parse(readFileSync(new URL('../data/subgraph.json', import.meta.url)));
const published = JSON.parse(readFileSync(new URL('../data/rps_results.json', import.meta.url)));
const net = new Network(sub, wiring(sub, 'real', 1), { seed: 1 });
const agent = new RpsBioRL(net, { seed: 1 });
const channel = new ActivityChannel(sub, net);
const weights = readFileSync(new URL('../data/rps_teacher.bin', import.meta.url));
const edges = net.plastic.flatMap(ids => [...ids]);
if (weights.length !== edges.length * 8) throw Error('Teacher weights have incompatible length');
edges.forEach((edge, i) => { net.m[edge] = weights.readDoubleLE(i * 8); });
net.rand = rng(1);
agent.rand = rng(38);

const rounds = [];
for (let episode = 0; episode < 3; episode++) {
  const ep = new RpsEpisode(net, agent,
    new Opponent({ strategy: 'previous_opponent', offset: 1, noise: 0.1, seed: 1009 + episode }),
    { rounds: 24, learn: false, channel, captureSignals: true });
  while (!ep.done) {
    if (ep.t % 400 === 180) rounds.push({ episode, round: Math.floor(ep.t / 400),
      previousOpponent: ep.obs.previousOpponent,
      raw: channel.source.map(i => net.count[i]) });
    ep.step();
  }
  for (let i = 0; i < 24; i++) {
    const row = rounds[episode * 24 + i];
    row.quantized = ep.signals[i];
    row.nextOpponent = ep.events[i].opponent;
  }
}
const expected = published.conditions.trained.runs[0].transmitted.slice(0, rounds.length);
const matches = rounds.filter((row, i) => row.quantized.every((v, j) => v === expected[i][j])).length;
if (matches !== rounds.length) throw Error('Raw-source replay differs from published messages');
writeFileSync(new URL('../data/rps_channel_source_audit.json', import.meta.url),
  JSON.stringify({ seed: 1, episodes: 3, source: channel.source,
    sampledMs: 180, countsResetMs: 50, matchesPublished: matches, rounds }, null, 2));
console.log(`Saved ${rounds.length} raw source vectors; all ${matches} quantized messages match the published run.`);
