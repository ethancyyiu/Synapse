// Small cross-language trace for tests. Run: node tools/rps_trace.mjs
import { readFileSync } from 'node:fs';
import { Network, wiring } from '../js/sim.js';
import { ActivityChannel, Opponent, RpsBioRL, RpsEpisode } from '../js/rps.js';

const sub = JSON.parse(readFileSync(new URL('../data/subgraph.json', import.meta.url)));
const net = new Network(sub, wiring(sub), { seed: 7 });
const agent = new RpsBioRL(net, { seed: 7 });
const channel = new ActivityChannel(sub, net);
const ep = new RpsEpisode(net, agent, new Opponent({ seed: 91 }),
  { rounds: 6, channel, captureSignals: true }).run();
console.log(JSON.stringify({ events: ep.events.map(e => [e.action, e.opponent, e.reward]),
  signals: ep.signals, mSum: net.m.reduce((a, b) => a + b, 0) }));
