// Isolated, teacher-disconnected frozen checks for the opt-in live Challenge.
import { Network, wiring } from './sim.js';
import { ChallengeCueHeadBandit, comparisonRound, studentWeights } from './rps_challenge_students.js';

export const CHECKPOINT_ROUNDS = Object.freeze([0, 6, 12, 24, 48, 96, 288]);

// Frozen trials have no learning and reset neural state before the next cue.
// Once both actions and rewards are recorded at 280 ms, the remaining steps
// cannot affect this trial or the next one.
export function finishFrozenDecision(round) {
  while (round.rows.some(row => row.reward === null)) round.step();
  round.rows.forEach(row => { row.updateStep = 0; });
}

// `engine` chooses the simulator the frozen copies run on; it defaults to js/sim.js.
export async function frozenStudentCheckpoint({ sub, students, studentSeed, featureMean,
  featureStd, makeOpponent, yieldEpisode = async () => {}, engine = { Network, wiring } }) {
  const original = students.map(s => studentWeights(s.agent));
  const copies = students.map((s, i) => {
    const net = new engine.Network(sub, engine.wiring(sub, 'real', studentSeed), { seed: studentSeed });
    const agent = new ChallengeCueHeadBandit(net, studentSeed, featureMean, featureStd);
    const saved = JSON.parse(original[i]);
    saved.heads.forEach((head, h) => {
      head.W.forEach((row, a) => agent.heads[h].W[a].set(row));
      agent.heads[h].b.set(head.b);
    });
    agent.baseline = saved.baseline;
    agent.explore = false;
    agent.readoutLearn = false;
    return { net, agent, game: null };
  });
  const copiedWeights = copies.map(s => studentWeights(s.agent));
  const wins = [0, 0], byCue = [0, 1].map(() => [0, 1, 2].map(() => ({ wins: 0, n: 0 })));
  const actions = [[], []], schedules = [[], []];
  const custom = makeOpponent(51000).strategy === 'sequence';
  const games = custom ? [makeOpponent(51000), makeOpponent(51000)] : null;
  for (let episode = 0; episode < 3; episode++) {
    const pair = games ?? [makeOpponent(51000 + episode), makeOpponent(51000 + episode)];
    copies.forEach((s, i) => { s.game = pair[i]; });
    for (let r = 0; r < 24; r++) {
      const key = studentSeed * 10000000 + (100 + episode) * 1000 + r;
      const round = comparisonRound(null, copies[0], copies[1], key, false);
      finishFrozenDecision(round);
      round.rows.forEach((row, i) => {
        if (row.decodedBit !== null || row.dnCounts.some(Boolean) || row.updateStep !== 0)
          throw Error('Frozen checkpoint received a message or learned');
        wins[i] += Number(row.reward === 1);
        actions[i].push(row.action);
        schedules[i].push([row.cue, row.opponent]);
        if (row.cue !== null) {
          byCue[i][row.cue].n++;
          byCue[i][row.cue].wins += Number(row.reward === 1);
        }
      });
      await yieldEpisode();
    }
  }
  if (JSON.stringify(schedules[0]) !== JSON.stringify(schedules[1]) ||
      copies.some((s, i) => studentWeights(s.agent) !== copiedWeights[i] ||
        !s.net.m.every(value => value === 1)) ||
      students.some((s, i) => studentWeights(s.agent) !== original[i]))
    throw Error('Frozen checkpoint changed weights or used unmatched opponent streams');
  return { wins, byCue, actions, schedule: schedules[0], disconnected: true, weightsUnchanged: true };
}
