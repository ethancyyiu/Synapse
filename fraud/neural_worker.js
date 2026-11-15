// A selected mock purchase supplies only its observable three-way cue. The
// trained fly is frozen and the teacher is absent during this browser probe.
import { Network, wiring } from '../js/sim.js';
import { PreferenceStudent, comparisonRound, studentWeights } from '../js/rps_challenge_students.js';
import { FraudResponseGame, NEW_NEED, ACTIONS } from '../js/fraud_neural_game.js';

let resources;
async function load() {
  if (!resources) resources = Promise.all([
    fetch('../data/subgraph.json').then(r => { if (!r.ok) throw Error('Connectome unavailable'); return r.json(); }),
    fetch('../data/rps_contextual_bandit.json').then(r => { if (!r.ok) throw Error('Readout design unavailable'); return r.json(); }),
    fetch('../data/fraud_neural/seed_401.json').then(r => { if (!r.ok) throw Error('Saved neural run unavailable'); return r.json(); }),
  ]);
  return resources;
}

self.onmessage = async ({ data }) => {
  const { cue, requestId } = data;
  try {
    if (![0, 1, 2].includes(cue)) throw Error('Invalid purchase cue');
    const [sub, design, artifact] = await load();
    if (artifact.status !== 'completed' || !artifact.teacherFrozen) throw Error('Frozen Fly B artifact unavailable');
    const saved = JSON.parse(artifact.trainedStudentWeights);
    const makeFly = () => {
      const net = new Network(sub, wiring(sub, 'real', artifact.studentSeed), { seed: artifact.studentSeed });
      const agent = new PreferenceStudent(net, artifact.studentSeed,
        design.design.featureMean, design.design.featureStd);
      saved.heads.forEach((h, i) => {
        h.W.forEach((row, a) => agent.heads[i].W[a].set(row));
        agent.heads[i].b.set(h.b);
      });
      agent.baseline = saved.baseline;
      agent.explore = false;
      agent.readoutLearn = false;
      return { net, agent, game: new FraudResponseGame(NEW_NEED, 1, [cue]) };
    };
    const alone = makeFly(), flyB = makeFly();
    const before = studentWeights(flyB.agent);
    const round = comparisonRound(null, alone, flyB, artifact.studentSeed * 1e7 + cue, false);
    while (!round.done) round.step();
    if (studentWeights(flyB.agent) !== before) throw Error('Frozen Fly B weights changed');
    const row = round.rows[1];
    if (row.selectedHead !== cue || row.dnCounts.some(Boolean) || row.decodedBit !== null)
      throw Error('Teacher-disconnected probe verification failed');
    self.postMessage({ requestId, action: ACTIONS[row.action], cue, selectedHead: row.selectedHead,
      activeKcs: row.activeKcs, neurons: sub.neurons.role.length, teacherDisconnected: true,
      weightsUnchanged: true });
  } catch (error) {
    self.postMessage({ requestId, error: error.message });
  }
};
