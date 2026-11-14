// Locked, opt-in no-message feasibility pilot for an observable-context RPS task.
// node tools/check_rps_context_cued_feasibility.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { RpsContextualBandit, result } from '../js/rps.js';

const output = new URL('../data/rps_context_cued_feasibility.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite context-cued feasibility result');
const read = path => JSON.parse(readFileSync(new URL(path, import.meta.url)));
const sub = read('../data/subgraph.json');
const { featureMean, featureStd } = read('../data/rps_contextual_bandit.json').design;
const seeds = [951, 952, 953];
const episodes = 12;
const observedRounds = 18;
const roundCount = observedRounds + 1;
const checkpoints = [0, 1, 3, 12];
const evaluationOpponentSeeds = (seed, index) => 52000 + seed * 100 + index;
const trainingOpponentSeed = (seed, episode) => 41000 + seed * 100 + episode;
const same = (left, right) => JSON.stringify(left) === JSON.stringify(right);
const close = (left, right) => Math.abs(left - right) < 1e-10;
const dot = (left, right) => left.reduce((sum, value, i) => sum + value * right[i], 0);
const countActions = actions => [0, 1, 2].map(action => actions.filter(x => x === action).length);
const correctAction = (context, cue) => context === 0 ? (cue + 2) % 3 : cue;
const contextForRound = round => round === 0 ? 0 : Math.floor((round - 1) / 3) % 2;
const makeHead = () => ({ W: [0, 1, 2].map(() => new Float64Array(3)), b: new Float64Array(3) });
const headCopy = head => ({ W: head.W.map(row => [...row]), b: [...head.b] });
const qValues = (head, features) => head.W.map((row, action) =>
  head.b[action] + dot(row, features));

const wires = new Map();
const netFor = seed => {
  if (!wires.has(seed)) wires.set(seed, wiring(sub, 'real', seed));
  return new Network(sub, wires.get(seed), { seed });
};

class ContextOpponent {
  constructor(seed) {
    this.random = rng(seed);
    this.initial = Math.floor(this.random() * 3);
    this.previousOpponent = null;
  }
  play(player, context) {
    const offset = context === 0 ? 1 : 2;
    const opponent = this.previousOpponent === null
      ? this.initial
      : (this.previousOpponent + offset) % 3;
    const reward = result(player, opponent);
    this.previousOpponent = opponent;
    return { opponent, reward };
  }
}

// The first two VPNs in lanes 0 and 1 carry context. The original move-cue
// pathway, laneVpn[cue].slice(2), is left unchanged.
class ContextCueBandit extends RpsContextualBandit {
  constructor(net, seed) {
    super(net, { seed, featureMean, featureStd, readoutLearn: true });
    this.heads = Array.from({ length: 8 }, makeHead);
    this.activeHead = null;
    this.route = null;
    this.kind = 'rps_context_cue_heads_opt_in';
  }
  decide(net) {
    const counts = ids => ids.reduce((sum, id) => sum + net.count[id], 0);
    const contextCounts = [0, 1].map(context => counts(net.laneVpn[context].slice(0, 2)));
    const maxContext = Math.max(...contextCounts);
    if (maxContext === 0 || contextCounts.filter(value => value === maxContext).length !== 1)
      throw Error('Ambiguous or absent VPN context token');
    const context = contextCounts.indexOf(maxContext);
    const cueCounts = net.laneVpn.map(lane => counts(lane.slice(2)));
    const maxCue = Math.max(...cueCounts);
    if (maxCue > 0 && cueCounts.filter(value => value === maxCue).length !== 1)
      throw Error('Ambiguous VPN move cue');
    const cue = maxCue === 0 ? null : cueCounts.indexOf(maxCue);
    this.activeHead = cue === null ? 6 + context : context * 3 + cue;
    this.route = { context, cue, contextCounts, cueCounts };
    const head = this.heads[this.activeHead];
    this.W = head.W;
    this.b = head.b;
    return super.decide(net);
  }
}

const snapshot = agent => ({ heads: agent.heads.map(headCopy), baseline: agent.baseline });
function load(agent, params) {
  for (let head = 0; head < agent.heads.length; head++) {
    for (let action = 0; action < 3; action++)
      agent.heads[head].W[action].set(params.heads[head].W[action]);
    agent.heads[head].b.set(params.heads[head].b);
  }
  agent.baseline = params.baseline;
}

function summarize(rows) {
  const observed = rows.filter(row => row.cue !== null);
  const byCell = [];
  for (let context = 0; context < 2; context++) {
    for (let cue = 0; cue < 3; cue++) {
      const cell = observed.filter(row => row.context === context && row.cue === cue);
      byCell.push({
        context,
        cue,
        n: cell.length,
        optimal: cell.filter(row => row.action === correctAction(context, cue)).length,
        actions: countActions(cell.map(row => row.action)),
      });
    }
  }
  return {
    rounds: rows.length,
    observed: observed.length,
    wins: rows.filter(row => row.reward === 1).length,
    byCell,
    optimal: observed.filter(row => row.action === correctAction(row.context, row.cue)).length,
    actions: countActions(observed.map(row => row.action)),
  };
}

function runEpisode(seed, params, opponentSeed, episodeKey, rounds, learn) {
  const net = netFor(seed);
  const agent = new ContextCueBandit(net, seed);
  if (params) load(agent, params);
  agent.explore = learn;
  const before = snapshot(agent);
  const opponent = new ContextOpponent(opponentSeed);
  const rows = [];

  for (let round = 0; round < rounds; round++) {
    if (round > 0) net.resetState();
    const context = contextForRound(round);
    const cue = opponent.previousOpponent;
    const contextInput = Array.from(net.laneVpn[context].slice(0, 2));
    const moveInput = cue === null ? [] : Array.from(net.laneVpn[cue].slice(2));
    const key = seed * 10000000 + episodeKey * 1000 + round;
    net.rand = rng(key + 2);
    agent.rand = rng(key + 501);
    net.setDrive(contextInput, 150);
    if (moveInput.length) net.setDrive(moveInput, 150);

    let row;
    for (let time = 0; time < 680; time++) {
      if (time === 50) net.resetCounts();
      if (time === 240) {
        net.setDrive(contextInput, 0);
        if (moveInput.length) net.setDrive(moveInput, 0);
        agent.decide(net);
        row = {
          round,
          context,
          cue,
          route: { ...agent.route, contextCounts: [...agent.route.contextCounts],
            cueCounts: [...agent.route.cueCounts] },
          head: agent.activeHead,
          candidate: agent.action,
          rates: [...agent.rates],
          features: [...agent.f],
          scores: [...agent.scores],
          action: null,
          opponent: null,
          reward: null,
          update: null,
        };
      }
      if (time === 280) {
        const played = opponent.play(row.candidate, context);
        agent.reward(net, played.reward);
        row.action = row.candidate;
        row.opponent = played.opponent;
        row.reward = played.reward;
        net.resetCounts();
      }
      if (time === 620) {
        const beforeUpdate = snapshot(agent);
        agent.afterDopamine(net, learn);
        const afterUpdate = snapshot(agent);
        const selected = row.action;
        const step = learn ? agent.alpha * (row.reward - row.scores[selected]) : 0;
        const active = row.head;
        if (learn && (
          beforeUpdate.heads.some((head, index) => index !== active &&
            !same(head, afterUpdate.heads[index])) ||
          !close(afterUpdate.heads[active].b[selected], beforeUpdate.heads[active].b[selected] + step) ||
          row.features.some((value, index) => !close(
            afterUpdate.heads[active].W[selected][index],
            beforeUpdate.heads[active].W[selected][index] + step * value)) ||
          [0, 1, 2].some(action => action !== selected &&
            (!same(afterUpdate.heads[active].W[action], beforeUpdate.heads[active].W[action]) ||
              afterUpdate.heads[active].b[action] !== beforeUpdate.heads[active].b[action]))
        )) throw Error('Context-head update deviated from the original selected-action rule');
        row.update = { selectedHead: active, selectedAction: selected, step };
      }
      net.step();
    }

    const expectedHead = cue === null ? 6 + context : context * 3 + cue;
    if (row.route.context !== context || row.route.cue !== cue || row.head !== expectedHead ||
        row.features.length !== 3 || row.features.some((value, index) => !close(value,
          (Math.log1p(row.rates[index]) - featureMean[index]) / featureStd[index])) ||
        row.reward !== result(row.action, row.opponent) || net.m.some(value => value !== 1))
      throw Error('Sensory routing, feature, reward, or KC-to-MBON integrity failed');
    row.expectedAction = cue === null ? null : correctAction(context, cue);
    rows.push(row);
  }

  const after = snapshot(agent);
  if (!learn && !same(before, after)) throw Error('Frozen controller weights changed');
  return { before, after, rows, summary: summarize(rows), synapsesUnchanged: net.m.every(x => x === 1) };
}

function frozenEvaluation(seed, params, checkpoint) {
  const evaluations = [0, 1, 2].map(index => runEpisode(seed, params,
    evaluationOpponentSeeds(seed, index), 100 + checkpoint * 10 + index,
    1 + 4 * observedRounds, false));
  const rows = evaluations.flatMap(evaluation => evaluation.rows);
  const summary = summarize(rows);
  if (summary.observed !== 3 * 4 * observedRounds ||
      summary.byCell.some(cell => cell.n !== 36) ||
      evaluations.some(evaluation => !evaluation.synapsesUnchanged ||
        !same(evaluation.before, evaluation.after)))
    throw Error('Frozen context evaluation coverage or integrity failed');
  return { rows, summary, weightsUnchanged: true };
}

function expectedTrainingCoverage(episodeRows) {
  const observed = episodeRows.filter(row => row.cue !== null);
  return observed.length === observedRounds &&
    observed.every(row => row.route.context === row.context && row.route.cue === row.cue) &&
    Array.from({ length: 2 }, (_, context) => Array.from({ length: 3 }, (_, cue) =>
      observed.filter(row => row.context === context && row.cue === cue).length).flat()).flat()
      .every(count => count === 3);
}

const data = {
  protocol: {
    command: 'node tools/check_rps_context_cued_feasibility.mjs',
    hypothesis: 'Public context plus previous-opponent cue supports six learnable mappings with early no-message headroom.',
    task: 'context 0 uses opponent offset +1; context 1 uses offset +2; context is not a rule label.',
    inputs: 'context 0: first two VPNs in lane 0; context 1: first two VPNs in lane 1; move cue: existing laneVpn[cue].slice(2); both at 150 Hz, 0-240 ms.',
    schedule: 'one cue-free warmup, then 18 observed rounds in six fixed three-round context blocks alternating 0/1; each context-move cell appears three times per episode.',
    learner: 'eight opt-in three-action heads (six context-move plus two context-specific cue-free); original MBON features, alpha .03, epsilon .20, selected-action own-reward update; no teacher/message.',
    students: seeds,
    training: `${episodes} episodes x ${observedRounds} observed rounds, checkpoints 0/1/3/12 episodes`,
    evaluation: 'three fixed 73-round sequences per checkpoint; one warmup plus 72 scored rounds, 12 per context-move cell per sequence; greedy, disconnected, no learning.',
    streams: 'network rng(key+2), action rng(key+501), key=studentSeed*10000000+episodeKey*1000+round; separate deterministic opponent seeds.',
    gates: {
      route: 'exact context and move routing on every input; no ties; cue-free only on warmup',
      trainingCoverage: 'exactly 3 trials per context-move cell per episode',
      heldOutCoverage: '36 trials per context-move cell across three evaluation sequences',
      earlyHeadroom: 'each seed mean expected-optimal accuracy <=75% after 1 episode',
      eventualLearnability: 'each seed >=9/12 expected-optimal choices in every cell after 12 episodes',
      integrity: 'unchanged KC-to-MBON multipliers; frozen Q weights; exact selected-head update only',
    },
    seedAudit: '951-953 absent from saved JSON seed fields, research docs, and experiment scripts; 901-903 were excluded as previously used.',
  },
  stage: 0,
  status: 'not_started',
  passed: false,
  students: [],
  gates: {},
};

function finish(status, gates) {
  data.status = status;
  data.passed = status === 'passed';
  data.gates = gates;
  writeFileSync(output, `${JSON.stringify(data, null, 2)}\n`, { flag: 'wx' });
  console.log(JSON.stringify({ status, gates,
    students: data.students.map(student => ({ seed: student.seed,
      checkpoints: Object.fromEntries(Object.entries(student.checkpoints).map(([episode, result]) =>
        [episode, { accuracy: result.summary.optimal / result.summary.observed,
          byCell: result.summary.byCell.map(cell => `${cell.optimal}/${cell.n}`) }])) })) }, null, 2));
}

function main() {
  const laneLengths = [0, 1, 2].map(lane =>
    sub.neurons.role.reduce((sum, role, index) =>
      sum + Number(role === 'vpn' && sub.neurons.tag[index] === lane), 0));
  if (laneLengths.some(length => length < 3)) throw Error('Insufficient VPN lane capacity');

  for (const seed of seeds) {
    let params = null;
    const student = { seed, training: [], checkpoints: {} };
    const checkpointAt = new Set(checkpoints);
    if (checkpointAt.has(0)) student.checkpoints[0] = frozenEvaluation(seed,
      snapshot(new ContextCueBandit(netFor(seed), seed)), 0);

    for (let episode = 0; episode < episodes; episode++) {
      const trained = runEpisode(seed, params, trainingOpponentSeed(seed, episode), episode,
        roundCount, true);
      if (!expectedTrainingCoverage(trained.rows))
        throw Error(`Training coverage failed for seed ${seed}, episode ${episode}`);
      student.training.push({ episode: episode + 1, rows: trained.rows,
        summary: trained.summary, synapsesUnchanged: trained.synapsesUnchanged });
      params = trained.after;
      const completed = episode + 1;
      if (checkpointAt.has(completed))
        student.checkpoints[completed] = frozenEvaluation(seed, params, completed);
    }
    data.students.push(student);
  }

  const gates = {
    laneCapacity: laneLengths[0] >= 3 && laneLengths[1] >= 3 && laneLengths[2] >= 3,
    allTrainingCoverage: data.students.every(student => student.training.length === episodes &&
      student.training.every(episode => expectedTrainingCoverage(episode.rows))),
    heldOutCoverage: data.students.every(student => checkpoints.every(checkpoint =>
      student.checkpoints[checkpoint].summary.byCell.every(cell => cell.n === 36))),
    earlyHeadroom: data.students.every(student => {
      const result = student.checkpoints[1].summary;
      return result.optimal / result.observed <= 0.75;
    }),
    eventualLearnability: data.students.every(student =>
      student.checkpoints[12].summary.byCell.every(cell => cell.optimal >= 27)),
    updateAndSynapseIntegrity: data.students.every(student =>
      student.training.every(episode => episode.synapsesUnchanged) &&
      checkpoints.every(checkpoint => student.checkpoints[checkpoint].weightsUnchanged)),
  };
  finish(Object.values(gates).every(Boolean) ? 'passed' : 'stopped_infeasible', gates);
}

try {
  main();
} catch (error) {
  data.status = 'invalid';
  data.error = String(error?.stack ?? error);
  if (!existsSync(output)) writeFileSync(output, `${JSON.stringify(data, null, 2)}\n`, { flag: 'wx' });
  throw error;
}