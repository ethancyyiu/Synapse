// Isolated two-label communication test. No changes to connectome, dynamics, or BioRL.
// Run: node tools/calibrate_rps_channel.mjs screen|evaluate
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { RpsBioRL } from '../js/rps.js';

const sub = JSON.parse(readFileSync(new URL('../data/subgraph.json', import.meta.url)));
const ROUND_MS = 400, T_DECIDE = 240, T_LEARN = 340, ROUNDS = 24;
const CALIBRATION_SEEDS = [11, 12];
const EVALUATION_SEEDS = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110];
const CONFIRMATION_SEEDS = [201, 202, 203, 204, 205, 206, 207, 208, 209, 210];
const EPISODES = 12;
const TEST_EPISODES = ['quick', 'quick_late', 'quick_precue_kc', 'quick_precue_kc_nocue', 'quick_precue_kc_struct', 'quick_precue_kc_retain', 'quick_precue_kc_retain_cue'].includes(process.argv[2]) ? 1
  : process.argv[2] === 'screen' ? 2 : 3;
const mean = a => a.reduce((s, x) => s + x, 0) / a.length;

const configurations = [
  { name: 'current_pair', width: 2, hz: 150, start: 180, lanes: [0, 1] },
  { name: 'four_60ms', width: 4, hz: 150, start: 180, lanes: [0, 1] },
  { name: 'eight_60ms', width: 8, hz: 150, start: 180, lanes: [0, 1] },
  { name: 'four_120ms', width: 4, hz: 150, start: 120, lanes: [0, 1] },
  { name: 'eight_120ms', width: 8, hz: 150, start: 120, lanes: [0, 1] },
  { name: 'eight_120ms_300hz', width: 8, hz: 300, start: 120, lanes: [0, 1] },
  { name: 'eight_240ms', width: 8, hz: 150, start: 0, lanes: [0, 1] },
  { name: 'all_lane_240ms', width: 20, hz: 150, start: 0, lanes: [0, 1] },
];
const targetControls = [
  { name: 'lanes_0_2', width: 8, hz: 300, start: 120, lanes: [0, 2] },
  { name: 'lanes_1_2', width: 8, hz: 300, start: 120, lanes: [1, 2] },
  { name: 'rate_coded_lane_0', width: 8, hz: 300, start: 120, lanes: [0, 1], encoding: 'rate' },
];

function fly(seed) {
  const net = new Network(sub, wiring(sub, 'real', seed), { seed });
  return { net, agent: new RpsBioRL(net, { seed }) };
}

function balancedLabels(seed) {
  const rand = rng(seed);
  const labels = Array.from({ length: ROUNDS }, (_, i) => i % 2);
  for (let i = labels.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [labels[i], labels[j]] = [labels[j], labels[i]];
  }
  return labels;
}

function targets(net, config) {
  const groups = config.targetIds ?? config.lanes.map(lane => Array.from(net.laneVpn[lane].slice(0, config.width)));
  if (groups.some(g => g.length !== config.width) || groups[0].some(i => groups[1].includes(i)) ||
      (config.targetIds && groups.some(g => g.some(i => sub.neurons.role[i] !== 'kc'))))
    throw Error('Invalid channel targets');
  return groups;
}

function runEpisode(fly, config, labels, { presentation = 'correct', learn = true } = {}) {
  const { net, agent } = fly;
  const groups = targets(net, config);
  const presented = presentation === 'shuffled' ? balancedLabels(labels.seed + 7919) : labels.values;
  const cueCounts = [labels.seed % 2, (labels.seed >>> 1) % 2];
  const cues = config.precue ? labels.values.map(label => cueCounts[label]++ % 2) : null;
  const cueTargets = config.precue ? [0, 2].map(lane => net.laneVpn[lane].slice(8)) : null;
  const events = [];
  const messageActivity = config.noCue ? Array.from({ length: ROUNDS }, () =>
    ({ kc: 0, mbon: [0, 0, 0] })) : null;
  net.resetState();
  for (let t = 0; t < ROUNDS * ROUND_MS; t++) {
    const round = Math.floor(t / ROUND_MS), phase = t % ROUND_MS;
    if (config.retainMbonCounts && phase === 0) net.resetCounts();
    if (phase === 50) {
      const retained = config.retainMbonCounts ? Array.from(net.idx.mbon, i => net.count[i]) : null;
      net.resetCounts();
      if (retained) net.idx.mbon.forEach((i, k) => { net.count[i] = retained[k]; });
    }
    if (phase === config.start && presentation !== 'disconnected') {
      if (config.encoding === 'rate') net.setDrive(groups[0], presented[round] === 0 ? 75 : config.hz);
      else net.setDrive(groups[presented[round]], config.hz);
    }
    if (config.precue && phase === 40) groups.forEach(g => net.setDrive(g, 0));
    if (config.precue && !config.noCue && phase === 70) net.setDrive(cueTargets[cues[round]], 150);
    if (phase === T_DECIDE) {
      groups.forEach(g => net.setDrive(g, 0));
      if (config.precue && !config.noCue) net.setDrive(cueTargets[cues[round]], 0);
      const targetSpikes = groups.map(g => g.reduce((sum, i) => sum + net.count[i], 0));
      const kcSpikes = net.idx.kc.reduce((sum, i) => sum + net.count[i], 0);
      const activeKc = net.idx.kc.reduce((sum, i) => sum + Number(net.trace[i] > 0.01), 0);
      const action = agent.decide(net);
      const correct = Number(action === (config.precue ? [0, 2][labels.values[round]] : labels.values[round]));
      events.push({ label: labels.values[round], presented: presentation === 'disconnected' ? null : presented[round],
        action, correct, rates: [...agent.rates], targetSpikes, kcSpikes, activeKc,
        ...(config.precue ? { cue: config.noCue ? null : cues[round],
          targetTrace: groups.map(g => g.map(i => net.trace[i])),
          eligibility: agent.elig.reduce((sum, x) => sum + x, 0) } : {}),
        ...(config.noCue ? { messageActivity: messageActivity[round] } : {}) });
      agent.reward(net, correct ? 1 : -1);
      net.resetCounts();
    }
    if (phase === T_LEARN) agent.afterDopamine(net, learn);
    net.step();
    if (config.noCue && phase < 40) for (let i = 0; i < net.nSpikes; i++) {
      const neuron = net.spikes[i], role = sub.neurons.role[neuron];
      if (role === 'kc') messageActivity[round].kc++;
      if (role === 'mbon') messageActivity[round].mbon[net.groupOf[neuron]]++;
    }
  }
  return events;
}

function summarize(events) {
  const byLabel = [0, 1].map(label => {
    const sample = events.filter(e => e.label === label);
    return { label, n: sample.length, accuracy: mean(sample.map(e => e.correct)),
      actions: [0, 1, 2].map(a => sample.filter(e => e.action === a).length),
      mbonRates: [0, 1, 2].map(k => mean(sample.map(e => e.rates[k]))),
      targetSpikes: [0, 1].map(k => mean(sample.map(e => e.targetSpikes[k]))),
      kcSpikes: mean(sample.map(e => e.kcSpikes)), activeKc: mean(sample.map(e => e.activeKc)),
      ...(sample[0].targetTrace ? { targetTrace: [0, 1].map(k =>
        mean(sample.map(e => mean(e.targetTrace[k])))),
        eligibility: mean(sample.map(e => e.eligibility)) } : {}),
      ...(sample[0].messageActivity ? { messageKcSpikes: mean(sample.map(e => e.messageActivity.kc)),
        messageMbonSpikes: [0, 1, 2].map(k => mean(sample.map(e => e.messageActivity.mbon[k]))) } : {}) };
  });
  const actionTv = 0.5 * [0, 1, 2].reduce((sum, a) =>
    sum + Math.abs(byLabel[0].actions[a] / byLabel[0].n - byLabel[1].actions[a] / byLabel[1].n), 0);
  return { accuracy: mean(events.map(e => e.correct)), byLabel, actionTv };
}

function heldout(fly, config, seed, presentation) {
  const all = [];
  for (let test = 0; test < TEST_EPISODES; test++) {
    const labelSeed = seed * 100000 + 500 + test;
    all.push(...runEpisode(fly, config, { seed: labelSeed, values: balancedLabels(labelSeed) },
      { presentation, learn: false }));
  }
  return summarize(all);
}

function runSeed(config, seed) {
  const student = fly(seed + 500);
  const before = heldout(fly(seed + 500), config, seed, 'correct');
  const training = [];
  for (let ep = 0; ep < (config.trainingEpisodes ?? EPISODES); ep++) {
    const labelSeed = seed * 1009 + ep;
    const events = runEpisode(student, config, { seed: labelSeed, values: balancedLabels(labelSeed) });
    training.push(summarize(events).accuracy);
  }
  const weights = student.net.m.slice();
  const evaluations = {};
  for (const presentation of ['correct', 'disconnected', 'shuffled']) {
    const evaluator = fly(seed + 500);
    evaluator.net.m.set(weights);
    evaluations[presentation] = heldout(evaluator, config, seed, presentation);
    if (evaluator.net.m.some((x, i) => x !== weights[i])) throw Error('Evaluation changed synapses');
  }
  const baselineStudent = fly(seed + 500);
  const baselineLabels = seed * 1009;
  for (let ep = 0; ep < (config.precue ? config.trainingEpisodes : 1); ep++) {
    const labelSeed = config.precue ? seed * 1009 + ep : baselineLabels;
    runEpisode(baselineStudent, config, { seed: labelSeed, values: balancedLabels(labelSeed) },
      { presentation: 'disconnected' });
  }
  if (!config.precue && baselineStudent.net.plastic.some(ids => ids.some(i => baselineStudent.net.m[i] !== 1)))
    throw Error('No-signal training unexpectedly changed plastic weights');
  const baselineEvaluator = fly(seed + 500);
  baselineEvaluator.net.m.set(baselineStudent.net.m);
  evaluations.signalIndependent = heldout(baselineEvaluator, config, seed, 'disconnected');
  return { seed, before, training, evaluations,
    changedPlasticWeights: student.net.plastic.flatMap(ids => [...ids]).filter(i => student.net.m[i] !== 1).length };
}

function run(config, seeds) {
  const runs = seeds.map(seed => {
    const result = runSeed(config, seed);
    console.log(`${config.name} seed ${seed}: before ${result.before.accuracy.toFixed(3)}, ` +
      ['correct', 'disconnected', 'shuffled', 'signalIndependent']
        .map(k => `${k} ${result.evaluations[k].accuracy.toFixed(3)}`).join(', '));
    return result;
  });
  const score = mean(runs.map(r => r.evaluations.correct.accuracy -
    Math.max(r.evaluations.disconnected.accuracy, r.evaluations.shuffled.accuracy)));
  return { config, runs, calibrationScore: score };
}

const mode = process.argv[2];
if (!['screen', 'evaluate', 'target_controls', 'confirm', 'quick', 'quick_late', 'quick_precue_kc', 'quick_precue_kc_nocue', 'quick_precue_kc_struct', 'quick_precue_kc_retain', 'quick_precue_kc_retain_cue'].includes(mode))
  throw Error('Use screen, evaluate, target_controls, confirm, quick, quick_late, quick_precue_kc, quick_precue_kc_nocue, quick_precue_kc_struct, quick_precue_kc_retain, or quick_precue_kc_retain_cue');
const screenPath = new URL('../data/rps_channel_calibration.json', import.meta.url);
let output;
if (mode === 'quick' || mode === 'quick_late' || mode === 'quick_precue_kc' || mode === 'quick_precue_kc_nocue' || mode === 'quick_precue_kc_struct' || mode === 'quick_precue_kc_retain' || mode === 'quick_precue_kc_retain_cue') {
  const path = new URL(mode === 'quick_precue_kc_retain_cue'
    ? '../data/rps_channel_quick_precue_kc_retain_cue.json'
    : mode === 'quick_precue_kc_retain'
    ? '../data/rps_channel_quick_precue_kc_retain.json'
    : mode === 'quick_precue_kc_struct'
    ? '../data/rps_channel_quick_precue_kc_struct.json'
    : mode === 'quick_precue_kc_nocue'
    ? '../data/rps_channel_quick_precue_kc_nocue.json'
    : mode === 'quick_precue_kc'
    ? '../data/rps_channel_quick_precue_kc.json'
    : mode === 'quick_late'
    ? '../data/rps_channel_quick_eight_50ms_150hz.json'
    : '../data/rps_channel_quick_eight_60ms_300hz.json', import.meta.url);
  if (existsSync(path)) throw Error(`Refusing to overwrite ${path.pathname}`);
  const config = mode === 'quick_precue_kc_retain_cue'
    ? { name: 'precue_kc_retain_cue', width: 2, hz: 300, start: 0,
      targetIds: [[506, 545], [113, 1421]], precue: true,
      retainMbonCounts: true, trainingEpisodes: 8 }
    : mode === 'quick_precue_kc_retain'
    ? { name: 'precue_kc_retain', width: 2, hz: 300, start: 0,
      targetIds: [[506, 545], [113, 1421]], precue: true, noCue: true,
      retainMbonCounts: true, trainingEpisodes: 8 }
    : mode === 'quick_precue_kc_struct'
    ? { name: 'precue_kc_struct', width: 2, hz: 300, start: 0,
      targetIds: [[506, 545], [113, 1421]], precue: true, noCue: true, trainingEpisodes: 8 }
    : mode === 'quick_precue_kc_nocue'
    ? { name: 'precue_kc_nocue', width: 2, hz: 300, start: 0,
      targetIds: [[1319, 2016], [818, 1462]], precue: true, noCue: true, trainingEpisodes: 8 }
    : mode === 'quick_precue_kc'
    ? { name: 'precue_kc', width: 2, hz: 300, start: 0,
      targetIds: [[1319, 2016], [818, 1462]], precue: true, trainingEpisodes: 8 }
    : mode === 'quick_late'
    ? { name: 'eight_50ms_150hz', width: 8, hz: 150, start: 190, lanes: [0, 1] }
    : { name: 'eight_60ms_300hz', width: 8, hz: 300, start: 180, lanes: [0, 1] };
  const seed = 11;
  const result = runSeed(config, seed);
  output = { mode, config, seed, episodes: config.trainingEpisodes ?? EPISODES, testEpisodes: TEST_EPISODES,
    roundsPerEpisode: ROUNDS, command: `node tools/calibrate_rps_channel.mjs ${mode}`,
    ...(config.precue ? { scheduleMs: { message: [0, 40], gap: [40, 70], cue: config.noCue ? null : [70, 240],
      decision: 240, learning: 340 } } : {}), result };
  console.log(`${config.name} seed ${seed}: before ${result.before.accuracy.toFixed(3)}, ` +
    ['correct', 'disconnected', 'shuffled', 'signalIndependent']
      .map(k => `${k} ${result.evaluations[k].accuracy.toFixed(3)}`).join(', '));
  writeFileSync(path, JSON.stringify(output, null, 2));
} else if (mode === 'screen' || mode === 'target_controls') {
  const path = mode === 'screen' ? screenPath
    : new URL('../data/rps_channel_target_controls.json', import.meta.url);
  try { output = JSON.parse(readFileSync(path)); }
  catch { output = { mode, seeds: CALIBRATION_SEEDS, episodes: EPISODES,
    testEpisodes: TEST_EPISODES, roundsPerEpisode: ROUNDS, configurations: [] }; }
  for (const config of mode === 'screen' ? configurations : targetControls) {
    if (output.configurations.some(c => c.config.name === config.name)) continue;
    output.configurations.push(run(config, CALIBRATION_SEEDS));
    writeFileSync(path, JSON.stringify(output, null, 2));
  }
} else if (mode === 'confirm') {
  const path = new URL('../data/rps_channel_confirmation.json', import.meta.url);
  const config = configurations.find(c => c.name === 'eight_120ms_300hz');
  try { output = JSON.parse(readFileSync(path)); }
  catch { output = { mode, seeds: CONFIRMATION_SEEDS,
    note: 'Fixed 300 Hz setting chosen before any confirmation-seed run',
    episodes: EPISODES, testEpisodes: TEST_EPISODES, roundsPerEpisode: ROUNDS,
    configuration: { config, runs: [] } }; }
  if (output.configuration.config.name !== config.name) throw Error('Confirmation setting changed');
  for (const seed of CONFIRMATION_SEEDS) {
    if (output.configuration.runs.some(r => r.seed === seed)) continue;
    const result = runSeed(config, seed);
    output.configuration.runs.push(result);
    console.log(`confirmation seed ${seed}: before ${result.before.accuracy.toFixed(3)}, ` +
      ['correct', 'disconnected', 'shuffled', 'signalIndependent']
        .map(k => `${k} ${result.evaluations[k].accuracy.toFixed(3)}`).join(', '));
    writeFileSync(path, JSON.stringify(output, null, 2));
  }
} else {
  const screen = JSON.parse(readFileSync(screenPath));
  const best = [...screen.configurations].sort((a, b) => b.calibrationScore - a.calibrationScore)[0];
  const candidate = process.argv[3]
    ? screen.configurations.find(c => c.config.name === process.argv[3]) : best;
  if (!candidate || candidate.calibrationScore !== best.calibrationScore)
    throw Error('Only a configuration tied for the top calibration score may be evaluated');
  const evaluationPath = new URL(candidate === best ? '../data/rps_channel_evaluation.json'
    : `../data/rps_channel_evaluation_${candidate.config.name}.json`, import.meta.url);
  try { output = JSON.parse(readFileSync(evaluationPath)); }
  catch { output = { mode, seeds: EVALUATION_SEEDS, selectedOn: CALIBRATION_SEEDS,
    episodes: EPISODES, testEpisodes: TEST_EPISODES, roundsPerEpisode: ROUNDS,
    selection: { name: candidate.config.name, score: candidate.calibrationScore },
    configuration: { config: candidate.config, runs: [] } }; }
  if (output.selection.name !== candidate.config.name) throw Error('Calibration selection changed');
  for (const seed of EVALUATION_SEEDS) {
    if (output.configuration.runs.some(r => r.seed === seed)) continue;
    const result = runSeed(candidate.config, seed);
    output.configuration.runs.push(result);
    console.log(`${candidate.config.name} evaluation seed ${seed}: before ${result.before.accuracy.toFixed(3)}, ` +
      ['correct', 'disconnected', 'shuffled', 'signalIndependent']
        .map(k => `${k} ${result.evaluations[k].accuracy.toFixed(3)}`).join(', '));
    writeFileSync(evaluationPath, JSON.stringify(output, null, 2));
  }
}
if (mode === 'screen') writeFileSync(screenPath, JSON.stringify(output, null, 2));
