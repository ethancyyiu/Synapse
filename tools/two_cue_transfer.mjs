// Two-cue reversal diagnostic using the existing connectome and RpsBioRL.
// node tools/two_cue_transfer.mjs probe|baseline|fresh1|diag0|diag1|explore1|source SEED [01|02] [lanes01|lanes02] [150|300]
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { RpsBioRL } from '../js/rps.js';

const sub = JSON.parse(readFileSync(new URL('../data/subgraph.json', import.meta.url)));
const [mode, seedText, pair = '01', cuePattern = 'lanes01', hzText = '150', teacherSeedText] = process.argv.slice(2);
const seed = Number(seedText), cueHz = Number(hzText), teacherSeed = Number(teacherSeedText);
const voltageProbe = mode === 'single_cue_order_voltage';
const refractoryProbe = mode === 'single_cue_order_refractory';
const learningResetProbe = mode === 'single_cue_learning_reset';
const disconnectedControlProbe = mode === 'single_cue_disconnected_control';
const lockedLearningProbe = ['single_cue_learning_locked', 'single_cue_learning_locked_v2'].includes(mode);
const prospectiveLearningProbe = mode === 'single_cue_learning_prospective';
const freshLearningProbe = mode === 'single_cue_learning_fresh';
const aloneBaseline = mode === 'two_cue_alone_baseline';
const blockSpeedPilot = mode === 'two_cue_block_speed_pilot';
const matchedLearningProbe = lockedLearningProbe || prospectiveLearningProbe || freshLearningProbe;
const gradedPopulationProbe = ['graded_population_decode', 'graded_population_pathway', 'graded_population_choice', 'graded_population_drive', 'graded_population_ties', 'graded_population_silence503', 'graded_population_initial_silence'].includes(mode);
if (!(aloneBaseline || blockSpeedPilot || learningResetProbe || disconnectedControlProbe || matchedLearningProbe || gradedPopulationProbe || ['probe', 'baseline', 'fresh1', 'diag0', 'diag1', 'diag_readout', 'explore1', 'source', 'teach', 'teach_live', 'teach_live_pretrain_long', 'teach_live_threshold', 'decode_mapping_signal', 'decode_mapping_signal_confirm', 'decode_mapping_signal_paired', 'decode_mapping_signal_replication', 'decode_mapping_signal_teacher', 'decode_mapping_signal_mechanism', 'decode_mapping_signal_counterfactual', 'single_cue_transfer', 'single_cue_baseline_probe', 'single_cue_initial_drive', 'single_cue_round_reset', 'single_cue_preceding_bit', 'single_cue_order_probe', 'single_cue_order_reset', 'single_cue_order_rng', 'single_cue_order_rng_full', 'single_cue_order_rng_reset', 'single_cue_order_conductance', 'single_cue_order_voltage', 'single_cue_order_refractory', 'interference', 'interference_late', 'interference_kc', 'interference_temporal', 'interference_retain', 'encoding_channel_probe', 'encoding_channel_natural', 'encoding_channel_active0_cue', 'decode_two_bit', 'decode_two_bit_active0_conditional', 'decode_two_bit_active0_probe', 'decode_two_bit_cue_gated_pair', 'decode_two_bit_cue_gated_probe', 'receiver_frozen', 'decode_bit2146', 'decode_bit2134', 'decode_bit_pathway', 'decode_bit2146_explore', 'decode_bit2146_active0'].includes(mode)) ||
    !Number.isInteger(seed) || seed < 1 || (aloneBaseline && ![901, 902, 903].includes(seed)) || (blockSpeedPilot && seed !== 904) || ((mode === 'single_cue_order_conductance' || voltageProbe || refractoryProbe || learningResetProbe || disconnectedControlProbe) && seed !== 202) || !['01', '02'].includes(pair) ||
    !['lanes01', 'lanes02'].includes(cuePattern) || ![150, 300].includes(cueHz) ||
    (['decode_mapping_signal_teacher', 'decode_mapping_signal_mechanism', 'decode_mapping_signal_counterfactual'].includes(mode) && ![501, 502, 503].includes(teacherSeed)) ||
    (matchedLearningProbe && (!Number.isInteger(teacherSeed) || teacherSeed < 1)) ||
    (freshLearningProbe && (teacherSeed < 603 || teacherSeed > 624 ||
      (teacherSeed - 603) % 3 !== 0 || seed !== 801 + (teacherSeed - 603) / 3)) ||
    ((aloneBaseline || blockSpeedPilot || learningResetProbe || disconnectedControlProbe || matchedLearningProbe || gradedPopulationProbe || ['source', 'teach', 'teach_live', 'teach_live_pretrain_long', 'teach_live_threshold', 'decode_mapping_signal', 'decode_mapping_signal_confirm', 'decode_mapping_signal_paired', 'decode_mapping_signal_replication', 'decode_mapping_signal_teacher', 'decode_mapping_signal_mechanism', 'decode_mapping_signal_counterfactual', 'single_cue_transfer', 'single_cue_baseline_probe', 'single_cue_initial_drive', 'single_cue_round_reset', 'single_cue_preceding_bit', 'single_cue_order_probe', 'single_cue_order_reset', 'single_cue_order_rng', 'single_cue_order_rng_full', 'single_cue_order_rng_reset', 'single_cue_order_conductance', 'single_cue_order_voltage', 'single_cue_order_refractory', 'diag_readout', 'interference', 'interference_late', 'interference_kc', 'interference_temporal', 'interference_retain', 'encoding_channel_probe', 'encoding_channel_natural', 'encoding_channel_active0_cue', 'decode_two_bit', 'decode_two_bit_active0_conditional', 'decode_two_bit_cue_gated_pair', 'decode_two_bit_cue_gated_probe', 'receiver_frozen', 'decode_bit2146', 'decode_bit2134', 'decode_bit_pathway', 'decode_bit2146_explore', 'decode_bit2146_active0'].includes(mode)) &&
      (pair !== '02' || cuePattern !== 'lanes02' || cueHz !== 150)))
  throw Error('usage: node tools/two_cue_transfer.mjs MODE SEED [01|02] [lanes01|lanes02] [150|300]');
const actions = pair.split('').map(Number);
const suffix = `${cuePattern === 'lanes01' ? '' : '_' + cuePattern}${cueHz === 150 ? '' : '_' + cueHz + 'hz'}`;
const output = new URL(`../data/two_cue_${mode}_${pair}_${seed}${suffix}${blockSpeedPilot ? '_corrected' : ''}${['decode_mapping_signal_teacher', 'decode_mapping_signal_mechanism', 'decode_mapping_signal_counterfactual'].includes(mode) ? `_teacher${teacherSeed}` : ''}.json`, import.meta.url);
if (existsSync(output)) throw Error(`Refusing to overwrite ${output.pathname}`);
const ROUNDS = 24, EPISODES = 10, WIDTH = 8;

function fly(flySeed) {
  const net = new Network(sub, wiring(sub, 'real', flySeed), { seed: flySeed });
  return { net, agent: new RpsBioRL(net, { seed: flySeed }) };
}
function cues(cueSeed) {
  const values = Array.from({ length: ROUNDS }, (_, i) => i % 2), random = rng(cueSeed);
  for (let i = values.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [values[i], values[j]] = [values[j], values[i]];
  }
  return values;
}
function episode(source, mapping, cueValues, learn = true, diagnose = false, explore = false,
  sample = null, feedback = true, messageBits = null, stimulation = {}) {
  const { net, agent } = source, events = [];
  if (messageBits && messageBits.length !== ROUNDS) throw Error('Invalid message count');
  if (stimulation.mappingByRound && (stimulation.mappingByRound.length !== ROUNDS ||
      stimulation.mappingByRound.some(value => value !== 0 && value !== 1)))
    throw Error('Invalid mapping schedule');
  if (stimulation.trialRandomSeeds && stimulation.trialRandomSeeds.length !== ROUNDS)
    throw Error('Invalid trial random seeds');
  if (stimulation.trialAgentRandomSeeds && stimulation.trialAgentRandomSeeds.length !== ROUNDS)
    throw Error('Invalid trial agent random seeds');
  const gradedMessages = stimulation.gradedMessages;
  if (gradedMessages && (gradedMessages.length !== ROUNDS ||
      gradedMessages.some(row => row.length !== 4 || row.some(rate => !Number.isFinite(rate) || rate < 0 || rate > 300))))
    throw Error('Invalid graded message rates');
  const { messageHz = 300, messageStart = 120, messageEnd = 240,
    cueStart = 0, cueEnd = 240, cueVisible = true, observeStep = null,
    retainMbonCounts = false, messageSource = null, messageCodes = null } = stimulation;
  if (messageCodes && (messageCodes.length !== ROUNDS ||
      messageCodes.some(code => !Number.isInteger(code) || code < 0 || code > 3)))
    throw Error('Invalid two-bit message codes');
  if ((messageBits || messageCodes) && (messageStart < 0 || messageEnd > 240 || messageStart >= messageEnd))
    throw Error('Invalid message timing');
  const targets = (cuePattern === 'lanes02' ? [0, 2] : [0, 1])
    .map(lane => net.laneVpn[lane].slice(WIDTH));
  const channelTargets = stimulation.messageTargets ??
    [0, 1].map(lane => net.laneVpn[lane].slice(0, WIDTH));
  const zeroBitTarget = stimulation.zeroBitTarget;
  if (stimulation.messageTargets && channelTargets.some(group =>
    group.some(i => sub.neurons.role[i] !== 'kc'))) throw Error('KC target mismatch');
  if (gradedMessages && channelTargets.length !== 4) throw Error('Four KC groups required');
  if (zeroBitTarget?.some(i => sub.neurons.role[i] !== 'kc')) throw Error('Zero-bit KC target mismatch');
  const forcedCounts = [0, 0];
  const liveBits = [];
  net.resetState();
  for (let t = 0; t < ROUNDS * 400; t++) {
    const round = Math.floor(t / 400), phase = t % 400, cue = cueValues[round];
    if (phase === 0 && stimulation.trialRandomSeeds)
      net.rand = rng(stimulation.trialRandomSeeds[round]);
    if (phase === 0 && stimulation.trialAgentRandomSeeds)
      agent.rand = rng(stimulation.trialAgentRandomSeeds[round]);
    if (phase === 0 && stimulation.observeRoundStart) stimulation.observeRoundStart(net, round);
    if ((stimulation.resetEachRound || stimulation.resetEachPair && round % 2 === 0) && phase === 0) net.resetState();
    if (retainMbonCounts && phase === 0) net.resetCounts();
    if (phase === cueStart && cueVisible) net.setDrive(targets[cue], cueHz);
    if (phase === 50) {
      const retained = retainMbonCounts ? Array.from(net.idx.mbon, i => net.count[i]) : null;
      net.resetCounts();
      if (retained) net.idx.mbon.forEach((i, k) => { net.count[i] = retained[k]; });
    }
    if (phase === 120 && sample) sample(net, round, cue);
    if (phase === cueEnd && cueVisible) net.setDrive(targets[cue], 0);
    if (phase === messageStart && (messageBits || messageSource || messageCodes)) {
      const bit = messageCodes ? messageCodes[round] :
        messageSource ? messageSource(round, cue) : messageBits[round];
      if (messageCodes) {
        channelTargets.forEach((group, i) => net.setDrive(group,
          (!stimulation.cueGatedBits || cue === i) && (bit & (i === 0 ? 2 : 1)) ? messageHz : 0));
        if (zeroBitTarget) net.setDrive(zeroBitTarget,
          (!stimulation.cueGatedBits || cue === 0) && !(bit & 2) ? messageHz : 0);
      } else {
        if (bit !== 0 && bit !== 1) throw Error('Invalid teacher bit');
        net.setDrive(channelTargets[bit], messageHz);
      }
      liveBits[round] = bit;
    }
    if (phase === messageStart && gradedMessages)
      channelTargets.forEach((group, i) => net.setDrive(group, gradedMessages[round][i]));
    if (phase === messageEnd && (messageBits || messageSource || messageCodes))
      [...channelTargets, ...(zeroBitTarget ? [zeroBitTarget] : [])].forEach(group => net.setDrive(group, 0));
    if (phase === messageEnd && gradedMessages)
      channelTargets.forEach(group => net.setDrive(group, 0));
    if (phase === 240) {
      const channelSpikes = diagnose && (messageBits || messageSource || messageCodes)
        ? channelTargets.map(group => group.reduce((sum, i) => sum + net.count[i], 0)) : null;
      let action = agent.decide(net);
      if (explore && learn) {
        action = actions[forcedCounts[cue]++ % 2];
        agent.action = action;
        agent.V = Math.tanh(agent.rates[action] / agent.o.cRef);
        agent.elig = Float64Array.from(agent.plasticPre[action], i => net.trace[i]);
      }
      const trialMapping = stimulation.mappingByRound?.[round] ?? mapping;
      const correctAction = actions[cue ^ trialMapping];
      const correct = Number(action === correctAction);
      const event = { round, cue, mapping: trialMapping, correctAction, action, correct, rates: [...agent.rates] };
      if (diagnose) {
        event.cueSpikes = targets[cue].reduce((sum, i) => sum + net.count[i], 0);
        event.kcSpikes = net.idx.kc.reduce((sum, i) => sum + net.count[i], 0);
        event.kcActive = Array.from(net.idx.kc).filter(i => net.count[i] > 0);
        event.eligibilitySum = agent.elig.reduce((sum, x) => sum + x, 0);
        if (['interference_late', 'interference_kc', 'interference_temporal', 'interference_retain', 'teach_live', 'teach_live_threshold', 'decode_mapping_signal_mechanism', 'decode_bit_pathway', 'decode_bit2146_explore', 'decode_bit2146_active0', 'decode_two_bit_active0_probe', 'decode_two_bit_cue_gated_probe'].includes(mode))
          event.eligibilityByAction = actions.map(action => agent.plasticPre[action]
            .reduce((sum, i) => sum + net.trace[i], 0));
        if (mode === 'interference_temporal' && messageBits)
          event.messageTargetTrace = channelTargets.map(group => group.map(i => net.trace[i]));
        if (messageBits || messageSource || messageCodes) { event.bit = liveBits[round]; event.channelSpikes = channelSpikes; }
      }
      events.push(event);
      if (feedback) agent.reward(net, correct ? 1 : -1);
      net.resetCounts();
    }
    if (phase === 340) {
      const event = events.at(-1), edges = diagnose && learn ? net.plastic[event.action] : null;
      const before = edges ? Float64Array.from(edges, i => net.m[i]) : null;
      if (diagnose && mode === 'interference_temporal')
        event.eligibilityAtReward = agent.elig.reduce((sum, x) => sum + x, 0);
      agent.afterDopamine(net, learn);
      if (diagnose) {
        event.dopamine = agent.D;
        event.weightChange = before ? edges.reduce((sum, i, k) => sum + Math.abs(net.m[i] - before[k]), 0) : 0;
        if (mode === 'decode_mapping_signal_mechanism' || mode === 'decode_bit_pathway' || mode === 'decode_bit2146_explore' || mode === 'decode_bit2146_active0' || mode === 'decode_two_bit_active0_probe')
          event.weightDelta = before ? edges.reduce((sum, i, k) => sum + net.m[i] - before[k], 0) : 0;
      }
    }
    net.step();
    if (observeStep) observeStep(net, round, phase);
  }
  return events;
}
function run(flySeed) {
  const source = fly(flySeed), training = [];
  for (let ep = 0; ep < (['fresh1', 'diag0', 'diag1', 'explore1'].includes(mode) ? EPISODES : EPISODES * 2); ep++) {
    const mapping = mode === 'fresh1' || mode === 'diag1' || mode === 'explore1'
      ? 1 : mode === 'diag0' ? 0 : Number(ep >= EPISODES);
    const cueSeed = seed * 1009 + ep;
    const events = episode(source, mapping, cues(cueSeed), true,
      ['diag0', 'diag1', 'explore1'].includes(mode), mode === 'explore1');
    training.push({ episode: ep, mapping, accuracy: events.reduce((s, e) => s + e.correct, 0) / ROUNDS,
      actions: [0, 1, 2].map(action => events.filter(e => e.action === action).length), events });
    console.log(`seed ${seed} fly ${flySeed} ep ${ep} mapping ${mapping} accuracy ${training.at(-1).accuracy.toFixed(3)}`);
  }
  const frozen = [];
  for (let mapping = 0; mapping < 2; mapping++) {
    const evaluator = fly(flySeed + 10000);
    evaluator.net.m.set(source.net.m);
    for (let test = 0; test < 3; test++) {
      const events = episode(evaluator, mapping, cues(seed * 100000 + mapping * 1000 + test), false);
      if (evaluator.net.m.some((weight, i) => weight !== source.net.m[i]))
        throw Error('Frozen evaluation changed synaptic weights');
      frozen.push({ mapping, test, accuracy: events.reduce((s, e) => s + e.correct, 0) / ROUNDS,
        actions: [0, 1, 2].map(action => events.filter(e => e.action === action).length) });
    }
  }
  return { training, frozen };
}
function sourceRun() {
  const ids = Array.from(fly(seed).net.idx.mbon), runs = [];
  for (const mapping of [0, 1]) for (const trained of [false, true]) {
    const teacher = fly(seed), training = [], samples = [];
    if (trained) for (let ep = 0; ep < EPISODES; ep++) {
      const events = episode(teacher, mapping, cues(seed * 1009 + ep));
      training.push(events.reduce((sum, e) => sum + e.correct, 0) / ROUNDS);
    }
    const weights = teacher.net.m.slice(), frozenAccuracy = [];
    for (let test = 0; test < 3; test++) {
      const values = cues(seed * 100000 + 800 + test);
      const events = episode(teacher, mapping, values, false, false, false, (net, round, cue) => {
        samples.push({ test, round, cue, trace: ids.map(id => net.trace[id]) });
      }, false);
      frozenAccuracy.push(events.reduce((sum, row) => sum + row.correct, 0) / ROUNDS);
    }
    if (teacher.net.m.some((weight, i) => weight !== weights[i]))
      throw Error('Teacher evaluation changed frozen weights');
    runs.push({ mapping, trained, training, frozenAccuracy, samples });
    console.log(`source seed ${seed} mapping ${mapping} trained ${trained}: ` +
      `late accuracy ${training.length ? training.slice(-3).reduce((a, b) => a + b) / 3 : 'n/a'}`);
  }
  const a = runs[0].samples, b = runs[2].samples;
  if (a.some((row, i) => JSON.stringify(row) !== JSON.stringify(b[i])))
    throw Error('Untrained sources differ across mappings without feedback');
  return { sampleMs: 120, sourceIds: ids, runs };
}
function teachRun() {
  const lock = JSON.parse(readFileSync(new URL('../data/two_cue_teacher_source_lock.json', import.meta.url)));
  if (lock.neuron !== 2132 || lock.feature !== 'trace>=0.5' || lock.sampleMs !== 120 ||
      lock.channel.strengthHz !== 300 || lock.channel.endMs !== 240)
    throw Error('Two-cue source/channel lock changed');
  const teacher = fly(seed), untrained = fly(seed);
  const teacherTraining = [];
  for (let ep = 0; ep < EPISODES; ep++) {
    const events = episode(teacher, 1, cues(seed * 2009 + 1000 + ep));
    teacherTraining.push(events.reduce((sum, e) => sum + e.correct, 0) / ROUNDS);
  }
  const postCues = Array.from({ length: 3 }, (_, ep) => cues(seed * 100000 + 200 + ep));
  function teacherStream(source) {
    const weights = source.net.m.slice(), rows = [], accuracy = [];
    for (let ep = 0; ep < postCues.length; ep++) {
      const samples = [];
      const events = episode(source, 1, postCues[ep], false, false, false,
        (net, round, cue) => samples.push({ episode: ep, round, cue,
          trace: net.trace[lock.neuron], bit: Number(net.trace[lock.neuron] < .5) }), false);
      samples.forEach((row, i) => { row.teacherAction = events[i].action; });
      rows.push(...samples);
      accuracy.push(events.reduce((sum, e) => sum + e.correct, 0) / ROUNDS);
    }
    if (source.net.m.some((weight, i) => weight !== weights[i]))
      throw Error('Frozen teacher weights changed');
    return { rows, accuracy };
  }
  const trained = teacherStream(teacher), naive = teacherStream(untrained);
  const shuffled = trained.rows.map(row => row.bit), random = rng(seed * 7919 + 271);
  for (const cue of [0, 1]) {
    const positions = trained.rows.flatMap((row, i) => row.cue === cue ? [i] : []);
    const values = positions.map(i => shuffled[i]);
    for (let i = values.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [values[i], values[j]] = [values[j], values[i]];
    }
    positions.forEach((position, i) => { shuffled[position] = values[i]; });
  }
  const student = fly(seed + 500), pretraining = [];
  for (let ep = 0; ep < EPISODES; ep++) {
    const events = episode(student, 0, cues(seed * 1009 + ep));
    pretraining.push(events.reduce((sum, e) => sum + e.correct, 0) / ROUNDS);
  }
  const startWeights = student.net.m.slice(), conditions = {};
  for (const kind of ['alone', 'trained', 'untrained', 'shuffled']) {
    const pupil = fly(seed + 500);
    pupil.net.m.set(startWeights);
    if (pupil.net.m.some((weight, i) => weight !== startWeights[i])) throw Error('Student weights differ');
    const bits = kind === 'trained' ? trained.rows.map(row => row.bit)
      : kind === 'untrained' ? naive.rows.map(row => row.bit)
        : kind === 'shuffled' ? shuffled : null;
    const events = [];
    for (let ep = 0; ep < postCues.length; ep++) {
      const messages = bits?.slice(ep * ROUNDS, (ep + 1) * ROUNDS) ?? null;
      events.push(...episode(pupil, 1, postCues[ep], true, true, false, null, true, messages)
        .map(row => ({ ...row, episode: ep })));
    }
    const evaluator = fly(seed + 1500);
    evaluator.net.m.set(pupil.net.m);
    const weights = evaluator.net.m.slice(), frozen = [];
    for (let test = 0; test < 3; test++) {
      const values = cues(seed * 100000 + 500 + test);
      const heldout = episode(evaluator, 1, values, false);
      frozen.push(heldout.reduce((sum, e) => sum + e.correct, 0) / ROUNDS);
    }
    if (evaluator.net.m.some((weight, i) => weight !== weights[i]))
      throw Error('Frozen student weights changed');
    conditions[kind] = { events, frozen };
    console.log(`teach seed ${seed} ${kind}: first24 ${events.slice(0, 24)
      .reduce((sum, e) => sum + e.correct, 0) / 24}, frozen ${frozen.reduce((a, b) => a + b) / 3}`);
  }
  function probe(bit) {
    const pupil = fly(seed + 500);
    pupil.net.m.set(startWeights);
    return episode(pupil, 1, cues(seed * 100000 + 900), false, true, false, null, false,
      Array(ROUNDS).fill(bit));
  }
  const probe0 = probe(0), probe1 = probe(1);
  return { lock, teacherTraining, teacherStreams: { trained, untrained: naive },
    shuffledBits: shuffled, studentPretraining: pretraining, conditions,
    pairedProbe: { bit0: probe0, bit1: probe1 } };
}
function teachLiveRun() {
  const sourceId = 2132;
  const sourceThreshold = mode === 'teach_live_threshold' ? 1.915 : 0.5;
  const lock = JSON.parse(readFileSync(new URL('../data/two_cue_teacher_source_lock.json', import.meta.url)));
  if (lock.neuron !== sourceId || lock.sampleMs !== 120 ||
      (mode !== 'teach_live_threshold' && lock.feature !== 'trace>=0.5'))
    throw Error('Teacher source lock changed');
  const setting = { messageHz: 300, messageStart: 0, messageEnd: 40,
    cueStart: 70, cueEnd: 240, retainMbonCounts: true,
    messageTargets: [[506, 545], [113, 1421]] };
  const trainer = fly(seed), teacherTraining = [];
  for (let ep = 0; ep < EPISODES; ep++) {
    const events = episode(trainer, 1, cues(seed * 2009 + 1000 + ep));
    teacherTraining.push(events.reduce((sum, e) => sum + e.correct, 0) / ROUNDS);
  }
  const teacherWeights = trainer.net.m.slice();
  const trained = fly(seed), untrained = fly(seed);
  trained.net.m.set(teacherWeights);
  const postCues = Array.from({ length: 2 }, (_, ep) => cues(seed * 100000 + 200 + ep));
  function teacherRound(source, cue, round) {
    const { net, agent } = source, target = net.laneVpn[cue === 0 ? 0 : 2].slice(WIDTH);
    if (round === 0) net.resetState();
    let trace;
    for (let phase = 0; phase < 400; phase++) {
      if (phase === 0) net.setDrive(target, cueHz);
      if (phase === 50) net.resetCounts();
      if (phase === 120) trace = net.trace[sourceId];
      if (phase === 240) { net.setDrive(target, 0); agent.decide(net); net.resetCounts(); }
      if (phase === 340) agent.afterDopamine(net, false);
      net.step();
    }
    return { cue, trace, bit: Number(trace < sourceThreshold) };
  }
  const student = fly(seed + 500), studentPretraining = [];
  const pretrainCueStart = ['teach_live_pretrain_long', 'teach_live_threshold'].includes(mode) ? 0 : 70;
  for (let ep = 0; ep < EPISODES; ep++) {
    const events = episode(student, 0, cues(seed * 1009 + ep), true, false,
      false, null, true, null, { ...setting, cueStart: pretrainCueStart });
    studentPretraining.push(events.reduce((sum, e) => sum + e.correct, 0) / ROUNDS);
  }
  if (['teach_live_pretrain_long', 'teach_live_threshold'].includes(mode) &&
      studentPretraining.slice(-3).reduce((sum, x) => sum + x, 0) / 3 < 0.75)
    return { setting, sourceId, sourceThreshold, teacherSampleMs: 120, teacherTraining,
      studentPretraining, pretrainCueStart, status: 'stopped: pretraining below 75%' };
  const startWeights = student.net.m.slice(), streams = {}, conditions = {};
  for (const kind of ['alone', 'trained', 'untrained', 'shuffled']) {
    const pupil = fly(seed + 500);
    pupil.net.m.set(startWeights);
    const source = kind === 'trained' ? trained : kind === 'untrained' ? untrained : null;
    const rows = [], events = [];
    let shuffled = null;
    if (kind === 'shuffled') {
      shuffled = streams.trained.map(row => row.bit);
      const random = rng(seed * 7919 + 271);
      for (const cue of [0, 1]) {
        const positions = streams.trained.flatMap((row, i) => row.cue === cue ? [i] : []);
        const values = positions.map(i => shuffled[i]);
        for (let i = values.length - 1; i > 0; i--) {
          const j = Math.floor(random() * (i + 1));
          [values[i], values[j]] = [values[j], values[i]];
        }
        positions.forEach((position, i) => { shuffled[position] = values[i]; });
      }
    }
    for (let ep = 0; ep < postCues.length; ep++) {
      const messageSource = source ? (round, cue) => {
        const row = teacherRound(source, cue, round);
        rows.push({ episode: ep, round, ...row });
        return row.bit;
      } : null;
      const messages = shuffled?.slice(ep * ROUNDS, (ep + 1) * ROUNDS) ?? null;
      events.push(...episode(pupil, 1, postCues[ep], true, true, false, null, true,
        messages, { ...setting, messageSource }).map(row => ({ ...row, episode: ep })));
    }
    if (kind === 'trained' || kind === 'untrained') {
      if (source.net.m.some((weight, i) => weight !== (kind === 'trained' ? teacherWeights[i] : 1)))
        throw Error('Teacher weights changed during live communication');
      if (rows.some((row, i) => row.cue !== postCues[row.episode][row.round] ||
        row.bit !== Number(row.trace < sourceThreshold) || events[i].bit !== row.bit))
        throw Error('Teacher bit does not match current-cue neural activity');
      streams[kind] = rows;
    }
    if (kind === 'shuffled') streams.shuffled = shuffled;
    const evaluator = fly(seed + 1500);
    evaluator.net.m.set(pupil.net.m);
    const frozen = episode(evaluator, 1, cues(seed * 100000 + 500), false, true,
      false, null, false, null, setting);
    conditions[kind] = { events, frozen };
    const score = n => events.slice(0, n).reduce((sum, row) => sum + row.correct, 0);
    console.log(`live seed ${seed} ${kind}: first12 ${score(12)}/12, first24 ${score(24)}/24, ` +
      `first48 ${score(48)}/48, frozen ${frozen.reduce((sum, row) => sum + row.correct, 0)}/24`);
  }
  return { setting, sourceId, sourceThreshold, teacherSampleMs: 120, teacherTraining, pretrainCueStart,
    studentPretraining, streams, conditions,
    leakageCheck: { sourceInputs: 'current cue only; no evaluation reward or mapping',
      bitEqualsThresholdOfRecordedTrace: true, futureCuesProcessedBeforeMessage: false,
      conditionalCorrectActionInformationBits: 0 } };
}
function readoutDiagnostic() {
  const results = {};
  for (const retained of [false, true]) {
    const pupil = fly(seed + 500), training = [];
    const setting = { cueStart: 70, cueEnd: 240, retainMbonCounts: retained };
    for (let ep = 0; ep < EPISODES; ep++) {
      const rows = episode(pupil, 0, cues(seed * 1009 + ep), true, false,
        false, null, true, null, setting);
      training.push(rows.reduce((sum, row) => sum + row.correct, 0));
    }
    const frozen = fly(seed + 1500);
    frozen.net.m.set(pupil.net.m);
    const rows = episode(frozen, 0, cues(seed * 100000 + 500), false, false,
      false, null, false, null, setting);
    results[retained ? 'retained' : 'original'] = { training,
      frozenCorrect: rows.reduce((sum, row) => sum + row.correct, 0),
      frozenByCue: [0, 1].map(cue => {
        const sample = rows.filter(row => row.cue === cue);
        return { cue, correct: sample.reduce((sum, row) => sum + row.correct, 0),
          actions: [0, 1, 2].map(action => sample.filter(row => row.action === action).length) };
      }) };
  }
  return results;
}
function decodeMappingSignal() {
  const mechanism = mode === 'decode_mapping_signal_mechanism';
  const sourceSeed = ['decode_mapping_signal_teacher', 'decode_mapping_signal_mechanism', 'decode_mapping_signal_counterfactual'].includes(mode) ? teacherSeed : 12;
  const threshold = 1.915, episodes = 8;
  const replication = ['decode_mapping_signal_replication', 'decode_mapping_signal_teacher', 'decode_mapping_signal_mechanism', 'decode_mapping_signal_counterfactual'].includes(mode);
  const source = JSON.parse(readFileSync(new URL(`../data/two_cue_source_02_${sourceSeed}_lanes02.json`, import.meta.url))).source;
  const index = source.sourceIds.indexOf(2132);
  if (index < 0) throw Error('Teacher source missing');
  const trained = [0, 1].map(mapping => source.runs.find(row => row.mapping === mapping && row.trained));
  const naive = source.runs.find(row => row.mapping === 0 && !row.trained);
  const teacherLearning = trained.map(row => ({ mapping: row.mapping,
    lateTrainingAccuracy: row.training.slice(-3).reduce((sum, value) => sum + value, 0) / 3,
    frozenAccuracy: row.frozenAccuracy,
    frozenMean: row.frozenAccuracy?.reduce((sum, value) => sum + value, 0) / 3 }));
  const random = rng(seed * 1009 + 2132);
  const shuffle = values => {
    for (let i = values.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [values[i], values[j]] = [values[j], values[i]];
    }
    return values;
  };
  function plan(test) {
    const rows = trained.map(run => run.samples.filter(row => row.test === test));
    const untrained = naive.samples.filter(row => row.test === test);
    const cueValues = rows[0].map(row => row.cue), mappings = Array(ROUNDS).fill(-1);
    for (const cue of [0, 1]) {
      const positions = cueValues.flatMap((value, i) => value === cue ? [i] : []);
      if (positions.length !== ROUNDS / 2) throw Error('Cues are not balanced');
      const labels = shuffle([...Array(positions.length / 2).fill(0),
        ...Array(positions.length / 2).fill(1)]);
      positions.forEach((position, i) => { mappings[position] = labels[i]; });
    }
    const correctBits = cueValues.map((cue, i) => {
      if (rows[1][i].cue !== cue || untrained[i].cue !== cue) throw Error('Teacher cues are misaligned');
      return Number(rows[mappings[i]][i].trace[index] < threshold);
    });
    const untrainedBits = untrained.map(row => Number(row.trace[index] < threshold));
    const shuffledBits = Array(ROUNDS).fill(null);
    for (const cue of [0, 1]) {
      const positions = cueValues.flatMap((value, i) => value === cue ? [i] : []);
      const ones = positions.reduce((sum, i) => sum + correctBits[i], 0);
      const first = Math.floor(ones / 2) + (ones % 2 && random() < .5 ? 1 : 0);
      for (const mapping of [0, 1]) {
        const selected = positions.filter(i => mappings[i] === mapping);
        const count = mapping === 0 ? first : ones - first;
        const values = shuffle([...Array(count).fill(1), ...Array(selected.length - count).fill(0)]);
        selected.forEach((position, i) => { shuffledBits[position] = values[i]; });
      }
    }
    return { test, cueValues, mappings, correctBits, untrainedBits, shuffledBits };
  }
  const trainingPlans = Array.from({ length: episodes }, (_, ep) => plan(ep % 2));
  const heldout = plan(2);
  const entropy = p => p === 0 || p === 1 ? 0 : -p * Math.log2(p) - (1 - p) * Math.log2(1 - p);
  function mappingInformation(plans, key) {
    const rows = plans.flatMap(p => p.cueValues.map((cue, i) =>
      ({ cue, mapping: p.mappings[i], bit: p[key][i] })));
    const byCue = [0, 1].map(cue => {
      const sample = rows.filter(row => row.cue === cue);
      const rates = [0, 1].map(mapping => {
        const selected = sample.filter(row => row.mapping === mapping);
        return selected.reduce((sum, row) => sum + row.bit, 0) / selected.length;
      });
      return { cue, bit1ByMapping: rates,
        informationBits: entropy((rates[0] + rates[1]) / 2) -
          (entropy(rates[0]) + entropy(rates[1])) / 2 };
    });
    return { byCue, conditionalBits: (byCue[0].informationBits + byCue[1].informationBits) / 2 };
  }
  const information = { trainingCorrect: mappingInformation(trainingPlans, 'correctBits'),
    trainingShuffled: mappingInformation(trainingPlans, 'shuffledBits'),
    heldoutCorrect: mappingInformation([heldout], 'correctBits'),
    heldoutShuffled: mappingInformation([heldout], 'shuffledBits') };
  if (['decode_mapping_signal_teacher', 'decode_mapping_signal_mechanism', 'decode_mapping_signal_counterfactual'].includes(mode) &&
      teacherLearning.some(row => !(row.frozenMean >= .75)))
    return { status: 'teacher failed frozen mapping accuracy gate', sourceSeed, threshold,
      teacherLearning, information };
  if (!replication && (information.heldoutCorrect.conditionalBits < .1 ||
      information.trainingShuffled.conditionalBits > .05 ||
      information.heldoutShuffled.conditionalBits > .05))
    return { status: 'stopped at source/control information gate', sourceSeed, threshold, information };
  const setting = { messageHz: 300, messageStart: 0, messageEnd: 40,
    cueStart: 70, cueEnd: 240, retainMbonCounts: true,
    messageTargets: [[506, 545], [113, 1421]] };
  const summarize = events => ({ correct: events.reduce((sum, row) => sum + row.correct, 0),
    total: events.length, actions: [0, 1, 2].map(action => events.filter(row => row.action === action).length),
    byMapping: [0, 1].map(mapping => {
      const rows = events.filter(row => row.mapping === mapping);
      return { mapping, correct: rows.reduce((sum, row) => sum + row.correct, 0), total: rows.length,
        actions: [0, 1, 2].map(action => rows.filter(row => row.action === action).length) };
    }), byCueMapping: [0, 1].flatMap(cue => [0, 1].map(mapping => {
      const rows = events.filter(row => row.cue === cue && row.mapping === mapping);
      return { cue, mapping, correct: rows.reduce((sum, row) => sum + row.correct, 0),
        total: rows.length, actions: [0, 1, 2].map(action => rows.filter(row => row.action === action).length) };
    })) });
  function frozen(weights, bits) {
    const pupil = fly(seed + 1500);
    pupil.net.m.set(weights);
    const windows = mechanism ? Array.from({ length: ROUNDS }, () => ({
      message: { spikes: 0, active: new Set() }, cue: { spikes: 0, active: new Set() } })) : null;
    const observeStep = mechanism ? (net, round, phase) => {
      const window = phase < 40 ? windows[round].message :
        phase >= 70 && phase < 240 ? windows[round].cue : null;
      if (!window) return;
      for (let i = 0; i < net.nSpikes; i++) if (sub.neurons.role[net.spikes[i]] === 'kc') {
        window.spikes++;
        window.active.add(net.spikes[i]);
      }
    } : null;
    const events = episode(pupil, 0, heldout.cueValues, false, mechanism, false,
      null, false, bits, { ...setting, mappingByRound: heldout.mappings, observeStep });
    if (pupil.net.m.some((weight, i) => weight !== weights[i])) throw Error('Frozen weights changed');
    return { ...summarize(events), ...(mechanism ? { neuralRows: events.map((row, i) => ({
      round: i, cue: row.cue, mapping: row.mapping, bit: row.bit ?? null,
      action: row.action, correct: row.correct, rates: row.rates,
      eligibilityByAction: row.eligibilityByAction, decisionKcSpikes: row.kcSpikes,
      decisionKcActive: row.kcActive,
      messageKcSpikes: windows[i].message.spikes,
      messageKcActive: [...windows[i].message.active],
      cueKcSpikes: windows[i].cue.spikes,
      cueKcActive: [...windows[i].cue.active] })) } : {}) };
  }
  function weightDeltas(net, initial) {
    return [0, 2].map(action => {
      const edges = net.plastic[action], pres = net.plasticPre[action];
      const summarizeEdges = selected => ({ edges: selected.length,
        signed: selected.reduce((sum, n) => sum + net.m[edges[n]] - initial[edges[n]], 0),
        absolute: selected.reduce((sum, n) => sum + Math.abs(net.m[edges[n]] - initial[edges[n]]), 0) });
      return { action, all: summarizeEdges(Array.from(edges, (_, n) => n)),
        messageGroups: [[506, 545], [113, 1421]].map(group =>
          summarizeEdges(Array.from(edges, (_, n) => n).filter(n => group.includes(pres[n])))) };
    });
  }
  const initialWeights = fly(seed + 500).net.m.slice();
  if (mode === 'decode_mapping_signal_counterfactual') {
    const pupil = fly(seed + 500);
    pupil.net.m.set(initialWeights);
    for (const p of trainingPlans)
      episode(pupil, 0, p.cueValues, true, false, false, null, true,
        p.correctBits, { ...setting, mappingByRound: p.mappings });
    const weights = pupil.net.m.slice();
    const forcedBits = heldout.correctBits.map((bit, i) =>
      heldout.cueValues[i] === 0 && heldout.mappings[i] === 0 ? 0 : bit);
    return { sourceSeed, threshold, schedule: setting, heldout, forcedBits,
      original: frozen(weights, heldout.correctBits), forced: frozen(weights, forcedBits) };
  }
  const paired = mode === 'decode_mapping_signal_paired';
  const before = { correct: frozen(initialWeights, heldout.correctBits),
    ...(paired ? { shuffled: frozen(initialWeights, heldout.shuffledBits) } : {}),
    ...(!replication ? { disconnected: frozen(initialWeights, null) } : {}) };
  const conditions = {};
  for (const kind of (replication ? ['correct', 'disconnected'] :
    ['correct', 'shuffled', 'untrained', 'disconnected'])) {
    const pupil = fly(seed + 500);
    pupil.net.m.set(initialWeights);
    const trainingRows = [];
    const training = trainingPlans.map((p, episodeIndex) => {
      const bits = kind === 'disconnected' ? null : p[`${kind === 'correct' ? 'correct' : kind === 'shuffled' ? 'shuffled' : 'untrained'}Bits`];
      const events = episode(pupil, 0, p.cueValues, true, mechanism, false,
        null, true, bits, { ...setting, mappingByRound: p.mappings });
      if (mechanism) trainingRows.push(...events.map(row => ({ episode: episodeIndex,
        round: row.round, cue: row.cue, mapping: row.mapping, bit: row.bit ?? null,
        action: row.action, correct: row.correct, rates: row.rates,
        kcSpikes: row.kcSpikes, eligibilityByAction: row.eligibilityByAction,
        eligibilitySum: row.eligibilitySum, dopamine: row.dopamine,
        weightDelta: row.weightDelta, weightChange: row.weightChange })));
      return summarize(events);
    });
    const weights = pupil.net.m.slice();
    conditions[kind] = { training, frozenCorrect: frozen(weights, heldout.correctBits),
      ...(paired ? { frozenShuffled: frozen(weights, heldout.shuffledBits) } : {}),
      ...(!replication ? { frozenDisconnected: frozen(weights, null) } : {}),
      ...(mechanism ? { trainingRows, weightDeltas: weightDeltas(pupil.net, initialWeights) } : {}) };
    console.log(replication
      ? `decode seed ${seed} ${kind}: frozen ${conditions[kind].frozenCorrect.correct}/24`
      : `decode seed ${seed} ${kind}: training ${training.map(row => row.correct).join(',')}, ` +
        `frozen correct ${conditions[kind].frozenCorrect.correct}/24, ` +
        `disconnected ${conditions[kind].frozenDisconnected.correct}/24`);
  }
  return { sourceSeed, threshold, episodes, schedule: setting, teacherLearning, information,
    trainingPlans, heldout, before, conditions };
}
function singleCueTransfer(baselineProbe = false) {
  const threshold = 1.915, studentSeed = matchedLearningProbe ? seed : seed + 500;
  const evaluationTeacherSeed = matchedLearningProbe ? teacherSeed : seed + 304;
  const trainingTeacherSeeds = freshLearningProbe ? [teacherSeed - 2, teacherSeed - 1] : [501, 502];
  const teacherSeeds = [...trainingTeacherSeeds, evaluationTeacherSeed];
  const setting = { messageHz: 300, messageStart: 0, messageEnd: 40,
    cueStart: 70, cueEnd: 240, retainMbonCounts: true,
    messageTargets: [[506, 545], [113, 1421]] };
  const sources = Object.fromEntries(teacherSeeds.map(id => [id, JSON.parse(readFileSync(
    new URL(`../data/two_cue_source_02_${id}_lanes02.json`, import.meta.url))).source]));
  const shuffle = (values, random) => {
    for (let i = values.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [values[i], values[j]] = [values[j], values[i]];
    }
    return values;
  };
  function samples(id, mapping, trained, test) {
    const source = sources[id], index = source.sourceIds.indexOf(2132);
    const run = source.runs.find(row => row.mapping === mapping && row.trained === trained);
    if (index < 0 || !run) throw Error('Missing teacher 2132 recording');
    const rows = run.samples.filter(row => row.test === test && row.cue === 1);
    if (rows.length !== 12) throw Error('Incomplete cue-1 teacher recording');
    return rows.map(row => ({ round: row.round, bit: Number(row.trace[index] < threshold) }));
  }
  function plan(id, test, planSeed) {
    const random = rng(planSeed), mappings = shuffle([...Array(12).fill(0), ...Array(12).fill(1)], random);
    const recorded = [0, 1].map(mapping => samples(id, mapping, true, test));
    const naive = samples(id, 0, false, test), used = [0, 0];
    const sourceRounds = mappings.map(mapping => recorded[mapping][used[mapping]++].round);
    used.fill(0);
    const correctBits = mappings.map(mapping => recorded[mapping][used[mapping]++].bit);
    const untrainedBits = mappings.map((_, i) => naive[i % 12].bit);
    const shuffledBits = Array(ROUNDS).fill(0), ones = correctBits.reduce((a, b) => a + b, 0);
    const first = Math.floor(ones / 2) + Number(ones % 2 && random() < .5);
    for (const mapping of [0, 1]) {
      const positions = mappings.flatMap((value, i) => value === mapping ? [i] : []);
      const count = mapping === 0 ? first : ones - first;
      const bits = shuffle([...Array(count).fill(1), ...Array(12 - count).fill(0)], random);
      positions.forEach((position, i) => { shuffledBits[position] = bits[i]; });
    }
    return { teacherSeed: id, test, sourceRounds, cueValues: Array(ROUNDS).fill(1),
      mappings, correctBits, untrainedBits, shuffledBits };
  }
  const heldout = plan(evaluationTeacherSeed, 2, seed * 1009 + 9000);
  const teacherFrozen = [0, 1].map(mapping => {
    const run = sources[evaluationTeacherSeed].runs.find(row => row.mapping === mapping && row.trained);
    return run.frozenAccuracy.reduce((a, b) => a + b, 0) / run.frozenAccuracy.length;
  });
  const bitCorrect = heldout.mappings.reduce((sum, mapping, i) =>
    sum + Number(heldout.correctBits[i] === 1 - mapping), 0);
  const teacherQuality = freshLearningProbe ? teacherSeeds.map(id => ({ seed: id,
    frozenAccuracy: [0, 1].map(mapping => {
      const run = sources[id].runs.find(row => row.mapping === mapping && row.trained);
      return run.frozenAccuracy.reduce((a, b) => a + b, 0) / run.frozenAccuracy.length;
    }),
    bitCorrect: [0, 1].reduce((sum, mapping) => sum + samples(id, mapping, true, 2)
      .filter(row => row.bit === 1 - mapping).length, 0) })) : null;
  if (!freshLearningProbe && (teacherFrozen.some(value => value < .75) || bitCorrect < 18))
    return { status: 'fresh teacher source gate failed', threshold, teacherFrozen, bitCorrect, heldout };
  if (mode === 'single_cue_preceding_bit') {
    const probes = heldout.mappings.flatMap((mapping, i) => mapping === 1 ?
      [{ sourceRound: heldout.sourceRounds[i], bit: heldout.correctBits[i] }] : []);
    if (probes.length !== 12 || probes.some(row => row.bit !== 0))
      throw Error('Expected 12 held-out teacher bit-0 probes');
    const cueValues = Array(ROUNDS).fill(1), mappings = Array(ROUNDS).fill(1);
    function replay(id, precedingBit) {
      const evaluator = fly(id), weights = evaluator.net.m.slice();
      const bits = Array.from({ length: ROUNDS }, (_, i) => i % 2 ? 0 : precedingBit);
      const events = episode(evaluator, 0, cueValues, false, false, false, null, false,
        bits, { ...setting, mappingByRound: mappings, resetEachPair: true });
      if (evaluator.net.m.some((weight, i) => weight !== weights[i]))
        throw Error('Frozen weights changed');
      const current = events.filter((_, i) => i % 2 === 1);
      return { correct: current.reduce((sum, row) => sum + row.correct, 0),
        action0Choices: current.filter(row => row.action === 0).length,
        meanAction2Rate: current.reduce((sum, row) => sum + row.rates[2], 0) / current.length,
        rows: current.map((row, i) => ({ sourceRound: probes[i].sourceRound,
          action: row.action, action2Rate: row.rates[2] })) };
    }
    return { command: `node tools/two_cue_transfer.mjs ${mode} ${seed} 02 lanes02 150`,
      hypothesis: 'preceding bit 1 raises student 701 current bit-0 action-2 drive',
      teacherSeed: evaluationTeacherSeed, studentSeeds: [701, 702], threshold,
      setting, probeSourceRounds: probes.map(row => row.sourceRound),
      pairProtocol: '12 frozen two-trial pairs; state reset before prior trial; cue 1 and no feedback in both trials; current trained teacher bit 0',
      students: Object.fromEntries([701, 702].map(id => [id, {
        precedingBit0: replay(id, 0), precedingBit1: replay(id, 1) }])) };
  }
  if (mode === 'single_cue_order_probe' || mode === 'single_cue_order_reset' || mode === 'single_cue_order_rng' || mode === 'single_cue_order_rng_full' || mode === 'single_cue_order_rng_reset' || mode === 'single_cue_order_conductance' || voltageProbe || refractoryProbe) {
    const resetEachRound = mode === 'single_cue_order_reset' || mode === 'single_cue_order_rng_reset';
    const stateProbe = mode === 'single_cue_order_conductance' || voltageProbe || refractoryProbe;
    const archived = stateProbe ? JSON.parse(readFileSync(new URL('../data/two_cue_single_cue_order_rng_full_02_202_lanes02.json', import.meta.url))).singleCueOrderRngFull : null;
    const orders = [20261103, 20261104].map(orderSeed => {
      const indices = shuffle(Array.from({ length: ROUNDS }, (_, i) => i), rng(orderSeed));
      const mappings = indices.map(i => heldout.mappings[i]);
      const bits = indices.map(i => heldout.correctBits[i]);
      function replay(id, connected) {
        const evaluator = fly(id), weights = evaluator.net.m.slice();
        const trialRandomSeeds = mode.startsWith('single_cue_order_rng') || stateProbe
          ? indices.map(i => id * 1000003 + i * 7919 + 23) : null;
        const trialAgentRandomSeeds = ['single_cue_order_rng_full', 'single_cue_order_rng_reset', 'single_cue_order_conductance', 'single_cue_order_voltage', 'single_cue_order_refractory'].includes(mode)
          ? trialRandomSeeds.map(value => value + 101) : null;
        const startAction2State = [];
        const rows = episode(evaluator, 0, Array(ROUNDS).fill(1), false, false,
          false, null, false, connected ? bits : null,
          { ...setting, mappingByRound: mappings, resetEachRound, trialRandomSeeds, trialAgentRandomSeeds,
            ...(stateProbe ? { observeRoundStart: (net, round) => {
              const group = net.groupMbon[2];
              const values = voltageProbe ? net.v : refractoryProbe ? net.refr : net.g;
              startAction2State[round] = group.reduce((sum, i) => sum + (refractoryProbe ? Number(values[i] > 0) : values[i]), 0) / group.length;
            } } : {}) });
        if (evaluator.net.m.some((weight, i) => weight !== weights[i]))
          throw Error('Frozen weights changed');
        if (stateProbe) {
          const expected = archived.orders.find(order => order.orderSeed === orderSeed).students[id][connected ? 'correct' : 'disconnected'].actions;
          if (rows.some((row, i) => row.action !== expected[i])) throw Error('State replay did not match archived actions');
        }
        return { correct: rows.reduce((sum, row) => sum + row.correct, 0),
          byMapping: [0, 1].map(mapping => ({ mapping,
            correct: rows.filter(row => row.mapping === mapping && row.correct).length })),
          actions: rows.map(row => row.action),
          ...(stateProbe ? { [refractoryProbe ? 'startAction2RefractoryFraction' : voltageProbe ? 'startAction2Voltage' : 'startAction2Conductance']: startAction2State } : {}) };
      }
      return { orderSeed, originalIndices: indices,
        sourceRounds: indices.map(i => heldout.sourceRounds[i]), mappings, bits,
        students: Object.fromEntries([701, 702].map(id => [id, {
          correct: replay(id, true), disconnected: replay(id, false) }])) };
    });
    const stateField = refractoryProbe ? 'startAction2RefractoryFraction' : voltageProbe ? 'startAction2Voltage' : 'startAction2Conductance';
    const disconnectedStateField = refractoryProbe ? 'disconnectedStartAction2RefractoryFraction' : voltageProbe ? 'disconnectedStartAction2Voltage' : 'disconnectedStartAction2Conductance';
    const discordant = stateProbe ? heldout.mappings.flatMap((mapping, originalIndex) => {
      if (mapping !== 1) return [];
      const positions = orders.map(order => order.originalIndices.indexOf(originalIndex));
      const values = orders.map((order, i) => ({ orderSeed: order.orderSeed,
        action: order.students[701].correct.actions[positions[i]],
        [stateField]: order.students[701].correct[stateField][positions[i]],
        [disconnectedStateField]: order.students[701].disconnected[stateField][positions[i]] }));
      if (values[0].action === values[1].action || !values.some(row => row.action === 2)) return [];
      const action2 = values.find(row => row.action === 2), other = values.find(row => row.action !== 2);
      return [{ originalIndex, sourceRound: heldout.sourceRounds[originalIndex],
        action2Order: action2.orderSeed, otherAction: other.action,
        difference: action2[stateField] - other[stateField],
        disconnectedDifference: action2[disconnectedStateField] - other[disconnectedStateField],
        values }];
    }) : null;
    const stateGate = stateProbe ? { discordantCount: discordant.length,
      higherInAction2Count: discordant.filter(row => row.difference > 0).length,
      meanDifference: discordant.reduce((sum, row) => sum + row.difference, 0) / discordant.length,
      passes: discordant.length > 0 && discordant.filter(row => row.difference > 0).length / discordant.length >= .75 &&
        discordant.reduce((sum, row) => sum + row.difference, 0) > 0 } : null;
    return { command: `node tools/two_cue_transfer.mjs ${mode} ${seed} 02 lanes02 150`,
      ...(stateProbe ? { [refractoryProbe ? 'refractoryGate' : voltageProbe ? 'voltageGate' : 'conductanceGate']: stateGate, discordant } : {}),
      hypothesis: stateProbe ? `trial-start action-2 MBON ${refractoryProbe ? 'refractory fraction' : voltageProbe ? 'voltage' : 'conductance'} predicts discordant mapping-B choices` :
        mode === 'single_cue_order_rng_reset' ? 'student 701 mapping-B order gap falls to at most 2/12 when neural state resets with fixed trial random streams' :
        mode.startsWith('single_cue_order_rng') ? 'student 701 mapping-B order gap is at most 2/12 with fixed trial random streams' :
        resetEachRound ? 'student 701 mapping-B gap between orders shrinks to at most 2/12' :
        'student 702 exceeds 701 by at least 4/12 mapping-B choices on both new orders',
      teacherSeed: evaluationTeacherSeed, studentSeeds: [701, 702], threshold,
      setting, resetEachRound, trialRandomSeedFormula: mode.startsWith('single_cue_order_rng') || stateProbe ? 'studentId * 1000003 + originalHeldoutIndex * 7919 + 23' : null,
      trialAgentRandomSeedFormula: ['single_cue_order_rng_full', 'single_cue_order_rng_reset', 'single_cue_order_conductance', 'single_cue_order_voltage', 'single_cue_order_refractory'].includes(mode) ? 'networkTrialSeed + 101' : null,
      originalHeldout: heldout, orders };
  }
  const trainingPlans = Array.from({ length: 8 }, (_, ep) =>
    plan(trainingTeacherSeeds[ep % 2], Math.floor(ep / 2) % 2, seed * 1009 + ep));
  const summarize = rows => ({ correct: rows.reduce((sum, row) => sum + row.correct, 0),
    byMapping: [0, 1].map(mapping => {
      const selected = rows.filter(row => row.mapping === mapping);
      return { mapping, correct: selected.reduce((sum, row) => sum + row.correct, 0),
        actions: [0, 1, 2].map(action => selected.filter(row => row.action === action).length) };
    }) });
  function frozen(weights, bits) {
    const evaluator = fly(studentSeed);
    evaluator.net.m.set(weights);
    const rows = episode(evaluator, 0, heldout.cueValues, false, false, false, null, false,
      bits, { ...setting, mappingByRound: heldout.mappings });
    if (evaluator.net.m.some((weight, i) => weight !== weights[i])) throw Error('Frozen weights changed');
    return summarize(rows);
  }
  const initialWeights = fly(studentSeed).net.m.slice();
  if (matchedLearningProbe) {
    const command = `node tools/two_cue_transfer.mjs ${mode} ${seed} 02 lanes02 150 ${teacherSeed}`;
    const changedBits = heldout.correctBits.filter((bit, i) => bit !== heldout.shuffledBits[i]).length;
    const bitCounts = bits => [0, 1].map(mapping => bits.filter((bit, i) =>
      heldout.mappings[i] === mapping && bit === 1).length);
    const correctCounts = bitCounts(heldout.correctBits), shuffledCounts = bitCounts(heldout.shuffledBits);
    const orders = [Array.from({ length: ROUNDS }, (_, i) => i),
      Array.from({ length: ROUNDS }, (_, i) => ROUNDS - 1 - i)];
    function replay(weights, bits) {
      return orders.map(indices => {
        const evaluator = fly(studentSeed);
        evaluator.net.m.set(weights);
        const trialRandomSeeds = indices.map(i => studentSeed * 1000003 + heldout.sourceRounds[i] * 7919 + 23);
        const rows = episode(evaluator, 0, indices.map(i => heldout.cueValues[i]), false,
          false, false, null, false, bits && indices.map(i => bits[i]), {
            ...setting, mappingByRound: indices.map(i => heldout.mappings[i]),
            resetEachRound: true, trialRandomSeeds,
            trialAgentRandomSeeds: trialRandomSeeds.map(value => value + 101) });
        if (evaluator.net.m.some((weight, i) => weight !== weights[i]))
          throw Error('Frozen weights changed');
        const indexed = indices.map((originalIndex, round) => ({ originalIndex,
          sourceRound: heldout.sourceRounds[originalIndex], bit: bits?.[originalIndex] ?? null,
          ...rows[round], zeroTie: rows[round].rates[0] === 0 && rows[round].rates[2] === 0 }));
        indexed.sort((a, b) => a.originalIndex - b.originalIndex);
        return { ...summarize(indexed), rows: indexed };
      });
    }
    const sameOrders = pair => pair[0].rows.every((row, i) =>
      row.action === pair[1].rows[i].action && row.rates.every((value, k) =>
        value === pair[1].rows[i].rates[k]));
    const before = { correct: replay(initialWeights, heldout.correctBits),
      shuffled: replay(initialWeights, heldout.shuffledBits),
      disconnected: replay(initialWeights, null) };
    const validity = { changedBits, correctCounts, shuffledCounts,
      headroom: before.correct[0].correct <= 18,
      shuffleBalanced: changedBits >= 6 &&
        shuffledCounts[0] + shuffledCounts[1] === correctCounts[0] + correctCounts[1] &&
        Math.abs(shuffledCounts[0] - shuffledCounts[1]) <= 1,
      orderInvariant: Object.values(before).every(sameOrders) };
    const base = { command, studentSeed, teacherSeed, trainingTeacherSeeds,
      threshold, setting, teacherFrozen, bitCorrect,
      ...(teacherQuality ? { teacherQuality } : {}), heldout, validity,
      before: Object.fromEntries(Object.entries(before).map(([kind, pair]) => [kind, pair[0]])) };
    if ((!prospectiveLearningProbe && !freshLearningProbe && !validity.headroom) ||
        !validity.shuffleBalanced || !validity.orderInvariant)
      return { ...base, status: 'evaluation validity gate failed' };
    const conditions = {};
    for (const kind of ['correct', 'shuffled', 'untrained', 'disconnected']) {
      const pupil = fly(studentSeed);
      pupil.net.m.set(initialWeights);
      const training = trainingPlans.map((planRow, ep) => {
        const trialRandomSeeds = planRow.sourceRounds.map((_, i) =>
          studentSeed * 1000003 + ep * 100003 + i * 7919 + 23);
        const bits = kind === 'disconnected' ? null : planRow[`${kind}Bits`];
        return summarize(episode(pupil, 0, planRow.cueValues, true, false, false,
          null, true, bits, { ...setting, mappingByRound: planRow.mappings,
            resetEachRound: true, trialRandomSeeds,
            trialAgentRandomSeeds: trialRandomSeeds.map(value => value + 101) }));
      });
      const weights = pupil.net.m.slice();
      const frozenCorrect = replay(weights, heldout.correctBits);
      conditions[kind] = { training, frozenCorrect: frozenCorrect[0],
        orderInvariant: sameOrders(frozenCorrect),
        ...(kind === 'correct' ? {
          frozenShuffled: replay(weights, heldout.shuffledBits),
          frozenDisconnected: replay(weights, null) } : {}) };
      if (kind === 'correct') {
        conditions[kind].orderInvariant &&= sameOrders(conditions[kind].frozenShuffled) &&
          sameOrders(conditions[kind].frozenDisconnected);
        conditions[kind].frozenShuffled = conditions[kind].frozenShuffled[0];
        conditions[kind].frozenDisconnected = conditions[kind].frozenDisconnected[0];
      }
    }
    const orderInvariant = Object.values(conditions).every(row => row.orderInvariant);
    const trained = conditions.correct.frozenCorrect;
    const margins = Object.fromEntries(['shuffled', 'untrained', 'disconnected'].map(kind =>
      [kind, trained.correct - conditions[kind].frozenCorrect.correct]));
    const dependence = { shuffled: trained.correct - conditions.correct.frozenShuffled.correct,
      disconnected: trained.correct - conditions.correct.frozenDisconnected.correct };
    const success = prospectiveLearningProbe || freshLearningProbe ? null : orderInvariant &&
      trained.correct - before.correct[0].correct >= 4 &&
      Object.values(margins).every(value => value >= 4) &&
      trained.byMapping.every(row => row.correct >= 8) &&
      Object.values(dependence).every(value => value >= 4);
    return { ...base, status: orderInvariant ? 'valid' : 'evaluation validity gate failed',
      trainingPlans, conditions, orderInvariant, margins, dependence,
      ownGain: trained.correct - before.correct[0].correct, success };
  }
  if (learningResetProbe || disconnectedControlProbe) {
    const archived = JSON.parse(readFileSync(new URL(
      '../data/two_cue_single_cue_transfer_02_202_lanes02.json', import.meta.url))).singleCueTransfer;
    if (JSON.stringify(heldout) !== JSON.stringify(archived.heldout) ||
        JSON.stringify(trainingPlans) !== JSON.stringify(archived.trainingPlans))
      throw Error('Source or training schedule differs from archived replication');
    const trialRandomSeeds = heldout.sourceRounds.map((_, i) => studentSeed * 1000003 + i * 7919 + 23);
    const trialAgentRandomSeeds = trialRandomSeeds.map(value => value + 101);
    function resetFrozen(weights, bits) {
      const evaluator = fly(studentSeed);
      evaluator.net.m.set(weights);
      const rows = episode(evaluator, 0, heldout.cueValues, false, false, false,
        null, false, bits, { ...setting, mappingByRound: heldout.mappings,
          resetEachRound: true, trialRandomSeeds, trialAgentRandomSeeds });
      if (evaluator.net.m.some((weight, i) => weight !== weights[i]))
        throw Error('Frozen weights changed');
      return summarize(rows);
    }
    if (frozen(initialWeights, heldout.correctBits).correct !== archived.before.correct)
      throw Error('Initial state did not reproduce archived score');
    const before = resetFrozen(initialWeights, heldout.correctBits);
    const conditions = {};
    for (const kind of disconnectedControlProbe ? ['disconnected'] : ['correct', 'untrained', 'shuffled', 'disconnected']) {
      const pupil = fly(studentSeed);
      pupil.net.m.set(initialWeights);
      const training = trainingPlans.map(planRow => {
        const bits = kind === 'disconnected' ? null : planRow[`${kind}Bits`];
        return summarize(episode(pupil, 0, planRow.cueValues, true, false, false,
          null, true, bits, { ...setting, mappingByRound: planRow.mappings }));
      });
      if (training.some((row, i) => row.correct !== archived.conditions[kind].training[i].correct))
        throw Error(`Training replay differs for ${kind}`);
      const weights = pupil.net.m.slice();
      if (frozen(weights, heldout.correctBits).correct !== archived.conditions[kind].frozenCorrect.correct)
        throw Error(`Trained state did not reproduce archived score for ${kind}`);
      conditions[kind] = { frozenCorrect: resetFrozen(weights, heldout.correctBits),
        ...((kind === 'correct' || disconnectedControlProbe) ? { frozenDisconnected: resetFrozen(weights, null),
          frozenShuffled: resetFrozen(weights, heldout.shuffledBits) } : {}) };
    }
    if (disconnectedControlProbe) {
      const observed = conditions.disconnected;
      const reference = JSON.parse(readFileSync(new URL(
        '../data/two_cue_single_cue_learning_reset_02_202_lanes02.json', import.meta.url))).singleCueLearningReset;
      if (observed.frozenCorrect.correct !== reference.conditions.disconnected.frozenCorrect.correct)
        throw Error('Disconnected control does not reproduce fixed evaluation score');
      const drops = { shuffled: observed.frozenCorrect.correct - observed.frozenShuffled.correct,
        disconnected: observed.frozenCorrect.correct - observed.frozenDisconnected.correct };
      return { command: 'node tools/two_cue_transfer.mjs single_cue_disconnected_control 202 02 lanes02 150',
        hypothesis: 'disconnected-trained student perfect score requires informative evaluation messages',
        teacherSeed: evaluationTeacherSeed, studentSeed, trainingTeacherSeeds: [501, 502],
        trainingEpisodes: trainingPlans.length, threshold, setting,
        evaluation: { resetEachRound: true,
          trialRandomSeedFormula: 'studentId * 1000003 + originalHeldoutIndex * 7919 + 23',
          trialAgentRandomSeedFormula: 'networkTrialSeed + 101' },
        conditions: observed, drops, gatePassed: Object.values(drops).every(value => value >= 4) };
    }
    const gain = conditions.correct.frozenCorrect.correct - before.correct;
    const margins = Object.fromEntries(['untrained', 'shuffled', 'disconnected'].map(kind =>
      [kind, conditions.correct.frozenCorrect.correct - conditions[kind].frozenCorrect.correct]));
    return { command: 'node tools/two_cue_transfer.mjs single_cue_learning_reset 202 02 lanes02 150',
      hypothesis: 'reward-trained student state improves beyond immediate teacher stimulation under round-reset fixed-stream evaluation',
      teacherSeed: evaluationTeacherSeed, studentSeed, threshold, setting,
      trainingTeacherSeeds: [501, 502], trainingEpisodes: trainingPlans.length,
      evaluation: { resetEachRound: true,
        trialRandomSeedFormula: 'studentId * 1000003 + originalHeldoutIndex * 7919 + 23',
        trialAgentRandomSeedFormula: 'networkTrialSeed + 101' },
      before, conditions, gain, margins,
      gatePassed: gain >= 4 && Object.values(margins).every(value => value >= 4) };
  }
  if (mode === 'single_cue_initial_drive' || mode === 'single_cue_round_reset') {
    function initialReplay(id, bits, resetEachRound = false) {
      const evaluator = fly(id), weights = evaluator.net.m.slice(), messageRates = [];
      const observeStep = (net, round, phase) => {
        if (phase === 39) messageRates[round] = net.groupMbon.map(group =>
          group.reduce((sum, neuron) => sum + net.count[neuron], 0) / group.length);
      };
      const rows = episode(evaluator, 0, heldout.cueValues, false, false, false, null, false,
        bits, { ...setting, mappingByRound: heldout.mappings, observeStep, resetEachRound });
      if (evaluator.net.m.some((weight, i) => weight !== weights[i])) throw Error('Frozen weights changed');
      return { ...summarize(rows), rows: rows.map((row, i) => ({ round: i,
        mapping: row.mapping, bit: bits?.[i] ?? null, action: row.action,
        correct: row.correct, messageRates: messageRates[i], decisionRates: row.rates })) };
    }
    const isolated = mode === 'single_cue_round_reset';
    return { command: `node tools/two_cue_transfer.mjs ${mode} ${seed} 02 lanes02 150`,
      hypothesis: isolated ? 'action-2 suppression in student 702 persists when neural state is reset before every round' :
        'initial message-evoked action-0 MBON drive is stronger in student 702 than 701',
      teacherSeed: evaluationTeacherSeed, threshold, setting, heldout,
      students: Object.fromEntries([701, 702].map(id => [id, {
        correct: initialReplay(id, heldout.correctBits, isolated),
        disconnected: initialReplay(id, null, isolated) }])) };
  }
  const before = frozen(initialWeights, heldout.correctBits),
    beforeDisconnected = frozen(initialWeights, null), conditions = {};
  if (baselineProbe) return {
    command: `node tools/two_cue_transfer.mjs single_cue_baseline_probe ${seed} 02 lanes02 150`,
    studentSeed, evaluationTeacherSeed, sourceNeuron: 2132, threshold, setting,
    teacherFrozen, bitCorrect, heldout,
    frozenInitial: { correct: before, shuffled: frozen(initialWeights, heldout.shuffledBits),
      untrained: frozen(initialWeights, heldout.untrainedBits), disconnected: beforeDisconnected } };
  for (const kind of ['correct', 'untrained', 'shuffled', 'disconnected']) {
    const pupil = fly(studentSeed);
    pupil.net.m.set(initialWeights);
    const training = trainingPlans.map(planRow => {
      const bits = kind === 'disconnected' ? null : planRow[`${kind}Bits`];
      return summarize(episode(pupil, 0, planRow.cueValues, true, false, false, null, true,
        bits, { ...setting, mappingByRound: planRow.mappings }));
    });
    const weights = pupil.net.m.slice();
    conditions[kind] = { training, frozenCorrect: frozen(weights, heldout.correctBits),
      ...(kind === 'correct' ? { frozenDisconnected: frozen(weights, null),
        frozenShuffled: frozen(weights, heldout.shuffledBits) } : {}) };
    console.log(`single-cue seed ${studentSeed} ${kind}: ${conditions[kind].frozenCorrect.correct}/24`);
  }
  const trained = conditions.correct.frozenCorrect;
  const success = trained.correct >= before.correct + 4 &&
    trained.byMapping.every(row => row.correct >= 8) &&
    ['untrained', 'shuffled', 'disconnected'].every(kind =>
      trained.correct >= conditions[kind].frozenCorrect.correct + 3);
  return { command: `node tools/two_cue_transfer.mjs single_cue_transfer ${seed} 02 lanes02 150`,
    sourceNeuron: 2132, threshold, trainingTeacherSeeds: [501, 502], evaluationTeacherSeed,
    studentSeed, setting, teacherFrozen, bitCorrect, trainingPlans, heldout,
    before, beforeDisconnected, conditions, pilotGatePassed: success };
}
function interferenceRun() {
  const saved = JSON.parse(readFileSync(new URL(`../data/two_cue_teach_02_${seed}_lanes02.json`, import.meta.url)));
  const rows = saved.teaching.teacherStreams.trained.rows.slice(0, ROUNDS);
  if (rows.length !== ROUNDS || rows.some((row, i) => row.round !== i))
    throw Error('Missing aligned teacher messages');
  const cueValues = rows.map(row => row.cue), bits = rows.map(row => row.bit);
  const pupil = fly(seed + 500), training = [];
  for (let ep = 0; ep < 2 * EPISODES; ep++) {
    const mapping = Number(ep >= EPISODES);
    const events = episode(pupil, mapping, cues(seed * 1009 + ep));
    training.push(events.reduce((sum, e) => sum + e.correct, 0) / ROUNDS);
  }
  const weights = pupil.net.m.slice();
  const settings = mode === 'interference_retain'
    ? [{ name: 'kc_precue_retain', messageHz: 300, messageStart: 0,
      messageEnd: 40, cueStart: 70, cueEnd: 240, retainMbonCounts: true,
      messageTargets: [[506, 545], [113, 1421]] }]
    : mode === 'interference_temporal'
    ? [{ name: 'kc_before_cue', messageHz: 300, messageStart: 0,
      messageEnd: 40, cueStart: 70, cueEnd: 240,
      messageTargets: [[1319, 2016], [818, 1462]] }]
    : mode === 'interference_kc'
    ? [{ name: 'kc_pair_120ms_300hz', messageHz: 300, messageStart: 120,
      messageEnd: 240, cueEnd: 240, messageTargets: [[1319, 2016], [818, 1462]] }]
    : mode === 'interference_late'
    ? [{ name: 'late_150hz_50ms', messageHz: 150, messageStart: 190,
      messageEnd: 240, cueEnd: 180 }]
    : [
    { name: 'original', messageHz: 300, messageStart: 120, messageEnd: 240, cueEnd: 240 },
    { name: 'weaker', messageHz: 150, messageStart: 120, messageEnd: 240, cueEnd: 240 },
    { name: 'shorter', messageHz: 300, messageStart: 180, messageEnd: 240, cueEnd: 240 },
    { name: 'nonoverlap', messageHz: 300, messageStart: 180, messageEnd: 240, cueEnd: 180 },
  ];
  const results = [];
  for (const setting of settings) {
    const conditions = {}, windows = {};
    const conditionList = mode === 'interference_retain'
      ? [['cueOnlyRetained', null, true], ['combinedRetained', bits, true],
        ['cueOnlyOriginal', null, true], ['combined0', Array(ROUNDS).fill(0), true],
        ['combined1', Array(ROUNDS).fill(1), true]]
      : mode === 'interference_temporal'
      ? [['cueOnly', null, true], ['signalOnly', bits, false], ['combined', bits, true]]
      : [
      ['cueOnly', null, true], ['signalOnly', bits, false], ['combined', bits, true],
      ['combined0', Array(ROUNDS).fill(0), true], ['combined1', Array(ROUNDS).fill(1), true],
    ];
    for (const [name, message, cueVisible] of conditionList) {
      const evaluator = fly(seed + 500);
      evaluator.net.m.set(weights);
      let observeStep = null;
      if (mode === 'interference_temporal' || mode === 'interference_retain') {
        const window = () => ({ kc: 0, mbon: [0, 0, 0] });
        windows[name] = Array.from({ length: ROUNDS }, () => ({ message: window(),
          gap: window(), lateGap: window(), cue: window(), decision: window() }));
        observeStep = (net, round, phase) => {
          if (phase >= 240) return;
          const names = [phase < 40 ? 'message' : phase < 70 ? 'gap' : 'cue'];
          if (phase >= 60 && phase < 70) names.push('lateGap');
          if (phase >= 190 && phase < 240) names.push('decision');
          for (let i = 0; i < net.nSpikes; i++) {
            const neuron = net.spikes[i], role = sub.neurons.role[neuron];
            if (role === 'kc') names.forEach(key => { windows[name][round][key].kc++; });
            if (role === 'mbon') names.forEach(key => {
              windows[name][round][key].mbon[net.groupOf[neuron]]++;
            });
          }
        };
      }
      conditions[name] = episode(evaluator, 1, cueValues, false, true, false, null, false,
        message, { ...setting, cueVisible, observeStep,
          retainMbonCounts: setting.retainMbonCounts && name !== 'cueOnlyOriginal' });
      if (evaluator.net.m.some((weight, i) => weight !== weights[i]))
        throw Error('Interference probe changed frozen weights');
    }
    const accuracy = name => conditions[name].reduce((sum, row) => sum + row.correct, 0) / ROUNDS;
    console.log(mode === 'interference_retain'
      ? `interference seed ${seed} ${setting.name}: cue ${accuracy('cueOnlyRetained')}, combined ${accuracy('combinedRetained')}`
      : `interference seed ${seed} ${setting.name}: cue ${accuracy('cueOnly')}, combined ${accuracy('combined')}`);
    results.push(mode === 'interference_temporal' || mode === 'interference_retain'
      ? { setting, conditions, windows } : { setting, conditions });
  }
  return { training, messageSource: 'saved trained Fly A trace bits from first post-switch episode',
    cueValues, bits, settings: results };
}
function encodingChannelProbe() {
  const thresholds = JSON.parse(readFileSync(new URL('../data/two_cue_encoding_check.json', import.meta.url))).thresholds;
  const ids = [2146, 2134], available = [0, 1].map(() => [[], [], [], []]);
  for (const teacherSeed of [501, 502]) {
    const source = JSON.parse(readFileSync(new URL(`../data/two_cue_source_02_${teacherSeed}_lanes02.json`, import.meta.url))).source;
    const indices = ids.map(id => source.sourceIds.indexOf(id));
    for (const run of source.runs.filter(row => row.trained)) for (const row of run.samples) {
      const code = 2 * Number(row.trace[indices[0]] > thresholds[0]) +
        Number(row.trace[indices[1]] > thresholds[1]);
      available[row.cue][code].push({ teacherSeed, mapping: run.mapping, test: row.test, round: row.round, code });
    }
  }
  if (available.some(byCue => byCue.some(rows => rows.length === 0))) throw Error('Missing source pattern');
  const cueValues = cues(seed * 1009 + 55), occurrences = [0, 0];
  const exampleSources = cueValues.map(cue => available[cue][occurrences[cue]++ % 4][0]);
  const messageCodes = exampleSources.map(source => source.code);
  const pupil = fly(seed + 500);
  for (let ep = 0; ep < 2 * EPISODES; ep++)
    episode(pupil, Number(ep >= EPISODES), cues(seed * 1009 + ep));
  const weights = pupil.net.m.slice();
  const setting = { messageHz: 300, messageStart: 0, messageEnd: 40,
    cueStart: 70, cueEnd: 240, retainMbonCounts: true,
    messageTargets: [[506, 545], [113, 1421]] };
  function frozen(codes) {
    const evaluator = fly(seed + 500);
    evaluator.net.m.set(weights);
    const targetSpikes = Array.from({ length: ROUNDS }, () => [0, 0]);
    const observeStep = (net, round, phase) => {
      if (phase >= 40) return;
      for (let i = 0; i < net.nSpikes; i++)
        setting.messageTargets.forEach((group, k) => {
          if (group.includes(net.spikes[i])) targetSpikes[round][k]++;
        });
    };
    const events = episode(evaluator, 1, cueValues, false, true, false, null, false,
      null, { ...setting, messageCodes: codes, observeStep });
    if (evaluator.net.m.some((weight, i) => weight !== weights[i]))
      throw Error('Frozen weights changed');
    return events.map((event, round) => ({ cue: event.cue, action: event.action,
      correct: event.correct, rates: event.rates, targetSpikes: targetSpikes[round] }));
  }
  const cueOnly = frozen(null), combined = frozen(messageCodes);
  const byCode = [0, 1, 2, 3].map(code => {
    const rounds = messageCodes.flatMap((value, round) => value === code ? [round] : []);
    return { code, n: rounds.length,
      targetSpikes: [0, 1].map(i => rounds.reduce((sum, round) => sum + combined[round].targetSpikes[i], 0) / rounds.length),
      decisionRates: [0, 2].map(action => rounds.reduce((sum, round) => sum + combined[round].rates[action], 0) / rounds.length) };
  });
  console.log('channel cue-only', cueOnly.reduce((s, r) => s + r.correct, 0),
    'combined', combined.reduce((s, r) => s + r.correct, 0), 'of', ROUNDS);
  return { command: `node tools/two_cue_transfer.mjs encoding_channel_probe ${seed} 02 lanes02 150`,
    purpose: 'Frozen channel feasibility only; message patterns from saved calibration Fly A activity, no student learning',
    ids, thresholds: thresholds.slice(0, 2), setting, cueValues, messageCodes,
    exampleSources, cueOnly, combined, byCode };
}
function encodingChannelNatural() {
  const teacherSeed = 504, studentSeed = seed + 500;
  const source = JSON.parse(readFileSync(new URL('../data/two_cue_source_02_504_lanes02.json', import.meta.url))).source;
  const indices = [2146, 2134].map(id => source.sourceIds.indexOf(id));
  const run = source.runs.find(row => row.mapping === 1 && row.trained);
  if (indices.includes(-1) || !run || run.frozenAccuracy.reduce((a, b) => a + b, 0) / run.frozenAccuracy.length < .75)
    throw Error('Teacher source failed validation');
  const thresholds = [0.000368, 0.232596];
  const samples = run.samples.filter(row => row.test === 0).sort((a, b) => a.round - b.round);
  if (samples.length !== ROUNDS || samples.some((row, i) => row.round !== i))
    throw Error('Incomplete teacher round sequence');
  const cueValues = samples.map(row => row.cue);
  const sourceMessages = samples.map(row => {
    const traces = indices.map(i => row.trace[i]);
    const bits = traces.map((value, i) => Number(value < thresholds[i]));
    return { teacherSeed, mapping: 1, test: row.test, round: row.round,
      cue: row.cue, traces, bits, code: bits[0] * 2 + bits[1] };
  });
  const codes = sourceMessages.map(row => row.code), shuffledCodes = codes.slice();
  const random = rng(seed * 7919 + teacherSeed);
  for (const cue of [0, 1]) {
    const positions = cueValues.flatMap((value, i) => value === cue ? [i] : []);
    const values = positions.map(i => codes[i]);
    for (let i = values.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [values[i], values[j]] = [values[j], values[i]];
    }
    positions.forEach((position, i) => { shuffledCodes[position] = values[i]; });
  }
  const pupil = fly(studentSeed);
  for (let ep = 0; ep < 2 * EPISODES; ep++)
    episode(pupil, Number(ep >= EPISODES), cues(seed * 1009 + ep));
  const weights = pupil.net.m.slice();
  const setting = { messageHz: 300, messageStart: 0, messageEnd: 40,
    cueStart: 70, cueEnd: 240, retainMbonCounts: true,
    messageTargets: [[506, 545], [113, 1421]] };
  if (mode === 'encoding_channel_active0_cue') setting.zeroBitTarget = [1319, 2016];
  function frozen(messageCodes) {
    const evaluator = fly(studentSeed);
    evaluator.net.m.set(weights);
    const targetGroups = [...setting.messageTargets, ...(setting.zeroBitTarget ? [setting.zeroBitTarget] : [])];
    const activity = Array.from({ length: ROUNDS }, () => ({ target: targetGroups.map(() => 0),
      kc: [0, 0], mbon: [[0, 0], [0, 0]] }));
    const observeStep = (net, round, phase) => {
      const window = phase < 40 ? 0 : phase >= 70 && phase < 240 ? 1 : -1;
      if (window < 0) return;
      for (let i = 0; i < net.nSpikes; i++) {
        const neuron = net.spikes[i], role = sub.neurons.role[neuron];
        if (role === 'kc') activity[round].kc[window]++;
        if (role === 'mbon') for (const [k, action] of [0, 2].entries())
          if (net.groupOf[neuron] === action) activity[round].mbon[window][k]++;
        if (window === 0) targetGroups.forEach((group, k) => {
          if (group.includes(neuron)) activity[round].target[k]++;
        });
      }
    };
    const events = episode(evaluator, 1, cueValues, false, true, false, null, false,
      null, { ...setting, messageCodes, observeStep });
    if (evaluator.net.m.some((weight, i) => weight !== weights[i]))
      throw Error('Frozen student weights changed');
    return events.map((event, round) => ({ cue: event.cue, action: event.action,
      correct: event.correct, decisionRates: event.rates, ...activity[round],
      message: messageCodes ? { connected: true, code: messageCodes[round] } :
        { connected: false, code: null } }));
  }
  const cueOnly = frozen(null), actual = frozen(codes), shuffled = frozen(shuffledCodes);
  console.log('natural channel seed', studentSeed, 'accuracy',
    [cueOnly, actual, shuffled].map(rows => rows.reduce((sum, row) => sum + row.correct, 0)).join('/'),
    'of', ROUNDS);
  return { command: `node tools/two_cue_transfer.mjs ${mode} ${seed} 02 lanes02 150`,
    teacherSeed, studentSeed, thresholds, setting, sourceMessages, codes, shuffledCodes,
    changedByShuffle: codes.filter((code, i) => code !== shuffledCodes[i]).length,
    cueOnly, actual, shuffled };
}
function twoCueBlockSpeedPilot() {
  // The archived seed-904 output omitted zeroBitTarget; that run is invalid.
  // This source restores the previously fixed receiver and has not been rerun.
  const studentSeed = seed, trainingTeacher = 603, evaluationTeacher = 606;
  const thresholds = [0.000368, 0.232596];
  const setting = { cueStart: 70, cueEnd: 240, messageStart: 0, messageEnd: 40,
    messageHz: 300, messageTargets: [[506, 545], [113, 1421]],
    zeroBitTarget: [1319, 2016],
    retainMbonCounts: true, resetEachRound: true };
  const archived = JSON.parse(readFileSync(new URL(
    '../data/two_cue_receiver_frozen_02_20_lanes02.json', import.meta.url))).receiverFrozen;
  if (ROUNDS !== 24 || EPISODES !== 10 ||
      JSON.stringify(thresholds) !== JSON.stringify(archived.thresholds) ||
      Object.entries(archived.setting).some(([key, value]) =>
        JSON.stringify(setting[key]) !== JSON.stringify(value)) || !setting.resetEachRound)
    throw Error('Pilot receiver or budget differs from locked archived interface');
  function loadTeacher(id) {
    const source = JSON.parse(readFileSync(new URL(
      `../data/two_cue_source_02_${id}_lanes02.json`, import.meta.url))).source;
    const indices = [2146, 2134].map(neuron => source.sourceIds.indexOf(neuron));
    if (source.sampleMs !== 120 || indices.some(i => i < 0)) throw Error('Fixed source missing');
    return { source, indices };
  }
  const teachers = { 603: loadTeacher(603), 606: loadTeacher(606) };
  function codes(id, mapping, test, cueValues, trained = true) {
    const { source, indices } = teachers[id];
    const run = source.runs.find(row => row.mapping === mapping && row.trained === trained);
    if (!run) throw Error('Missing teacher run');
    const queue = [0, 1].map(cue => run.samples.filter(row => row.test === test && row.cue === cue)
      .sort((a, b) => a.round - b.round));
    if (queue.some(rows => rows.length !== 12)) throw Error('Unbalanced teacher recording');
    return cueValues.map(cue => {
      const row = queue[cue].shift();
      if (!row) throw Error('Teacher cue alignment failed');
      const bits = indices.map((i, k) => Number(row.trace[i] < thresholds[k]));
      return bits[0] * 2 + bits[1];
    });
  }
  const trainingCues = Array.from({ length: 20 }, (_, ep) => cues(2026110200 + ep % 10));
  if (trainingCues.some(values => [0, 1].some(cue => values.filter(x => x === cue).length !== 12)) ||
      trainingCues.slice(0, 10).some((values, i) => values.join() !== trainingCues[i + 10].join()))
    throw Error('Training sequence validity failed');
  const evalCues = cues(2026110300);
  const trainCodes = trainingCues.map((values, ep) => {
    const mapping = Number(ep >= 10), test = ep % 2;
    const correct = codes(trainingTeacher, mapping, test, values);
    const opposite = codes(trainingTeacher, 1 - mapping, test, values);
    const shuffled = correct.map((code, i) => i % 2 ? opposite[i] : code);
    const untrained = codes(trainingTeacher, 0, test, values, false);
    return { correct, shuffled, untrained };
  });
  const swapped = trainCodes.reduce((n, row) => n + row.correct.filter((v, i) => v !== row.shuffled[i]).length, 0);
  if (swapped < 40) throw Error('Shuffled teacher codes insufficiently changed');
  function streams(index) {
    const trialRandomSeeds = evalCues.map((_, round) => studentSeed * 1000000 + index * 1000 + round);
    return { trialRandomSeeds, trialAgentRandomSeeds: trialRandomSeeds.map(n => n + 7919) };
  }
  const evalCodes = [0, 1].map(mapping => {
    const correct = codes(evaluationTeacher, mapping, 2, evalCues);
    const opposite = codes(evaluationTeacher, 1 - mapping, 2, evalCues);
    return { correct, shuffled: correct.map((code, i) => i % 2 ? opposite[i] : code) };
  });
  function frozen(weights, mapping, message) {
    const evaluator = fly(studentSeed);
    evaluator.net.m.set(weights);
    const messageCodes = message === 'off' ? null : evalCodes[mapping][message];
    const events = episode(evaluator, mapping, evalCues, false, false, false, null, false, null,
      { ...setting, ...streams(100 + mapping), messageCodes });
    if (evaluator.net.m.some((weight, i) => weight !== weights[i]))
      throw Error('Frozen weights changed');
    return { correct: events.reduce((n, row) => n + row.correct, 0),
      byCue: [0, 1].map(cue => events.filter(row => row.cue === cue && row.correct).length),
      trials: events.map(row => ({ cue: row.cue, mapping: row.mapping, action: row.action,
        correct: row.correct, rates: row.rates })) };
  }
  const initialWeights = fly(studentSeed).net.m.slice();
  function arm(kind) {
    const student = fly(studentSeed);
    student.net.m.set(initialWeights);
    const initial = [0, 1].map(mapping => frozen(initialWeights, mapping, 'correct'));
    const training = [], checkpoints = [], ablations = {};
    ablations.startA = { off: frozen(initialWeights, 0, 'off'),
      shuffled: frozen(initialWeights, 0, 'shuffled') };
    for (let ep = 0; ep < 20; ep++) {
      const mapping = Number(ep >= 10), values = trainingCues[ep];
      const trialRandomSeeds = values.map((_, round) => studentSeed * 1000000 + ep * 1000 + round);
      const messageCodes = kind === 'off' ? null : trainCodes[ep][kind];
      const events = episode(student, mapping, values, true, false, false, null, true, null,
        { ...setting, trialRandomSeeds,
          trialAgentRandomSeeds: trialRandomSeeds.map(n => n + 7919), messageCodes });
      training.push({ episode: ep, mapping, correct: events.reduce((n, row) => n + row.correct, 0),
        byCue: [0, 1].map(cue => events.filter(row => row.cue === cue && row.correct).length) });
      const weights = student.net.m.slice();
      checkpoints.push(frozen(weights, mapping, 'correct'));
      if (ep === 9 || ep === 19) {
        const key = ep === 9 ? 'afterA' : 'afterB';
        ablations[key] = { off: frozen(weights, mapping, 'off'),
          shuffled: frozen(weights, mapping, 'shuffled') };
        if (ep === 9) ablations.startB = { off: frozen(weights, 1, 'off'),
          shuffled: frozen(weights, 1, 'shuffled') };
      }
    }
    const speed = [0, 1].map(block => {
      const early = checkpoints.slice(block * 10, block * 10 + 5);
      return { total: early.reduce((n, row) => n + row.correct, 0) / 5,
        byCue: [0, 1].map(cue => early.reduce((n, row) => n + row.byCue[cue], 0) / 5) };
    });
    return { kind, initial, training, checkpoints, ablations, speed };
  }
  const alone = arm('off');
  const learnability = {
    afterA: alone.ablations.afterA.off.correct >= 18 &&
      alone.ablations.afterA.off.byCue.every(n => n >= 8) &&
      alone.ablations.afterA.off.correct - alone.ablations.startA.off.correct >= 4,
    afterB: alone.ablations.afterB.off.correct >= 18 &&
      alone.ablations.afterB.off.byCue.every(n => n >= 8) &&
      alone.ablations.afterB.off.correct - alone.ablations.startB.off.correct >= 4
  };
  if (!Object.values(learnability).every(Boolean)) {
    console.log(`block pilot stopped: no-message gate A ${alone.ablations.startA.off.correct}->${alone.ablations.afterA.off.correct}, ` +
      `B ${alone.ablations.startB.off.correct}->${alone.ablations.afterB.off.correct}`);
    return { command: `node tools/two_cue_transfer.mjs ${mode} ${seed} 02 lanes02 150`,
      studentSeed, trainingTeacher, evaluationTeacher, thresholds, setting,
      switchAfterTrials: 240, swappedTrainingCodes: swapped,
      learnability, stop: 'No-message learnability validity gate failed; teacher arms not trained', alone };
  }
  const conditions = { off: alone, correct: arm('correct'),
    shuffled: arm('shuffled'), untrained: arm('untrained') };
  const controls = [conditions.off, conditions.shuffled, conditions.untrained];
  const speedGates = [0, 1].map(block => controls.every(control =>
    conditions.correct.speed[block].total - control.speed[block].total >= 3 &&
    [0, 1].every(cue => conditions.correct.speed[block].byCue[cue] -
      control.speed[block].byCue[cue] >= 1) &&
    conditions.correct.checkpoints[block * 10 + 9].correct >=
      control.checkpoints[block * 10 + 9].correct - 2));
  return { command: `node tools/two_cue_transfer.mjs ${mode} ${seed} 02 lanes02 150`,
    studentSeed, trainingTeacher, evaluationTeacher, thresholds, setting,
    switchAfterTrials: 240, swappedTrainingCodes: swapped, learnability,
    speedGates, pass: speedGates.every(Boolean), conditions };
}
function twoCueAloneBaseline() {
  const studentSeed = seed, setting = { cueStart: 70, cueEnd: 240,
    retainMbonCounts: true, resetEachRound: true };
  const shuffle = (values, random) => {
    for (let i = values.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [values[i], values[j]] = [values[j], values[i]];
    }
    return values;
  };
  const plan = index => {
    const random = rng(2026110200 + index);
    const trials = shuffle([0, 1].flatMap(cue => [0, 1].flatMap(mapping =>
      Array.from({ length: 6 }, () => ({ cue, mapping })))), random);
    return trials.map((row, trialIndex) => ({ ...row, trialIndex }));
  };
  const trainingPlans = Array.from({ length: 8 }, (_, ep) => plan(ep));
  const heldoutPlans = [plan(100), plan(101)];
  const balanced = p => [0, 1].every(cue => [0, 1].every(mapping =>
    p.filter(r => r.cue === cue && r.mapping === mapping).length === 6));
  if (![...trainingPlans, ...heldoutPlans].every(balanced)) throw Error('Unbalanced no-message schedule');
  const randomSeeds = (rows, index) => rows.map(r => studentSeed * 1000000 + index * 1000 + r.trialIndex);
  function runEpisode(student, rows, index, learn) {
    const trialRandomSeeds = randomSeeds(rows, index);
    return episode(student, 0, rows.map(r => r.cue), learn, false, false, null, learn,
      null, { ...setting, mappingByRound: rows.map(r => r.mapping), trialRandomSeeds,
        trialAgentRandomSeeds: trialRandomSeeds.map(n => n + 7919) });
  }
  const compact = (events, rows) => events.map((event, i) => ({ trialIndex: rows[i].trialIndex,
    cue: event.cue, mapping: event.mapping, action: event.action,
    correctAction: event.correctAction, correct: event.correct, rates: event.rates,
    zeroRateTie: event.rates.every(rate => rate === 0) }));
  function frozen(weights, rows, index) {
    const evaluator = fly(studentSeed);
    evaluator.net.m.set(weights);
    const original = compact(runEpisode(evaluator, rows, index, false), rows);
    if (evaluator.net.m.some((weight, i) => weight !== weights[i])) throw Error('Frozen weights changed');
    const reversedRows = [...rows].reverse(), reverseEvaluator = fly(studentSeed);
    reverseEvaluator.net.m.set(weights);
    const reversed = compact(runEpisode(reverseEvaluator, reversedRows, index, false), reversedRows);
    if (reverseEvaluator.net.m.some((weight, i) => weight !== weights[i])) throw Error('Reverse frozen weights changed');
    const byIndex = new Map(reversed.map(r => [r.trialIndex, r]));
    if (original.some(row => {
      const other = byIndex.get(row.trialIndex);
      return !other || row.action !== other.action || row.rates.some((rate, i) => rate !== other.rates[i]);
    })) throw Error('Frozen trial-order invariance failed');
    return original;
  }
  const student = fly(studentSeed), initialWeights = student.net.m.slice();
  const before = heldoutPlans.map((rows, i) => frozen(initialWeights, rows, 100 + i));
  const training = trainingPlans.map((rows, ep) => {
    const events = runEpisode(student, rows, ep, true);
    return { episode: ep, correct: events.reduce((n, e) => n + e.correct, 0),
      byCueMapping: [0, 1].flatMap(cue => [0, 1].map(mapping =>
        events.filter(e => e.cue === cue && e.mapping === mapping && e.correct).length)) };
  });
  const finalWeights = student.net.m.slice();
  const weightChange = { changed: 0, absolute: 0 };
  finalWeights.forEach((weight, i) => {
    const delta = Math.abs(weight - initialWeights[i]);
    if (delta) { weightChange.changed++; weightChange.absolute += delta; }
  });
  const after = heldoutPlans.map((rows, i) => frozen(finalWeights, rows, 100 + i));
  const summarize = sets => {
    const rows = sets.flat();
    return { correct: rows.reduce((n, r) => n + r.correct, 0), total: rows.length,
      byCue: [0, 1].map(cue => ({ cue, correct: rows.filter(r => r.cue === cue && r.correct).length,
        total: 24, zeroRateTies: rows.filter(r => r.cue === cue && r.zeroRateTie).length })),
      byCueMapping: [0, 1].flatMap(cue => [0, 1].map(mapping => ({ cue, mapping,
        correct: rows.filter(r => r.cue === cue && r.mapping === mapping && r.correct).length,
        total: 12 }))) };
  };
  const initial = summarize(before), final = summarize(after);
  const gates = { overallAtLeast36: final.correct >= 36,
    everyCellAtLeast8: final.byCueMapping.every(r => r.correct >= 8),
    gainAtLeast8: final.correct - initial.correct >= 8 };
  console.log(`alone seed ${seed}: initial ${initial.correct}/48 final ${final.correct}/48; ` +
    `cells ${final.byCueMapping.map(r => r.correct).join('/')}; ` +
    `training ${training.map(r => r.correct).join(',')}`);
  return { command: `node tools/two_cue_transfer.mjs two_cue_alone_baseline ${seed} 02 lanes02 150`,
    studentSeed, actualMapping: 'actions[cue XOR mapping]', noTeacherMessage: true,
    setting, scheduleSeedFormula: '2026110200 + schedule index',
    trialRandomSeedFormula: 'studentSeed * 1000000 + schedule index * 1000 + original trial index',
    trainingPlans, heldoutPlans, training, weightChange, before, after,
    initial, final, gates, pass: Object.values(gates).every(Boolean),
    validity: { balanced: true, reverseOrderInvariant: true, frozenWeightsInvariant: true } };
}
function receiverFrozenReplay(studentSeed, weights, setting, reconstructed) {
  const sourceGate = JSON.parse(readFileSync(new URL('../data/two_cue_receiver_gate.json', import.meta.url)));
  if (!sourceGate.pass) throw Error('Gated source-preservation gate failed');
  const archive = JSON.parse(readFileSync(new URL(
    `../data/two_cue_decode_two_bit_cue_gated_probe_02_${seed}_lanes02.json`, import.meta.url))).channelPair.original;
  const sameActions = (a, b) => a.length === b.length && a.every((r, i) =>
    r.action === b[i].action && r.correctAction === b[i].correctAction);
  if (reconstructed.training.length !== archive.training.length ||
      reconstructed.training.some((episodeRows, i) => !sameActions(episodeRows, archive.training[i])) ||
      !sameActions(reconstructed.after.correct, archive.after.correct) ||
      !sameActions(reconstructed.after.disconnected, archive.after.disconnected) ||
      reconstructed.heldout.codes.join() !== archive.heldout.codes.join())
    throw Error('Archived student reconstruction mismatch');

  const teachers = [603, 606], rows = [];
  for (const teacher of teachers) {
    const source = JSON.parse(readFileSync(new URL(
      `../data/two_cue_source_02_${teacher}_lanes02.json`, import.meta.url))).source;
    const indices = [2146, 2134].map(id => source.sourceIds.indexOf(id));
    if (source.sampleMs !== 120 || indices.some(i => i < 0)) throw Error('Missing fixed teacher source');
    for (const mapping of [0, 1]) {
      const run = source.runs.find(r => r.trained && r.mapping === mapping);
      if (!run) throw Error('Missing trained teacher run');
      for (const sample of run.samples.filter(s => s.test === 2)) {
        const bits = indices.map((i, k) => Number(sample.trace[i] < reconstructed.thresholds[k]));
        rows.push({ teacher, mapping, sourceRound: sample.round, cue: sample.cue,
          code: bits[0] * 2 + bits[1], key: `${teacher}_${mapping}_${sample.round}` });
      }
    }
  }
  if (rows.length !== 96 || teachers.some(teacher => [0, 1].some(mapping =>
    [0, 1].some(cue => rows.filter(r => r.teacher === teacher && r.mapping === mapping && r.cue === cue).length !== 12))))
    throw Error('Unbalanced actual-mapping source schedule');
  for (const row of rows) {
    const sameCueRounds = rows.filter(r => r.teacher === row.teacher && r.mapping === row.mapping && r.cue === row.cue)
      .map(r => r.sourceRound).sort((a, b) => a - b);
    const partner = rows.find(r => r.teacher === row.teacher && r.mapping === 1 - row.mapping &&
      r.sourceRound === row.sourceRound && r.cue === row.cue);
    if (!partner) throw Error('Missing paired source round for shuffle');
    row.shuffledCode = sameCueRounds.indexOf(row.sourceRound) % 2 ? partner.code : row.code;
  }
  const conditionSpec = {
    originalCorrect: { gated: false, message: 'code' },
    gatedCorrect: { gated: true, message: 'code' },
    originalShuffled: { gated: false, message: 'shuffledCode' },
    gatedShuffled: { gated: true, message: 'shuffledCode' },
    disconnected: { gated: false, message: null }
  };
  function evaluate(trials, spec, reverse = false) {
    const ordered = reverse ? [...trials].reverse() : trials;
    const output = [];
    for (let start = 0; start < ordered.length; start += ROUNDS) {
      const block = ordered.slice(start, start + ROUNDS);
      if (block.length !== ROUNDS) throw Error('Incomplete frozen block');
      const evaluator = fly(studentSeed);
      evaluator.net.m.set(weights);
      const windows = block.map(() => ({ messageMbon: [0, 0], cueSpikes: 0 }));
      const cueTargets = [0, 2].map(lane => evaluator.net.laneVpn[lane].slice(WIDTH));
      const observeStep = (net, round, phase) => {
        const window = windows[round];
        for (let i = 0; i < net.nSpikes; i++) {
          const neuron = net.spikes[i];
          if (phase < 40 && sub.neurons.role[neuron] === 'mbon') {
            if (net.groupOf[neuron] === 0) window.messageMbon[0]++;
            if (net.groupOf[neuron] === 2) window.messageMbon[1]++;
          }
          if (phase >= 70 && phase < 240 && cueTargets[block[round].cue].includes(neuron))
            window.cueSpikes++;
        }
      };
      const trialRandomSeeds = block.map(r => studentSeed * 1000000 + r.teacher * 10000 + r.mapping * 1000 + r.sourceRound);
      const trialAgentRandomSeeds = trialRandomSeeds.map(n => n + 7919);
      const events = episode(evaluator, 0, block.map(r => r.cue), false, false, false, null, false,
        null, { ...setting, cueGatedBits: spec.gated, resetEachRound: true,
          messageCodes: spec.message ? block.map(r => r[spec.message]) : null,
          mappingByRound: block.map(r => r.mapping), trialRandomSeeds, trialAgentRandomSeeds, observeStep });
      if (evaluator.net.m.some((weight, i) => weight !== weights[i])) throw Error('Frozen weights changed');
      output.push(...events.map((event, i) => ({ ...block[i], action: event.action,
        correctAction: event.correctAction, correct: event.correct, rates: event.rates,
        zeroRateTie: event.rates.every(rate => rate === 0), ...windows[i] })));
    }
    return output;
  }
  const evaluations = {};
  for (const [name, spec] of Object.entries(conditionSpec)) {
    const first = evaluate(rows, spec), reversed = evaluate(rows, spec, true);
    const byKey = new Map(reversed.map(r => [r.key, r]));
    if (first.some(r => {
      const other = byKey.get(r.key);
      return !other || r.action !== other.action || r.rates.some((rate, i) => rate !== other.rates[i]) ||
        r.cueSpikes !== other.cueSpikes || r.messageMbon.some((n, i) => n !== other.messageMbon[i]);
    })) throw Error(`Source-keyed order failure: ${name}`);
    evaluations[name] = first;
  }
  const cueOnly = evaluate(rows.filter(r => r.teacher === 603 && r.mapping === 0), conditionSpec.disconnected);
  const cueRecognition = [0, 1].map(cue => cueOnly.filter(r => r.cue === cue && r.correct).length);
  const valid = cueRecognition.every(n => n >= 8);
  const count = (name, cue, mapping = null) => evaluations[name].filter(r => r.cue === cue &&
    (mapping === null || r.mapping === mapping) && r.correct).length;
  const zero = (name, cue) => evaluations[name].filter(r => r.cue === cue && r.zeroRateTie).length;
  const byReceiver = Object.fromEntries(['original', 'gated'].map(receiver => {
    const correctName = `${receiver}Correct`, shuffleName = `${receiver}Shuffled`;
    const byCue = [0, 1].map(cue => ({ cue, correct: count(correctName, cue),
      byMapping: [0, 1].map(mapping => count(correctName, cue, mapping)),
      shuffled: count(shuffleName, cue), disconnected: count('disconnected', cue),
      zeroRateTies: zero(correctName, cue) }));
    const gates = { cueAccuracy: byCue.every(c => c.correct >= 36),
      mappingAccuracy: byCue.every(c => c.byMapping.every(n => n >= 16)),
      shuffledMargin: byCue.every(c => c.correct - c.shuffled >= 8),
      disconnectedMargin: byCue.every(c => c.correct - c.disconnected >= 8),
      zeroTies: byCue.every(c => c.zeroRateTies <= zero('originalCorrect', c.cue)) };
    if (receiver === 'gated') {
      const pattern = (name, code, mapping) => evaluations[name].filter(r =>
        r.cue === 1 && r.code === code && r.mapping === mapping);
      const original10 = pattern('originalCorrect', 2, 1), gated10 = pattern('gatedCorrect', 2, 1);
      const margin = group => group.reduce((n, r) => n + r.rates[2] - r.rates[0], 0) / group.length;
      const original00 = pattern('originalCorrect', 0, 0), gated00 = pattern('gatedCorrect', 0, 0);
      gates.pattern10Drive = original10.length > 0 && gated10.length === original10.length &&
        margin(gated10) < margin(original10);
      gates.pattern00Choices = original00.length > 0 && gated00.length === original00.length &&
        gated00.filter(r => r.action === 2).length >= original00.filter(r => r.action === 2).length;
    }
    return [receiver, { byCue, gates, pass: valid && Object.values(gates).every(Boolean) }];
  }));
  console.log(`receiver frozen student ${studentSeed}: cue-only ${cueRecognition.join('/')} of 12; ` +
    `original ${byReceiver.original.byCue.map(r => r.correct).join('/')} gated ${byReceiver.gated.byCue.map(r => r.correct).join('/')} of 48`);
  return { teachers, studentSeed, sourceGroundTruth: 'actual teacher mapping',
    archivedReconstruction: true, orderInvariant: true, frozenWeightsInvariant: true,
    cueOnly: { byCueCorrectOutOf12: cueRecognition, valid }, byReceiver, evaluations };
}
function decodeTwoBit(targetBitOverride = null, mechanism = false, balancedTraining = false, cueGatedBits = false) {
  const targetBitIndex = mechanism ? targetBitOverride :
    mode === 'decode_bit2146' ? 0 : mode === 'decode_bit2134' ? 1 :
    ['decode_two_bit_active0_conditional', 'decode_two_bit_cue_gated_pair', 'decode_two_bit_cue_gated_probe', 'receiver_frozen'].includes(mode) ? 2 : null;
  const thresholds = [0.000368, 0.232596];
  const setting = { messageHz: 300, messageStart: 0, messageEnd: 40,
    cueStart: 70, cueEnd: 240, retainMbonCounts: true,
    messageTargets: [[506, 545], [113, 1421]] };
  if (['decode_bit2146_active0', 'decode_two_bit_active0_conditional', 'decode_two_bit_active0_probe', 'decode_two_bit_cue_gated_pair', 'decode_two_bit_cue_gated_probe', 'receiver_frozen'].includes(mode))
    setting.zeroBitTarget = [1319, 2016];
  if (cueGatedBits) setting.cueGatedBits = true;
  const pool = {};
  for (const teacherSeed of [501, 502, 503, 504]) {
    const source = JSON.parse(readFileSync(new URL(`../data/two_cue_source_02_${teacherSeed}_lanes02.json`, import.meta.url))).source;
    const indices = [2146, 2134].map(id => source.sourceIds.indexOf(id));
    if (indices.includes(-1)) throw Error('Missing source neuron');
    pool[teacherSeed] = source.runs.filter(run => run.trained).flatMap(run => run.samples.map(row => {
      const traces = indices.map(i => row.trace[i]);
      const bits = traces.map((value, i) => Number(value < thresholds[i]));
      return { teacherSeed, sourceMapping: run.mapping, test: row.test, sourceRound: row.round,
        cue: row.cue, traces, bits, code: bits[0] * 2 + bits[1] };
    }));
  }
  function plan(teacherSeeds, planSeed) {
    const random = rng(planSeed), trials = [];
    for (let copy = 0; copy < 3; copy++) for (const cue of [0, 1]) for (const code of [0, 1, 2, 3]) {
      const options = teacherSeeds.flatMap(teacherSeed => pool[teacherSeed])
        .filter(row => row.cue === cue && row.code === code);
      if (!options.length) throw Error(`Missing recorded pattern ${cue}/${code}`);
      trials.push(options[Math.floor(random() * options.length)]);
    }
    for (let i = trials.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [trials[i], trials[j]] = [trials[j], trials[i]];
    }
    return { sources: trials, cues: trials.map(row => row.cue),
      codes: trials.map(row => row.code),
      mappings: trials.map(row => targetBitIndex === null ? (row.bits[0] ^ row.bits[1]) :
        (row.cue ^ row.bits[targetBitIndex === 2 ? row.cue : targetBitIndex])) };
  }
  const heldout = plan([503, 504], seed * 1009 + 9000);
  const shuffledCodes = heldout.codes.slice(), random = rng(seed * 7919 + 9091);
  for (const cue of [0, 1]) {
    const positions = heldout.cues.flatMap((value, i) => value === cue ? [i] : []);
    const values = positions.map(i => shuffledCodes[i]);
    for (let i = values.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [values[i], values[j]] = [values[j], values[i]];
    }
    positions.forEach((position, i) => { shuffledCodes[position] = values[i]; });
  }
  const studentSeed = seed + 500, student = fly(studentSeed), initialWeights = student.net.m.slice();
  const compact = (events, codes, ep = null) => events.map((event, round) =>
    ({ ...(ep === null ? {} : { episode: ep }), round, cue: event.cue,
      code: codes?.[round] ?? null, action: event.action, correctAction: event.correctAction,
      correct: event.correct }));
  function frozen(weights, codes) {
    const evaluator = fly(studentSeed);
    evaluator.net.m.set(weights);
    const events = episode(evaluator, 0, heldout.cues, false, false, false, null, false,
      null, { ...setting, messageCodes: codes, mappingByRound: heldout.mappings });
    if (evaluator.net.m.some((weight, i) => weight !== weights[i])) throw Error('Frozen weights changed');
    return compact(events, codes);
  }
  const before = { correct: frozen(initialWeights, heldout.codes),
    disconnected: frozen(initialWeights, null), shuffled: frozen(initialWeights, shuffledCodes) };
  const training = [], trainingNeural = [];
  for (let ep = 0; ep < 8; ep++) {
    const trial = plan([501, 502], seed * 1009 + ep);
    const events = episode(student, 0, trial.cues, true, mechanism, balancedTraining, null, true,
      null, { ...setting, messageCodes: trial.codes, mappingByRound: trial.mappings });
    training.push(compact(events, trial.codes, ep));
    if (mechanism) trainingNeural.push(...events.map((row, round) => ({ episode: ep, round,
      cue: row.cue, code: trial.codes[round], action: row.action, correct: row.correct,
      rates: row.rates, eligibilityByAction: row.eligibilityByAction,
      dopamine: row.dopamine, weightDelta: row.weightDelta, weightChange: row.weightChange })));
  }
  const trainedWeights = student.net.m.slice();
  const after = { correct: frozen(trainedWeights, heldout.codes),
    disconnected: frozen(trainedWeights, null), shuffled: frozen(trainedWeights, shuffledCodes) };
  const nuisanceOffCodes = targetBitIndex === null || targetBitIndex === 2 ? null : heldout.codes.map(code =>
    targetBitIndex === 0 ? code & 2 : code & 1);
  if (nuisanceOffCodes) after.nuisanceOff = frozen(trainedWeights, nuisanceOffCodes);
  const score = rows => rows.reduce((sum, row) => sum + row.correct, 0);
  console.log(mode, 'seed', studentSeed, 'before/after',
    score(before.correct), score(after.correct), 'disconnected', score(after.disconnected),
    'shuffled', score(after.shuffled), 'of', ROUNDS);
  let pathway = null;
  if (mechanism) {
    const evaluator = fly(studentSeed);
    evaluator.net.m.set(trainedWeights);
    const targetGroups = [...setting.messageTargets, ...(setting.zeroBitTarget ? [setting.zeroBitTarget] : [])];
    let windows = Array.from({ length: ROUNDS }, () => ({ messageKc: 0, cueKc: 0,
      messageMbon: [0, 0], decisionMbon: [0, 0], targetKc: targetGroups.map(() => 0) }));
    const observeStep = (net, round, phase) => {
      const w = windows[round], message = phase < 40, cue = phase >= 70 && phase < 240;
      if (!message && !cue) return;
      for (let i = 0; i < net.nSpikes; i++) {
        const neuron = net.spikes[i], role = sub.neurons.role[neuron];
        if (role === 'kc') { if (message) w.messageKc++; else w.cueKc++; }
        if (message) targetGroups.forEach((group, k) => {
          if (group.includes(neuron)) w.targetKc[k]++;
        });
        if (role === 'mbon') for (const [k, action] of [0, 2].entries())
          if (net.groupOf[neuron] === action) {
            if (message) w.messageMbon[k]++;
            if (phase >= 190) w.decisionMbon[k]++;
          }
      }
    };
    const neural = episode(evaluator, 0, heldout.cues, false, true, false, null, false,
      null, { ...setting, messageCodes: heldout.codes, mappingByRound: heldout.mappings, observeStep });
    if (neural.some((row, i) => row.action !== after.correct[i].action) ||
        evaluator.net.m.some((weight, i) => weight !== trainedWeights[i]))
      throw Error('Frozen pathway probe changed the result');
    const weightDeltas = [0, 2].map(action => {
      const edges = student.net.plastic[action], pres = student.net.plasticPre[action];
      const summarize = group => {
        const chosen = Array.from(edges, (_, i) => i).filter(i => !group || group.includes(pres[i]));
        return { edges: chosen.length,
          signed: chosen.reduce((sum, i) => sum + trainedWeights[edges[i]] - initialWeights[edges[i]], 0),
          absolute: chosen.reduce((sum, i) => sum + Math.abs(trainedWeights[edges[i]] - initialWeights[edges[i]]), 0) };
      };
      return { action, all: summarize(null), targetKcs: targetGroups.map(summarize) };
    });
    pathway = { frozen: neural.map((row, i) => ({ cue: row.cue, code: heldout.codes[i],
      action: row.action, rates: row.rates, eligibilityByAction: row.eligibilityByAction,
      ...windows[i] })), trainingNeural, weightDeltas };
    if (mode === 'decode_two_bit_cue_gated_probe') {
      windows = Array.from({ length: ROUNDS }, () => ({ messageKc: 0, cueKc: 0,
        messageMbon: [0, 0], decisionMbon: [0, 0], targetKc: targetGroups.map(() => 0) }));
      const control = fly(studentSeed);
      control.net.m.set(trainedWeights);
      const cross = episode(control, 0, heldout.cues, false, true, false, null, false,
        null, { ...setting, cueGatedBits: !cueGatedBits, messageCodes: heldout.codes,
          mappingByRound: heldout.mappings, observeStep });
      if (control.net.m.some((weight, i) => weight !== trainedWeights[i]))
        throw Error('Cross-interface probe changed frozen weights');
      pathway.crossInterface = cross.map((row, i) => ({ cue: row.cue, code: heldout.codes[i],
        action: row.action, rates: row.rates, eligibilityByAction: row.eligibilityByAction,
        ...windows[i] }));
    }
    if (mode === 'decode_two_bit_active0_probe') {
      pathway.irrelevantBitFlips = heldout.codes.flatMap((code, round) =>
        heldout.cues[round] === 1 && code === 2 ? [round] : []).map(round => {
        const codes = heldout.codes.slice();
        codes[round] = 0; // cue 1 still rewards action 0; only the irrelevant 2146 bit changes.
        windows = Array.from({ length: ROUNDS }, () => ({ messageKc: 0, cueKc: 0,
          messageMbon: [0, 0], decisionMbon: [0, 0], targetKc: targetGroups.map(() => 0) }));
        const control = fly(studentSeed);
        control.net.m.set(trainedWeights);
        const events = episode(control, 0, heldout.cues, false, true, false, null, false,
          null, { ...setting, messageCodes: codes, mappingByRound: heldout.mappings, observeStep });
        if (events.slice(0, round).some((row, i) => row.action !== neural[i].action) ||
            control.net.m.some((weight, i) => weight !== trainedWeights[i]))
          throw Error('Counterfactual changed earlier rounds or frozen weights');
        const row = events[round];
        return { round, original: pathway.frozen[round], flipped: { cue: row.cue,
          code: codes[round], action: row.action, rates: row.rates,
          eligibilityByAction: row.eligibilityByAction, ...windows[round] } };
      });
    }
  }
  return { command: `node tools/two_cue_transfer.mjs ${mode} ${seed} 02 lanes02 150`,
    studentSeed, thresholds, setting, targetBitIndex, balancedTraining,
    rule: targetBitIndex === null ? 'correct action = actions[cue XOR bit2146 XOR bit2134]' :
      targetBitIndex === 2 ? 'cue 0: actions[bit2146]; cue 1: actions[bit2134]' :
      `correct action = actions[bit${targetBitIndex === 0 ? 2146 : 2134}]`,
    trainingTeacherSeeds: [501, 502], evaluationTeacherSeeds: [503, 504],
    heldout, shuffledCodes, shuffledChanged: heldout.codes.filter((code, i) => code !== shuffledCodes[i]).length,
    ...(nuisanceOffCodes ? { nuisanceOffCodes } : {}),
    before, training, after, ...(pathway ? { pathway } : {}),
    ...(mode === 'receiver_frozen' ? { receiverFrozen: receiverFrozenReplay(studentSeed, trainedWeights,
      setting, { thresholds, heldout, training, after }) } : {}) };
}
function gradedPopulationDecode() {
  const ids = [2146, 2134, 2124, 2122];
  const targetGroups = [[506, 545], [113, 1421], [1319, 2016], [818, 1462]];
  const teacherSeeds = ['graded_population_pathway', 'graded_population_choice', 'graded_population_drive', 'graded_population_ties'].includes(mode) ? [501, 502, 503, 504, 505, 506] : [501, 502, 503, 504];
  const pool = Object.fromEntries(teacherSeeds.map(id => {
    const source = JSON.parse(readFileSync(new URL(`../data/two_cue_source_02_${id}_lanes02.json`, import.meta.url))).source;
    const indices = ids.map(neuron => source.sourceIds.indexOf(neuron));
    if (indices.includes(-1)) throw Error(`Missing population source for teacher ${id}`);
    const rows = [0, 1].flatMap(mapping => {
      const run = source.runs.find(row => row.trained && row.mapping === mapping);
      if (!run || run.frozenAccuracy.reduce((a, b) => a + b, 0) / run.frozenAccuracy.length < .75)
        throw Error(`Teacher ${id} failed frozen source gate`);
      return run.samples.filter(row => row.cue === 1).map(row => ({ teacher: id, mapping,
        test: row.test, sourceRound: row.round, trace: indices.map(index => row.trace[index]) }));
    });
    return [id, rows];
  }));
  const calibration = [501, 502].flatMap(id => pool[id]);
  const ceilings = ids.map((_, i) => {
    const sorted = calibration.map(row => row.trace[i]).sort((a, b) => a - b);
    return sorted[Math.ceil(.95 * sorted.length) - 1];
  });
  if (ceilings.some(x => x <= 0)) throw Error('Invalid calibration ceiling');
  const rates = row => row.trace.map((value, i) => 300 * Math.min(1, Math.max(0, value / ceilings[i])));
  const shuffle = (rows, random) => {
    for (let i = rows.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [rows[i], rows[j]] = [rows[j], rows[i]];
    }
    return rows;
  };
  const plan = (teachers, test, planSeed) => {
    const rows = teachers.flatMap(id => [0, 1].flatMap(mapping =>
      pool[id].filter(row => row.mapping === mapping && row.test === test).slice(0, 6)));
    if (rows.length !== ROUNDS || [0, 1].some(mapping => rows.filter(row => row.mapping === mapping).length !== 12))
      throw Error('Incomplete balanced population plan');
    return shuffle(rows, rng(planSeed));
  };
  const setting = { messageStart: 0, messageEnd: 40, cueVisible: false,
    messageTargets: targetGroups, resetEachRound: true, retainMbonCounts: true };
  const heldout = plan([503, 504], 2, seed * 1009 + 9000);
  const a = heldout.filter(row => row.mapping === 0), b = heldout.filter(row => row.mapping === 1);
  const random = rng(seed * 7919 + 9091);
  const shuffledByMapping = [shuffle([...a.slice(0, 6), ...b.slice(0, 6)], random),
    shuffle([...a.slice(6), ...b.slice(6)], random)];
  const used = [0, 0];
  const shuffled = heldout.map(row => shuffledByMapping[row.mapping][used[row.mapping]++]);
  const studentSeed = seed + 500, student = fly(studentSeed), initialWeights = student.net.m.slice();
  const cueValues = Array(ROUNDS).fill(0);
  const trialRandomSeeds = heldout.map((_, i) => studentSeed * 100000 + i * 2 + 1);
  const trialAgentRandomSeeds = heldout.map((_, i) => studentSeed * 100000 + i * 2 + 2);
  function frozen(weights, messages) {
    const evaluator = fly(studentSeed);
    evaluator.net.m.set(weights);
    const events = episode(evaluator, 0, cueValues, false, false, false, null, false, null,
      { ...setting, mappingByRound: heldout.map(row => row.mapping),
        trialRandomSeeds, trialAgentRandomSeeds,
        ...(messages ? { gradedMessages: messages.map(rates) } : {}) });
    if (evaluator.net.m.some((weight, i) => weight !== weights[i])) throw Error('Frozen weights changed');
    return events.map((row, i) => ({ mapping: heldout[i].mapping, action: row.action, correct: row.correct }));
  }
  const before = frozen(initialWeights, heldout);
  const training = [];
  for (let ep = 0; ep < 8; ep++) {
    const rows = plan([501, 502], ep % 2, seed * 1009 + ep);
    const events = episode(student, 0, cueValues, true, false, false, null, true, null,
      { ...setting, mappingByRound: rows.map(row => row.mapping), gradedMessages: rows.map(rates) });
    training.push(events.reduce((sum, row) => sum + row.correct, 0));
  }
  const trainedWeights = student.net.m.slice();
  const after = { correct: frozen(trainedWeights, heldout),
    shuffled: frozen(trainedWeights, shuffled), disconnected: frozen(trainedWeights, null) };
  const score = rows => rows.reduce((sum, row) => sum + row.correct, 0);
  if (mode === 'graded_population_initial_silence') {
    const archived = JSON.parse(readFileSync(new URL('../data/two_cue_graded_population_decode_02_203_lanes02.json', import.meta.url))).gradedPopulationDecode;
    if (JSON.stringify(before) !== JSON.stringify(archived.before) ||
        JSON.stringify(training) !== JSON.stringify(archived.training) ||
        JSON.stringify(after) !== JSON.stringify(archived.after))
      throw Error('Initial student reconstruction differs from archived run');
    const sourceRows = teacher => pool[teacher].filter(row => row.test === 2);
    const record = (rows, messages) => {
      const evaluator = fly(studentSeed);
      evaluator.net.m.set(initialWeights);
      const trialRandomSeeds = rows.map(row => studentSeed * 1000000 + row.teacher * 10000 + row.test * 100 + row.sourceRound);
      const events = episode(evaluator, 0, cueValues, false, false, false, null, false, null,
        { ...setting, mappingByRound: rows.map(row => row.mapping), trialRandomSeeds,
          trialAgentRandomSeeds: trialRandomSeeds.map(x => x + 500000),
          ...(messages ? { gradedMessages: messages.map(rates) } : {}) });
      if (evaluator.net.m.some((weight, i) => weight !== initialWeights[i])) throw Error('Initial frozen weights changed');
      return events.map((event, i) => ({ mapping: rows[i].mapping, sourceRound: rows[i].sourceRound,
        rates: event.rates, action: event.action }));
    };
    const summarize = rows => ({ byMapping: [0, 1].map(mapping => rows.filter(row =>
      row.mapping === mapping && row.rates.every(rate => rate === 0)).length), trials: rows });
    const evaluation = Object.fromEntries([503, 504].map(teacher => {
      const rows = sourceRows(teacher);
      if (rows.length !== 24 || rows.filter(row => row.mapping === 0).length !== 12)
        throw Error('Unbalanced initial silence plan');
      const shuffled = [...rows.slice(6, 12), ...rows.slice(12, 18), ...rows.slice(0, 6), ...rows.slice(18, 24)];
      return [teacher, { correct: summarize(record(rows, rows)),
        shuffled: summarize(record(rows, shuffled)), disconnected: summarize(record(rows, null)) }];
    }));
    return { command: `node tools/two_cue_transfer.mjs ${mode} ${seed} 02 lanes02 150`,
      studentSeed, evaluationTeachers: [503, 504], trainingEpisodesForInitialWeights: 0,
      gate: 'Correct-message B-minus-A zero ties >= 4/12 on both teachers supports a structural channel effect; <= 2/12 on both supports a training-dependent effect; otherwise mixed or inconclusive',
      evaluation };
  }
  let pathway = null;
  if (['graded_population_pathway', 'graded_population_choice', 'graded_population_drive', 'graded_population_ties', 'graded_population_silence503'].includes(mode)) {
    const archived = JSON.parse(readFileSync(new URL('../data/two_cue_graded_population_decode_02_203_lanes02.json', import.meta.url))).gradedPopulationDecode;
    if (JSON.stringify(training) !== JSON.stringify(archived.training) ||
        JSON.stringify(after) !== JSON.stringify(archived.after))
      throw Error('Trained student reconstruction differs from archived run');
    const mbonIndices = [...student.net.groupMbon[0], ...student.net.groupMbon[2]];
    const record = (rows, connected = true, messageRows = rows) => {
      if (rows.length !== ROUNDS) throw Error('Pathway plan must have 24 rounds');
      const evaluator = fly(studentSeed);
      evaluator.net.m.set(trainedWeights);
      const measures = rows.map(row => ({ mapping: row.mapping, teacher: row.teacher,
        test: row.test, sourceRound: row.sourceRound,
        kc: targetGroups.map(() => 0), mbon: mbonIndices.map(() => 0) }));
      const observeStep = (net, round, phase) => {
        if (phase >= 240) return;
        for (let j = 0; j < net.nSpikes; j++) {
          const i = net.spikes[j];
          if (phase < 40) targetGroups.forEach((group, k) => {
            if (group.includes(i)) measures[round].kc[k]++;
          });
          const k = mbonIndices.indexOf(i);
          if (k >= 0) measures[round].mbon[k]++;
        }
      };
      const trialRandomSeeds = rows.map(row => studentSeed * 1000000 + row.teacher * 10000 + row.test * 100 + row.sourceRound);
      const events = episode(evaluator, 0, cueValues, false, false, false, null, false, null,
        { ...setting, mappingByRound: rows.map(row => row.mapping),
          trialRandomSeeds, trialAgentRandomSeeds: trialRandomSeeds.map(x => x + 500000),
          observeStep, ...(connected ? { gradedMessages: messageRows.map(rates) } : {}) });
      if (evaluator.net.m.some((weight, i) => weight !== trainedWeights[i]))
        throw Error('Pathway evaluation changed frozen weights');
      if (['graded_population_choice', 'graded_population_drive', 'graded_population_ties', 'graded_population_silence503'].includes(mode)) measures.forEach((row, i) => {
        row.action = events[i].action;
        row.correct = events[i].correct;
        if (mode === 'graded_population_drive' || mode === 'graded_population_ties' || mode === 'graded_population_silence503') row.rates = events[i].rates;
      });
      return measures;
    };
    const evaluationRows = (teacher, trained) => {
      if (trained) return pool[teacher].filter(row => row.test === 2);
      const source = JSON.parse(readFileSync(new URL(`../data/two_cue_source_02_${teacher}_lanes02.json`, import.meta.url))).source;
      const indices = ids.map(id => source.sourceIds.indexOf(id));
      return [0, 1].flatMap(mapping => source.runs.find(run => !run.trained && run.mapping === mapping)
        .samples.filter(sample => sample.cue === 1 && sample.test === 2)
        .map(sample => ({ teacher, mapping, test: 2, sourceRound: sample.round,
          trace: indices.map(i => sample.trace[i]) })));
    };
    const fitRows = [501, 502].flatMap(teacher => record(evaluationRows(teacher, true)));
    const fitReadout = key => {
      const n = fitRows[0][key].length;
      const center = Array.from({ length: n }, (_, i) => fitRows.reduce((sum, row) => sum + row[key][i], 0) / fitRows.length);
      const sd = center.map((c, i) => Math.sqrt(fitRows.reduce((sum, row) => sum + (row[key][i] - c) ** 2, 0) / fitRows.length) || 1);
      const vector = row => row[key].map((value, i) => (value - center[i]) / sd[i]);
      const centroids = [0, 1].map(mapping => Array.from({ length: n }, (_, i) => {
        const group = fitRows.filter(row => row.mapping === mapping);
        return group.reduce((sum, row) => sum + vector(row)[i], 0) / group.length;
      }));
      return row => {
        const x = vector(row), distance = mapping => x.reduce((sum, value, i) => sum + (value - centroids[mapping][i]) ** 2, 0);
        return distance(0) <= distance(1) ? 0 : 1;
      };
    };
    const decoders = { kc: fitReadout('kc'), mbon: fitReadout('mbon') };
    const assess = rows => Object.fromEntries(Object.entries(decoders).map(([key, predict]) => [key,
      { score: rows.filter(row => predict(row) === row.mapping).length,
        byMapping: [0, 1].map(mapping => rows.filter(row => row.mapping === mapping && predict(row) === mapping).length) }]));
    const evaluation = Object.fromEntries((mode === 'graded_population_silence503' ? [503, 504] : [505, 506]).map(teacher => {
      const trainedRows = evaluationRows(teacher, true);
      const trained = record(trainedRows), untrained = record(evaluationRows(teacher, false));
      const disconnected = record(trainedRows, false);
      const shuffledLabels = mode === 'graded_population_silence503' ? trained.map(row => row.mapping)
        : JSON.parse(readFileSync(new URL('../data/two_cue_population_fresh.json', import.meta.url)))
          .evaluation[teacher].trials.map(row => row.shuffledLabel);
      if (trained.length !== 24 || trained.filter(row => row.mapping === 0).length !== 12)
        throw Error('Unbalanced fresh evaluation');
      return [teacher, { trained: assess(trained), untrained: assess(untrained),
        disconnected: assess(disconnected), shuffledLabels: assess(trained.map((row, i) =>
          ({ ...row, mapping: shuffledLabels[i] }))) }];
    }));
    pathway = { hypothesis: 'Mapping information is lost between target KCs and action MBONs',
      calibrationTeachers: [501, 502], evaluationTeachers: [505, 506], studentSeed,
      trainingEpisodes: 8, kcWindowMs: [0, 40], mbonWindowMs: [0, 240],
      mbonIndices, gate: 'KC >= 18/24 on both and MBON <= 14/24 on both supports bottleneck; MBON >= 18/24 on both rejects; otherwise inconclusive',
      evaluation };
    if (mode === 'graded_population_silence503') {
      const summarize = rows => ({ byMapping: [0, 1].map(mapping => rows.filter(row => row.mapping === mapping && row.rates.every(rate => rate === 0)).length),
        trials: rows.map(row => ({ mapping: row.mapping, sourceRound: row.sourceRound, rates: row.rates, action: row.action })) });
      pathway.silence = Object.fromEntries([503, 504].map(teacher => {
        const rows = evaluationRows(teacher, true);
        const shuffledMessages = [...rows.slice(6, 12), ...rows.slice(12, 18), ...rows.slice(0, 6), ...rows.slice(18, 24)];
        return [teacher, { correct: summarize(record(rows)), shuffled: summarize(record(rows, true, shuffledMessages)),
          disconnected: summarize(record(rows, false)) }];
      }));
      pathway.silenceGate = 'Correct B-minus-A zero ties >= 4/12 on each teacher; shuffled B-minus-A <= 2/12 on each';
    }
    if (['graded_population_choice', 'graded_population_drive', 'graded_population_ties'].includes(mode)) {
      const archivedPathway = JSON.parse(readFileSync(new URL('../data/two_cue_graded_population_pathway_02_203_lanes02.json', import.meta.url))).gradedPopulationDecode.pathway;
      for (const teacher of [505, 506])
        if (JSON.stringify(evaluation[teacher].trained) !== JSON.stringify(archivedPathway.evaluation[teacher].trained))
          throw Error('Fresh-teacher pathway readouts changed');
      pathway.choice = Object.fromEntries([505, 506].map(teacher => {
        const rows = evaluationRows(teacher, true);
        const shuffledMessages = [...rows.slice(6, 12), ...rows.slice(12, 18), ...rows.slice(0, 6), ...rows.slice(18, 24)];
        const summarize = measured => ({ score: measured.reduce((n, row) => n + row.correct, 0),
          byMapping: [0, 1].map(mapping => measured.filter(row => row.mapping === mapping && row.correct).length),
          trials: measured.map(row => ({ mapping: row.mapping, teacher: row.teacher, sourceRound: row.sourceRound,
            action: row.action, correct: row.correct, mbonPrediction: decoders.mbon(row) })) });
        return [teacher, { correct: summarize(record(rows)), shuffled: summarize(record(rows, true, shuffledMessages)),
          disconnected: summarize(record(rows, false)) }];
      }));
      pathway.choiceGate = 'MBON >= 18/24 and choices <= 14/24 on both teachers supports decision mismatch; choices >= 18/24 on both suggests cohort-specific failure; otherwise inconclusive';
      if (mode === 'graded_population_drive' || mode === 'graded_population_ties') {
        const archivedChoice = JSON.parse(readFileSync(new URL('../data/two_cue_graded_population_choice_02_203_lanes02.json', import.meta.url))).gradedPopulationDecode.pathway.choice;
        pathway.drive = Object.fromEntries([505, 506].map(teacher => {
          const rows = evaluationRows(teacher, true);
          const shuffledMessages = [...rows.slice(6, 12), ...rows.slice(12, 18), ...rows.slice(0, 6), ...rows.slice(18, 24)];
          const conditions = { correct: record(rows), shuffled: record(rows, true, shuffledMessages), disconnected: record(rows, false) };
          for (const name of Object.keys(conditions))
            if (conditions[name].some((row, i) => row.action !== archivedChoice[teacher][name].trials[i].action))
              throw Error('Frozen drive probe changed archived choices');
          if (mode === 'graded_population_ties') {
            const summarizeTies = values => ({
              byMapping: [0, 1].map(mapping => ({
                errors: values.filter(row => row.mapping === mapping && !row.correct).length,
                errorTies: values.filter(row => row.mapping === mapping && !row.correct && row.rates.every(rate => rate === 0)).length,
                correct: values.filter(row => row.mapping === mapping && row.correct).length,
                correctTies: values.filter(row => row.mapping === mapping && row.correct && row.rates.every(rate => rate === 0)).length })),
              action1Intrusions: values.filter(row => row.mapping === 1 && row.action === 1).length,
              action1Ties: values.filter(row => row.mapping === 1 && row.action === 1 && row.rates.every(rate => rate === 0)).length,
              trials: values.map(row => ({ mapping: row.mapping, sourceRound: row.sourceRound,
                action: row.action, correct: row.correct, rates: row.rates })) });
            return [teacher, Object.fromEntries(Object.entries(conditions).map(([name, values]) =>
              [name, summarizeTies(values)]))];
          }
          const selected = archivedChoice[teacher].correct.trials
            .filter(row => row.mapping === 1 && row.action === 1).map(row => row.sourceRound);
          return [teacher, { selected: selected.map(sourceRound => {
            const i = rows.findIndex(row => row.mapping === 1 && row.sourceRound === sourceRound);
            if (i < 0) throw Error('Selected trial absent');
            const rates = Object.fromEntries(Object.entries(conditions).map(([name, values]) => [name, values[i].rates]));
            const margins = Object.fromEntries(Object.entries(rates).map(([name, values]) => [name, values[1] - values[2]]));
            return { sourceRound, rates, margins, correctMinusDisconnected: margins.correct - margins.disconnected };
          }) }];
        }));
        pathway.driveGate = mode === 'graded_population_ties'
          ? 'All 10 archived mapping-B action-1 intrusions are zero-rate ties, and mapping-B error tie fraction exceeds correct-choice tie fraction on each teacher'
          : 'On the eight archived teacher-505 mapping-B action-1 trials, correct-minus-disconnected action-1/action-2 margin > 0 on at least 6/8 and positive on average';
      }
    }
  }
  console.log('graded population', studentSeed, 'before/after', score(before), score(after.correct),
    'shuffled', score(after.shuffled), 'disconnected', score(after.disconnected));
  return { command: `node tools/two_cue_transfer.mjs ${mode} ${seed} 02 lanes02 150`,
    hypothesis: 'Fixed raw graded four-MBON activity supports cue-free mapping decoding',
    sourceNeurons: ids, calibrationTeachers: [501, 502], evaluationTeachers: [503, 504],
    studentSeed, trainingEpisodes: 8, setting, calibrationCeilings: ceilings,
    gate: 'correct >= 18/24, >= 8/12 each mapping, >= 4/24 above shuffled and disconnected',
    heldout, shuffledSource: shuffled.map(row => ({ teacher: row.teacher, mapping: row.mapping,
      test: row.test, sourceRound: row.sourceRound })), training, before, after,
    ...(pathway ? { pathway } : {}) };
}
const result = { schema: 1, seed,
  ...(['source', 'teach_live', 'teach_live_pretrain_long', 'teach_live_threshold', 'decode_mapping_signal', 'decode_mapping_signal_confirm', 'decode_mapping_signal_paired', 'decode_mapping_signal_replication', 'decode_mapping_signal_teacher', 'decode_mapping_signal_mechanism', 'decode_mapping_signal_counterfactual', 'diag_readout'].includes(mode)
    ? { command: `node tools/two_cue_transfer.mjs ${mode} ${seed} ${pair} ${cuePattern} ${cueHz}${['decode_mapping_signal_teacher', 'decode_mapping_signal_mechanism', 'decode_mapping_signal_counterfactual'].includes(mode) ? ` ${teacherSeed}` : ''}` } : {}),
  protocol: { roundsPerEpisode: ROUNDS, episodesPerMapping: EPISODES,
  mapping: 'correctAction = actions[cue XOR mapping]', actions, cueHz, cueVpnStart: WIDTH,
  cuePattern, balancedTrainingActions: mode === 'explore1' || mode === 'decode_bit2146_explore',
  channelVpnWidthReserved: WIDTH, cueStartMs: aloneBaseline || blockSpeedPilot || learningResetProbe || disconnectedControlProbe || matchedLearningProbe || ['interference_temporal', 'interference_retain', 'teach_live', 'teach_live_pretrain_long', 'teach_live_threshold', 'decode_mapping_signal', 'decode_mapping_signal_confirm', 'decode_mapping_signal_paired', 'decode_mapping_signal_replication', 'decode_mapping_signal_teacher', 'decode_mapping_signal_mechanism', 'decode_mapping_signal_counterfactual', 'single_cue_transfer', 'single_cue_baseline_probe', 'single_cue_initial_drive', 'single_cue_round_reset', 'single_cue_preceding_bit', 'single_cue_order_probe', 'single_cue_order_reset', 'single_cue_order_rng', 'single_cue_order_rng_full', 'single_cue_order_rng_reset', 'single_cue_order_conductance', 'single_cue_order_voltage', 'single_cue_order_refractory', 'diag_readout', 'decode_two_bit_active0_conditional', 'decode_two_bit_active0_probe', 'decode_two_bit_cue_gated_pair', 'decode_two_bit_cue_gated_probe', 'decode_bit2146', 'decode_bit2134', 'decode_bit_pathway', 'decode_bit2146_explore', 'decode_bit2146_active0'].includes(mode) ? 70 : 0,
  cueDurationMs: aloneBaseline || blockSpeedPilot || learningResetProbe || disconnectedControlProbe || matchedLearningProbe || ['interference_temporal', 'interference_retain', 'teach_live', 'teach_live_pretrain_long', 'teach_live_threshold', 'decode_mapping_signal', 'decode_mapping_signal_confirm', 'decode_mapping_signal_paired', 'decode_mapping_signal_replication', 'decode_mapping_signal_teacher', 'decode_mapping_signal_mechanism', 'decode_mapping_signal_counterfactual', 'single_cue_transfer', 'single_cue_baseline_probe', 'single_cue_initial_drive', 'single_cue_round_reset', 'single_cue_preceding_bit', 'single_cue_order_probe', 'single_cue_order_reset', 'single_cue_order_rng', 'single_cue_order_rng_full', 'single_cue_order_rng_reset', 'single_cue_order_conductance', 'single_cue_order_voltage', 'single_cue_order_refractory', 'diag_readout', 'decode_two_bit_active0_conditional', 'decode_two_bit_active0_probe', 'decode_two_bit_cue_gated_pair', 'decode_two_bit_cue_gated_probe', 'decode_bit2146', 'decode_bit2134', 'decode_bit_pathway', 'decode_bit2146_explore', 'decode_bit2146_active0'].includes(mode) ? 170 : 240,
  decisionMs: 240, dopamineEndMs: 340 },
  ...(blockSpeedPilot ? { twoCueBlockSpeedPilot: twoCueBlockSpeedPilot() }
    : aloneBaseline ? { twoCueAloneBaseline: twoCueAloneBaseline() }
    : mode === 'source' ? { source: sourceRun() } : mode === 'teach' ? { teaching: teachRun() }
    : ['teach_live', 'teach_live_pretrain_long', 'teach_live_threshold'].includes(mode) ? { liveTeaching: teachLiveRun() }
    : ['decode_mapping_signal', 'decode_mapping_signal_confirm', 'decode_mapping_signal_paired', 'decode_mapping_signal_replication', 'decode_mapping_signal_teacher', 'decode_mapping_signal_mechanism', 'decode_mapping_signal_counterfactual'].includes(mode) ? { mappingSignalDiagnostic: decodeMappingSignal() }
    : mode === 'single_cue_transfer' ? { singleCueTransfer: singleCueTransfer() }
    : freshLearningProbe ? { singleCueLearningFresh: singleCueTransfer() }
    : prospectiveLearningProbe ? { singleCueLearningProspective: singleCueTransfer() }
    : lockedLearningProbe ? { singleCueLearningLocked: singleCueTransfer() }
    : learningResetProbe ? { singleCueLearningReset: singleCueTransfer() }
    : disconnectedControlProbe ? { singleCueDisconnectedControl: singleCueTransfer() }
    : gradedPopulationProbe ? { gradedPopulationDecode: gradedPopulationDecode() }
    : mode === 'single_cue_baseline_probe' ? { singleCueBaselineProbe: singleCueTransfer(true) }
    : mode === 'single_cue_initial_drive' ? { singleCueInitialDrive: singleCueTransfer() }
    : mode === 'single_cue_round_reset' ? { singleCueRoundReset: singleCueTransfer() }
    : mode === 'single_cue_preceding_bit' ? { singleCuePrecedingBit: singleCueTransfer() }
    : mode === 'single_cue_order_probe' ? { singleCueOrderProbe: singleCueTransfer() }
    : mode === 'single_cue_order_reset' ? { singleCueOrderReset: singleCueTransfer() }
    : mode === 'single_cue_order_rng' ? { singleCueOrderRng: singleCueTransfer() }
    : mode === 'single_cue_order_rng_full' ? { singleCueOrderRngFull: singleCueTransfer() }
    : mode === 'single_cue_order_rng_reset' ? { singleCueOrderRngReset: singleCueTransfer() }
    : mode === 'single_cue_order_conductance' ? { singleCueOrderConductance: singleCueTransfer() }
    : voltageProbe ? { singleCueOrderVoltage: singleCueTransfer() }
    : refractoryProbe ? { singleCueOrderRefractory: singleCueTransfer() }
    : mode === 'diag_readout' ? { readoutDiagnostic: readoutDiagnostic() }
    : mode === 'encoding_channel_probe' ? { encodingChannelProbe: encodingChannelProbe() }
    : ['encoding_channel_natural', 'encoding_channel_active0_cue'].includes(mode)
      ? { encodingChannelNatural: encodingChannelNatural() }
    : ['decode_two_bit', 'decode_two_bit_active0_conditional', 'decode_bit2146', 'decode_bit2134'].includes(mode)
      ? { twoBitDecoding: decodeTwoBit() }
    : mode === 'decode_bit_pathway' ? { bitPathway: {
      bit2146: decodeTwoBit(0, true), bit2134: decodeTwoBit(1, true) } }
    : mode === 'decode_bit2146_explore' ? { twoBitDecoding: decodeTwoBit(0, true, true) }
    : mode === 'decode_bit2146_active0' ? { twoBitDecoding: decodeTwoBit(0, true) }
    : mode === 'decode_two_bit_active0_probe' ? { twoBitDecoding: decodeTwoBit(2, true) }
    : mode === 'decode_two_bit_cue_gated_pair' ? { channelPair: {
      original: decodeTwoBit(2), cueGated: decodeTwoBit(2, false, false, true) } }
    : mode === 'decode_two_bit_cue_gated_probe' ? { channelPair: {
      original: decodeTwoBit(2, true), cueGated: decodeTwoBit(2, true, false, true) } }
    : mode === 'receiver_frozen' ? { receiverFrozen: decodeTwoBit() }
    : ['interference', 'interference_late', 'interference_kc', 'interference_temporal', 'interference_retain'].includes(mode)
      ? { interference: interferenceRun() }
    : mode === 'probe' ? { untrained: [0, 1].map(mapping => ({ mapping,
    events: episode(fly(seed), mapping, cues(seed * 1009 + mapping), false, true) })) }
    : { flyA: run(seed), ...(mode === 'baseline' ? { flyB: run(seed + 500) } : {}) }) };
writeFileSync(output, JSON.stringify(result));
console.log(`saved ${output.pathname}`);
