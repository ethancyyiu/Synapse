// One locked frozen matched forced-bit/no-message RPS receiver diagnostic.
// node tools/check_rps_hidden_strategy_frozen_forced_bits.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { Network, rng, wiring } from '../js/sim.js';
import { RpsContextualBandit, result } from '../js/rps.js';

const output = new URL('../data/rps_hidden_strategy_frozen_forced_bits.json', import.meta.url);
if (existsSync(output)) throw Error('Refusing to overwrite frozen forced-bit result');
const read = path => JSON.parse(readFileSync(new URL(path, import.meta.url)));
const pilot = read('../data/rps_hidden_strategy_teaching_pilot.json');
const sourceArchive = read('../data/rps_hidden_strategy_continuous_source.json');
const teacherA = read('../data/rps_independent_teachers.json');
const teacherB = read('../data/rps_strategy_b_replication.json');
const cueCalibration = read('../data/rps_readout_information.json');
const sub = read('../data/subgraph.json');
const targets = [3141, 3165], dnScale = 7.388888888888889;
const seeds = [821, 822, 823, 824], teachers = { A: 783, B: 793 };
const sequenceSeeds = [52000, 52003, 52004], conditions = ['none', 'bit0', 'bit1'];
const { featureMean, featureStd } = pilot.design;
const command = 'node tools/check_rps_hidden_strategy_frozen_forced_bits.mjs';
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const mean = xs => xs.reduce((s, x) => s + x, 0) / xs.length;
const approx = (a, b) => Math.abs(a - b) < 1e-10;
const controller = agent => ({ W: agent.W.map(w => [...w]), b: [...agent.b], baseline: agent.baseline });
const cueGuess = rates => {
  const x = rates.map(Math.log1p);
  const d = cueCalibration.centroids.map(c =>
    c.reduce((z, v, j) => z + (x[j] - v) ** 2, 0));
  return d.indexOf(Math.min(...d));
};
const margin = scores => {
  const sorted = [...scores].sort((a, b) => b - a);
  return sorted[0] - sorted[1];
};
const dnSpread = (saved, counts) => {
  const q = saved.W.map(w => w[3] * counts[0] / dnScale + w[4] * counts[1] / dnScale);
  return Math.max(...q) - Math.min(...q);
};
const score = (saved, mbon, counts) => {
  const f = [...mbon, ...counts.map(x => x / dnScale)];
  const scores = saved.W.map((w, a) => w.reduce((z, v, j) => z + v * f[j], saved.b[a]));
  return { features: f, counts: [...counts], scores, action: scores.indexOf(Math.max(...scores)),
    margin: margin(scores), dnScoreSpread: dnSpread(saved, counts) };
};
const compared = (rows, branch, reference) => {
  const switched = rows.filter(r => r[branch].action !== r[reference].action);
  const wins = mode => rows.filter(r => r[mode].action === r.winningAction).length;
  return { changes: switched.length,
    helpful: switched.filter(r => r[branch].action === r.winningAction &&
      r[reference].action !== r.winningAction).length,
    harmful: switched.filter(r => r[branch].action !== r.winningAction &&
      r[reference].action === r.winningAction).length,
    neutral: switched.filter(r => r[branch].action !== r.winningAction &&
      r[reference].action !== r.winningAction).length,
    branchWins: wins(branch), referenceWins: wins(reference) };
};
const summarize = rows => {
  const modes = ['none', 'bit0', 'bit1', 'teacher', 'opposite'];
  return { trials: rows.length,
    byCondition: Object.fromEntries(modes.map(k => [k, {
      wins: rows.filter(r => r[k].action === r.winningAction).length,
      actionCounts: [0, 1, 2].map(a => rows.filter(r => r[k].action === a).length),
      meanMargin: mean(rows.map(r => r[k].margin)),
      meanDnScoreSpread: mean(rows.map(r => r[k].dnScoreSpread)) }])),
    comparisons: { bit0VsNone: compared(rows, 'bit0', 'none'),
      bit1VsNone: compared(rows, 'bit1', 'none'),
      bit1VsBit0: compared(rows, 'bit1', 'bit0'),
      teacherVsNone: compared(rows, 'teacher', 'none'),
      teacherVsOpposite: compared(rows, 'teacher', 'opposite') } };
};

const design = { students: seeds, teachers, source: 'archived continuous-state 680 ms teacher episodes',
  sequenceSeeds, episodeKeys: [100, 101, 102], cues: [0, 1, 2],
  observationsPerCue: 23, conditions, probesPerStudent: 207, totalProbes: 828,
  studentKey: 'seed*10000000 + episodeKey*1000 + round; network rng(key+2), cue rng(key+3), controller rng(key+501)',
  studentReset: 'resetState before every probe; same reset and key for all three conditions',
  schedule: 'student blank 0-240; DN 3141/3165 300 Hz 240-280; VPN cue 150 Hz 280-520; ordinary counts reset 330; frozen decision 520; no-learning call 620; end 680',
  dnScale, featureMean, featureStd,
  technicalGates: 'exact triplets and weights; source expected bit >=18/23 each cue; stimulated target +1 mean spike and >other count >=22/23 each cue/bit; fixed-decoder cue recognition >=19/23 for no/bit0/bit1 and <=2/23 drop versus no',
  sensitivityGates: 'each student: bit0 versus bit1 >=7/69 action changes AND actual teacher-selected bit versus no-message >=4/69 changes, holding no-message MBON fixed',
  usefulnessGates: 'each student: teacher-selected branch >=4/69 more wins than both no-message and opposite bit, with helpful > harmful action changes versus each; all four students must pass' };
if (!pilot.allValid || pilot.passed || !sourceArchive.passed || !cueCalibration.passed ||
    !same(pilot.design.students.map(x => x.seed), seeds) ||
    !same(pilot.design.teachers, teachers) ||
    !same(pilot.design.evalSeeds, sequenceSeeds) ||
    !same(pilot.design.targets, targets) || pilot.design.dnScale !== dnScale ||
    targets.some(i => sub.neurons.role[i] !== 'dn' || sub.neurons.type[i] !== 'DNge175'))
  throw Error('Archived pilot, teacher source, or receiver interface changed');

const teacherRecords = { A: teacherA.teachers.find(t => t.seed === teachers.A),
  B: teacherB.teachers.find(t => t.seed === teachers.B) };
const sourceByStrategy = {};
for (const strategy of ['A', 'B']) {
  const savedTeacher = teacherRecords[strategy];
  if (!savedTeacher || savedTeacher.mapping !== strategy ||
      !savedTeacher.frozen.controllerUnchanged || !savedTeacher.frozen.synapsesUnchanged)
    throw Error(`Archived ${strategy} teacher cannot be reconstructed`);
  const fromPilot = pilot.sourceRows[strategy].evaluation;
  const older = sourceArchive.teachers.find(t => t.seed === teachers[strategy]).rows;
  if (fromPilot.length !== 3 || fromPilot.some(xs => xs.length !== 24))
    throw Error('Incomplete teacher episode');
  sourceByStrategy[strategy] = fromPilot.map((episode, i) => episode.map(row => {
    if (row.cue === null) return { ...row, sequenceSeed: sequenceSeeds[i] };
    const mb = row.rates.map((rate, j) =>
      (Math.log1p(rate) - featureMean[j]) / featureStd[j]);
    const q = savedTeacher.final.W.map((w, a) =>
      w.reduce((z, v, j) => z + v * mb[j], savedTeacher.final.b[a]));
    const bit = Number(q[row.cue] > Math.max(...q.filter((_, a) => a !== row.cue)));
    const archived = older.find(x => x.sequenceSeed === sequenceSeeds[i] && x.round === row.round);
    if (!archived || archived.cue !== row.cue || archived.bit !== row.bit ||
        archived.bit !== bit || !q.every((v, a) => approx(v, row.scores[a])) ||
        !q.every((v, a) => approx(v, archived.scores[a])))
      throw Error(`Teacher source mismatch: ${strategy} ${i}/${row.round}`);
    return { ...row, sequenceSeed: sequenceSeeds[i] };
  }));
}

function probe(net, agent, saved, seed, cue, episodeIndex, round, condition) {
  const key = seed * 10000000 + (100 + episodeIndex) * 1000 + round;
  const vpn = Array.from(net.laneVpn[cue].slice(2));
  const target = condition === 'none' ? null : targets[condition === 'bit0' ? 0 : 1];
  if (vpn.some(i => targets.includes(i))) throw Error('VPN/DN overlap');
  net.resetState(); net.rand = rng(key + 2); agent.rand = rng(key + 501);
  agent.explore = false;
  const counts = [0, 0];
  let at280, at520, row, noLearningCall = false;
  for (let t = 0; t < 680; t++) {
    if (t === 240 && target !== null) net.setDrive([target], 300);
    if (t === 280) {
      at280 = [...counts];
      if (target !== null) net.setDrive([target], 0);
      net.rand = rng(key + 3);
      net.setDrive(vpn, 150);
    }
    if (t === 330) net.resetCounts();
    if (t === 520) {
      const cueActive = vpn.every(i => net.drive[i] === 150);
      net.setDrive(vpn, 0);
      at520 = [...counts];
      agent.setDnWindowCounts(at520);
      agent.decide(net);
      row = { episodeIndex, sequenceSeed: sequenceSeeds[episodeIndex], round, cue,
        condition, key, target, dnCounts: at520, decisionNetDnCounts: targets.map(i => net.count[i]),
        rates: [...agent.rates], features: [...agent.f], scores: [...agent.scores],
        action: agent.action, margin: margin(agent.scores),
        dnScoreSpread: dnSpread(saved, at520), cueGuess: cueGuess(agent.rates),
        cueActive };
      net.resetCounts();
    }
    if (t === 620) { agent.afterDopamine(net, false); noLearningCall = true; }
    net.step();
    if (t >= 240 && t < 280) for (let k = 0; k < net.nSpikes; k++) {
      if (net.spikes[k] === targets[0]) counts[0]++;
      else if (net.spikes[k] === targets[1]) counts[1]++;
    }
  }
  if (!row || !noLearningCall || !row.cueActive || !same(at280, at520) ||
      !row.features.every(Number.isFinite) || row.features.length !== 5 ||
      !row.features.every((x, j) => approx(x, j < 3 ?
        (Math.log1p(row.rates[j]) - featureMean[j]) / featureStd[j] :
        row.dnCounts[j - 3] / dnScale)) ||
      row.action !== row.scores.indexOf(Math.max(...row.scores)) ||
      !same(controller(agent), saved) || net.m.some(x => x !== 1))
    throw Error(`Frozen probe validity failed: ${seed}/${episodeIndex}/${round}/${condition}`);
  return row;
}

const students = [];
for (const seed of seeds) {
  const savedStudent = pilot.studentResults.find(s => s.seed === seed);
  const strategy = savedStudent.strategy;
  const run = savedStudent.runs.find(r => r.arm === 'correct');
  if (!run?.synapsesUnchanged || run.final.W.length !== 3 ||
      run.final.W.some(w => w.length !== 5) ||
      !run.initial.W.flat().every(x => x === 0))
    throw Error(`Missing exact final student controller ${seed}`);
  const net = new Network(sub, wiring(sub, 'real', seed), { seed });
  const agent = new RpsContextualBandit(net, { seed, featureMean, featureStd,
    dnCountScale: dnScale, readoutLearn: false });
  agent.W.forEach((w, a) => w.set(run.final.W[a]));
  agent.b.set(run.final.b); agent.baseline = run.final.baseline;
  agent.explore = false;
  const saved = controller(agent);
  if (!same(saved, run.final) || net.m.some(x => x !== 1))
    throw Error(`Student ${seed} reconstruction mismatch`);
  const triplets = [];
  for (let episodeIndex = 0; episodeIndex < 3; episodeIndex++)
    for (let round = 1; round < 24; round++) {
      const source = sourceByStrategy[strategy][episodeIndex][round];
      const { cue, opponent } = source;
      if (![0, 1, 2].includes(cue)) throw Error('Missing observed cue');
      const physical = Object.fromEntries(conditions.map(condition =>
        [condition, probe(net, agent, saved, seed, cue, episodeIndex, round, condition)]));
      const noMbon = physical.none.features.slice(0, 3);
      const analytical = Object.fromEntries(conditions.map(condition =>
        [condition, score(saved, noMbon, physical[condition].dnCounts)]));
      for (const condition of conditions) if (condition === 'none' &&
        (!analytical.none.scores.every((v, a) => approx(v, physical.none.scores[a])) ||
          analytical.none.action !== physical.none.action))
        throw Error('Analytical no-message score does not reproduce physical branch');
      const teacherCondition = `bit${source.bit}`;
      const oppositeCondition = `bit${1 - source.bit}`;
      const winningAction = (opponent + 1) % 3;
      if (result(winningAction, opponent) !== 1) throw Error('Invalid offline winner');
      triplets.push({ episodeIndex, sequenceSeed: sequenceSeeds[episodeIndex], round,
        cue, opponent, winningAction, teacherBit: source.bit,
        physical, analytical: { ...analytical,
          teacher: analytical[teacherCondition], opposite: analytical[oppositeCondition] } });
    }
  const rowsFor = type => triplets.map(t => ({ cue: t.cue, winningAction: t.winningAction,
    ...t[type], teacher: t[type][`bit${t.teacherBit}`],
    opposite: t[type][`bit${1 - t.teacherBit}`] }));
  const physicalRows = rowsFor('physical'), analyticalRows = rowsFor('analytical');
  const byCue = [0, 1, 2].map(cue => {
    const ys = triplets.filter(t => t.cue === cue);
    const condition = Object.fromEntries(conditions.map(c => [c, {
      recognized: ys.filter(t => t.physical[c].cueGuess === cue).length,
      meanDnCounts: [0, 1].map(j => mean(ys.map(t => t.physical[c].dnCounts[j]))),
      meanRates: [0, 1, 2].map(j => mean(ys.map(t => t.physical[c].rates[j]))) }]));
    const physicalReach = [0, 1].map(bit =>
      condition[`bit${bit}`].meanDnCounts[bit] - condition.none.meanDnCounts[bit]);
    const separation = [0, 1].map(bit => ys.filter(t =>
      t.physical[`bit${bit}`].dnCounts[bit] >
      t.physical[`bit${bit}`].dnCounts[1 - bit]).length);
    const sourceCorrect = ys.filter(t => t.teacherBit === (strategy === 'A' ? 0 : 1)).length;
    return { cue, trials: ys.length, condition, physicalReach, separation, sourceCorrect };
  });
  const technical = { triplets: triplets.length === 69 && byCue.every(x => x.trials === 23),
    frozenWeights: same(controller(agent), saved) && net.m.every(x => x === 1),
    source: byCue.every(x => x.sourceCorrect >= 18),
    physicalReach: byCue.every(x => x.physicalReach.every(v => v >= 1) &&
      x.separation.every(v => v >= 22)),
    cuePreservation: byCue.every(x => conditions.every(c =>
      x.condition[c].recognized >= 19) &&
      ['bit0', 'bit1'].every(c => x.condition.none.recognized -
        x.condition[c].recognized <= 2)) };
  const physicalSummary = summarize(physicalRows), analyticalSummary = summarize(analyticalRows);
  const c = analyticalSummary.comparisons;
  const sensitivity = c.bit1VsBit0.changes >= 7 && c.teacherVsNone.changes >= 4;
  const useful = c.teacherVsNone.branchWins - c.teacherVsNone.referenceWins >= 4 &&
    c.teacherVsOpposite.branchWins - c.teacherVsOpposite.referenceWins >= 4 &&
    c.teacherVsNone.helpful > c.teacherVsNone.harmful &&
    c.teacherVsOpposite.helpful > c.teacherVsOpposite.harmful;
  students.push({ seed, strategy, teacherSeed: teachers[strategy], finalController: saved,
    byCue, technical, physicalSummary, analyticalSummary,
    gates: { sensitivity, useful }, passed: Object.values(technical).every(Boolean) &&
      sensitivity && useful, triplets });
  console.log(`completed frozen student ${seed}`);
}
const gates = { technical: students.every(s => Object.values(s.technical).every(Boolean)),
  sensitivity: students.every(s => s.gates.sensitivity),
  useful: students.every(s => s.gates.useful) };
const data = { command, design, status: 'complete', previousTeachingPilotPassed: pilot.passed,
  teacherRowsReused: true, students, gates, passed: Object.values(gates).every(Boolean),
  limitation: 'Student state resets before each new matched probe. This cannot reproduce the continuous student history of the prior teaching pilot; no reward or learning was delivered.' };
writeFileSync(output, `${JSON.stringify(data)}\n`);
console.log(JSON.stringify({ gates, passed: data.passed,
  students: students.map(s => ({ seed: s.seed, technical: s.technical, gates: s.gates,
    byCue: s.byCue.map(c => ({ cue: c.cue, recognition: conditions.map(k =>
      c.condition[k].recognized), reach: c.physicalReach, separation: c.separation })),
    physical: s.physicalSummary.comparisons,
    analytical: s.analyticalSummary.comparisons,
    meanTeacherDnSpread: s.analyticalSummary.byCondition.teacher.meanDnScoreSpread })) }, null, 2));
