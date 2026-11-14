// Prospectively locked, isolated synthetic pickup-recovery study.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { access, open, readFile } from 'node:fs/promises';
import { Network, rng, wiring } from '../js/sim.js';
import { RpsContextualBandit } from '../js/rps.js';
import { ChallengeCueHeadBandit, studentWeights } from '../js/rps_challenge_students.js';

const stage = process.argv[2];
if (!['initial', 'confirmation'].includes(stage)) throw Error('Use initial or confirmation');
const root = new URL('../', import.meta.url);
const read = async p => JSON.parse(await readFile(new URL(p, root)));
const output = new URL(`data/package_recovery_${stage}.json`, root);
try { await access(output); throw Error('Write-once result already exists'); }
catch (e) { if (e.code !== 'ENOENT') throw e; }
if (stage === 'confirmation' && (await read('data/package_recovery_initial.json')).status !== 'passed')
  throw Error('Initial recovery study did not pass');
const protocolSha256 = createHash('sha256').update(await readFile(new URL('docs/package-recovery-protocol.md', root))).digest('hex');
const sub = await read('data/subgraph.json');
const { featureMean, featureStd } = (await read('data/rps_contextual_bandit.json')).design;
const c = stage === 'initial' ? { teacher: 14001, student: 14002, train: 2026110811,
  eval: 2026110812, generic: 2026110815 } : { teacher: 14003, student: 14004,
  train: 2026110813, eval: 2026110814, generic: 2026110816 };
const cueProb = [[.70,.20,.10],[.20,.70,.10],[.10,.10,.80],[.20,.60,.20]];
const successProb = [[.78,.35,.20],[.32,.78,.22],[.12,.15,.82],[.36,.62,.30]];
const delays = [2,4,8], DN = [3141,3165], codes = [[1,0],[0,1],[1,1]];
const checkpoints = [0,6,12,24,48,144];
const arms = ['trained','alone','untrained','shuffled','generic'];
const artifact = { stage, protocolSha256, config: c, status: 'started', task: null,
  sensory: null, teacher: null, source: null, receiver: null, independent: null, comparison: null };
async function finish(status) {
  artifact.status = status;
  const file = await open(output, 'wx');
  try { await file.writeFile(JSON.stringify(artifact, null, 2) + '\n'); }
  finally { await file.close(); }
  console.log(JSON.stringify({ status, task: artifact.task?.passed,
    sensory: artifact.sensory?.summary, teacher: artifact.teacher?.summary,
    source: artifact.source?.summary, receiver: artifact.receiver?.summary,
    independent: artifact.independent?.summary, comparison: artifact.comparison?.summary }));
}
const expected = (mode, action) => successProb[mode][action] * 1.4 - .4 - .03*delays[action] - (action===2?.20:0);
const cueExpected = [0,1,2].map(cue => [0,1,2].map(action =>
  cueProb.reduce((sum, row, mode) => sum + row[cue]*expected(mode, action), 0) /
  cueProb.reduce((sum,row) => sum+row[cue],0)));
const best = cueExpected.map(x => x.indexOf(Math.max(...x)));
function makeCases(seed, perMode) {
  const random = rng(seed), rows = [];
  for (let mode=0;mode<4;mode++) for (let j=0;j<perMode;j++) {
    const u=random(); let cue=0, cumulative=0;
    while (cue<2 && u >= (cumulative += cueProb[mode][cue])) cue++;
    const outcomes = successProb[mode].map(p => Number(random()<p));
    rows.push({ id: `${seed}-${mode}-${j}`, mode, cue, outcomes });
  }
  for (let i=rows.length-1;i>0;i--) { const j=Math.floor(random()*(i+1)); [rows[i],rows[j]]=[rows[j],rows[i]]; }
  return rows;
}
const trainingCases = makeCases(c.train,36), evaluationCases = makeCases(c.eval,9);
artifact.task = { cueProb, successProb, delays, rewardRule: 'success?1:-0.4, minus 0.03*delay and 0.20 for action 2',
  cueExpected, best, trainingCases, evaluationCases,
  passed: cueProb.every(row => row.every(p=>p>0) && Math.abs(row.reduce((a,b)=>a+b)-1)<1e-12) &&
    successProb.every(row=>row.length===3 && row.every(p=>p>0 && p<1)) &&
    best.join() === '0,1,2' && cueExpected.every(q => {
      const s=[...q].sort((a,b)=>b-a); return s[0]-s[1]>=.03;
    }) && trainingCases.length===144 && evaluationCases.length===36 &&
    new Set([...trainingCases,...evaluationCases].map(x=>x.id)).size===180 };
if (!artifact.task.passed) { await finish('stopped_task'); process.exit(0); }
const makeNet = seed => new Network(sub,wiring(sub,'real',seed),{seed});
const teacherWeights = a => JSON.stringify({W:a.W.map(w=>[...w]),b:[...a.b],baseline:a.baseline});
function teacher(frozen=false, source=null) {
  const net=makeNet(c.teacher), agent=new RpsContextualBandit(net,{seed:c.teacher,featureMean,featureStd,readoutLearn:!frozen});
  if (source) { source.agent.W.forEach((w,i)=>agent.W[i].set(w)); agent.b.set(source.agent.b); agent.baseline=source.agent.baseline; }
  agent.explore=!frozen; return {net,agent};
}
class PreferenceStudent extends ChallengeCueHeadBandit {
  imitate(decoded,enabled) {
    if (!enabled || decoded<0) return 0;
    assert.ok([0,1,2].includes(decoded));
    const h=this.heads[this.activeHead], q=this.logits(this.f), mx=Math.max(...q);
    const ex=q.map(x=>Math.exp(x-mx)), z=ex.reduce((a,b)=>a+b,0);
    for (let a=0;a<3;a++) { const g=.03*(Number(a===decoded)-ex[a]/z);
      h.b[a]+=g; for (let j=0;j<3;j++) h.W[a][j]+=g*this.f[j]; }
    return 1;
  }
}
function student(frozen=false,source=null) {
  const net=makeNet(c.student), agent=new PreferenceStudent(net,c.student,featureMean,featureStd);
  if (source) { source.agent.heads.forEach((h,i)=>{
    h.W.forEach((w,a)=>agent.heads[i].W[a].set(w)); agent.heads[i].b.set(h.b); });
    agent.baseline=source.agent.baseline; }
  agent.explore=!frozen; agent.readoutLearn=!frozen; return {net,agent};
}
function probe(net,cue,key) {
  net.resetState(); net.rand=rng(key+2); const vpn=net.laneVpn[cue].slice(2);
  net.setDrive(vpn,150);
  for (let t=0;t<240;t++) { if (t===50) net.resetCounts(); net.step(); }
  net.setDrive(vpn,0);
  const lanes=net.laneVpn.map(ids=>Array.from(ids.slice(2)).reduce((s,id)=>s+net.count[id],0));
  const rates=net.groupMbon.map(ids=>ids.reduce((s,id)=>s+net.count[id],0)/ids.length);
  const f=rates.map((x,j)=>(Math.log1p(x)-featureMean[j])/featureStd[j]);
  const mx=Math.max(...lanes), routed=mx>0 && lanes.filter(x=>x===mx).length===1?lanes.indexOf(mx):-1;
  return {cue,key,lanes,routed,rates,f};
}
function sensory() {
  const net=makeNet(c.student), calibration=[], heldout=[];
  for (let cue=0;cue<3;cue++) for (let repeat=0;repeat<12;repeat++) {
    const row={...probe(net,cue,c.student*10000000+500000+cue*100+repeat),repeat};
    (repeat<6?calibration:heldout).push(row);
  }
  const centroids=[0,1,2].map(cue=>[0,1,2].map(j=>calibration.filter(x=>x.cue===cue)
    .reduce((s,x)=>s+x.f[j],0)/6));
  const rows=heldout.map(row=>{ const d=centroids.map(v=>v.reduce((s,x,j)=>s+(x-row.f[j])**2,0));
    return {...row,distances:d,predicted:d.indexOf(Math.min(...d))}; });
  const byCue=[0,1,2].map(cue=>({cue,routed:rows.filter(x=>x.cue===cue && x.routed===cue).length,
    recognized:rows.filter(x=>x.cue===cue && x.predicted===cue).length}));
  const correct=byCue.reduce((s,x)=>s+x.recognized,0);
  const passed=correct>=14 && byCue.every(x=>x.routed===6 && x.recognized>=4) &&
    new Set(centroids.map(x=>JSON.stringify(x))).size===3 &&
    rows.every(x=>x.f.every(Number.isFinite)) && net.m.every(x=>x===1);
  return {calibration,heldout:rows,centroids,summary:{correct,byCue,passed,neuralWeightsUnchanged:true}};
}
artifact.sensory=sensory();
if (!artifact.sensory.summary.passed) { await finish('stopped_sensory'); process.exit(0); }

function outcome(spec,action) {
  const success=spec.outcomes[action], delay=delays[action], escalation=action===2?.20:0;
  return {success,failed:1-success,delay,escalation,
    reward:(success?1:-.4)-.03*delay-escalation,expectedReward:expected(spec.mode,action)};
}
function decode(counts) { return codes.findIndex(code=>code.every((v,i)=>v===Number(counts[i]>0))); }
function teacherRound(s,spec,index,learn, evaluation=false) {
  const {net,agent}=s, key=c.teacher*10000000+(evaluation?900000:0)+index*1000;
  net.resetState(); net.rand=rng(key+1); agent.rand=rng(key+501);
  const vpn=net.laneVpn[spec.cue].slice(2); net.setDrive(vpn,150);
  let row;
  for (let t=0;t<400;t++) {
    if (t===50) net.resetCounts();
    if (t===240) { net.setDrive(vpn,0); const action=agent.decide(net), result=outcome(spec,action);
      row={id:spec.id,mode:spec.mode,cue:spec.cue,index,action,rates:[...agent.rates],f:[...agent.f],
        scores:[...agent.scores],...result,updateStep:learn?agent.alpha*(result.reward-agent.scores[action]):0};
      agent.reward(net,result.reward); net.resetCounts(); }
    if (t===340) agent.afterDopamine(net,learn);
    net.step();
  }
  assert.ok(net.m.every(x=>x===1)); return row;
}
function studentRound(s,spec,index,target,learn,evaluation=false) {
  const {net,agent}=s, key=c.student*10000000+(evaluation?900000:0)+index*1000, counts=[0,0];
  net.resetState(); net.rand=rng(key+2); agent.rand=rng(key+501);
  const vpn=net.laneVpn[spec.cue].slice(2); net.setDrive(vpn,150);
  let row;
  for (let t=0;t<680;t++) {
    if (t===50) net.resetCounts();
    if (t===240) { net.setDrive(vpn,0); const action=agent.decide(net);
      row={id:spec.id,mode:spec.mode,cue:spec.cue,index,target,action,rates:[...agent.rates],f:[...agent.f],
        scores:[...agent.scores],routed:agent.activeHead,lanes:[...agent.laneCounts],
        counts:null,decoded:-1,updateStep:0,imitation:0};
      if (target!==null) DN.forEach((id,i)=>{if(codes[target][i]) net.setDrive([id],300);}); }
    if (t===280) { net.setDrive(DN,0); row.counts=[...counts]; row.decoded=decode(counts);
      const result=outcome(spec,row.action); Object.assign(row,result);
      agent.reward(net,result.reward); net.resetCounts(); }
    if (t===620) { row.updateStep=learn?agent.alpha*(row.reward-row.scores[row.action]):0;
      agent.afterDopamine(net,learn); row.imitation=agent.imitate(row.decoded,learn); }
    net.step();
    if (t>=240 && t<280) for (let i=0;i<net.nSpikes;i++) {
      if (net.spikes[i]===DN[0]) counts[0]++; else if (net.spikes[i]===DN[1]) counts[1]++; }
  }
  assert.equal(row.routed,spec.cue); assert.equal(row.decoded,target??-1);
  assert.ok(net.m.every(x=>x===1)); return row;
}
function summarize(rows) {
  return {n:rows.length,successes:rows.reduce((s,x)=>s+x.success,0),
    failures:rows.reduce((s,x)=>s+x.failed,0),delay:rows.reduce((s,x)=>s+x.delay,0),
    escalations:rows.reduce((s,x)=>s+Number(x.escalation>0),0),
    totalReward:rows.reduce((s,x)=>s+x.reward,0),
    meanExpectedReward:rows.reduce((s,x)=>s+x.expectedReward,0)/rows.length,
    actions:[0,1,2].map(a=>rows.filter(x=>x.action===a).length),
    byCue:[0,1,2].map(cue=>({cue,n:rows.filter(x=>x.cue===cue).length,
      successes:rows.filter(x=>x.cue===cue).reduce((s,x)=>s+x.success,0)}))};
}
function frozen(kind,source) {
  const s=kind==='teacher'?teacher(true,source):student(true,source);
  const snapshot=kind==='teacher'?teacherWeights(s.agent):studentWeights(s.agent);
  const rows=evaluationCases.map((spec,i)=>kind==='teacher'?teacherRound(s,spec,i,false,true):
    studentRound(s,spec,i,null,false,true));
  assert.equal(kind==='teacher'?teacherWeights(s.agent):studentWeights(s.agent),snapshot);
  assert.ok(s.net.m.every(x=>x===1));
  if(kind==='student') assert.ok(rows.every(x=>x.decoded===-1 && x.counts.every(v=>v===0) && x.imitation===0));
  return {...summarize(rows),rows,disconnected:true,explorationOff:true,weightsUnchanged:true};
}
const t=teacher(), teacherInitial=frozen('teacher',t), teacherBefore=teacherWeights(t.agent);
const teacherTraining=trainingCases.map((spec,i)=>teacherRound(t,spec,i,true));
const teacherFinal=frozen('teacher',t), teacherAfter=teacherWeights(t.agent);
const teacherPassed=teacherFinal.meanExpectedReward>=.20 &&
  teacherFinal.meanExpectedReward-teacherInitial.meanExpectedReward>=.10 &&
  teacherFinal.successes>=22 && teacherFinal.actions.every(n=>n>=4) &&
  teacherBefore!==teacherAfter && t.net.m.every(x=>x===1);
artifact.teacher={initial:teacherInitial,training:{...summarize(teacherTraining),rows:teacherTraining},
  final:teacherFinal,initialWeights:teacherBefore,finalWeights:teacherAfter,
  summary:{passed:teacherPassed,initialExpected:teacherInitial.meanExpectedReward,
    finalExpected:teacherFinal.meanExpectedReward,finalSuccesses:teacherFinal.successes,
    finalActions:teacherFinal.actions,controllerChanged:teacherBefore!==teacherAfter,neuralWeightsUnchanged:true}};
if (!teacherPassed) { await finish('stopped_teacher'); process.exit(0); }

function sourceTable(trained) {
  const s=teacher(true,trained?t:null), before=teacherWeights(s.agent), table=[];
  for(let i=0;i<144;i++) {
    const cue=trainingCases[i].cue, key=c.teacher*10000000+2000000+i*10+cue;
    probe(s.net,cue,key); s.agent.decide(s.net);
    table.push({index:i,cue,key,rates:[...s.agent.rates],scores:[...s.agent.scores],
      preference:s.agent.scores.indexOf(Math.max(...s.agent.scores))});
  }
  const probes=[];
  for(let cue=0;cue<3;cue++) for(let repeat=0;repeat<12;repeat++) {
    const key=c.teacher*10000000+3000000+cue*100+repeat;
    probe(s.net,cue,key); s.agent.decide(s.net);
    probes.push({cue,repeat,key,rates:[...s.agent.rates],scores:[...s.agent.scores],
      preference:s.agent.scores.indexOf(Math.max(...s.agent.scores))});
  }
  assert.equal(teacherWeights(s.agent),before); assert.ok(s.net.m.every(x=>x===1));
  return {table,probes};
}
const trainedSource=sourceTable(true), untrainedSource=sourceTable(false);
const sourceByCue=[0,1,2].map(cue=>({cue,correct:trainedSource.probes.filter(x=>x.cue===cue && x.preference===best[cue]).length}));
const sourceActions=[0,1,2].map(a=>trainedSource.table.filter(x=>x.preference===a).length);
const sourcePassed=sourceByCue.every(x=>x.correct>=9) && sourceActions.every(n=>n>0);
artifact.source={trainedSource,untrainedSource,summary:{passed:sourcePassed,byCue:sourceByCue,actions:sourceActions}};
if (!sourcePassed) { await finish('stopped_source'); process.exit(0); }

function receiverProbe(cue,repeat,message) {
  const s=student(true), before=studentWeights(s.agent), key=c.student*10000000+700000+cue*100+repeat;
  const net=s.net, agent=s.agent, counts=[0,0]; net.resetState(); net.rand=rng(key+2); agent.rand=rng(key+501);
  const vpn=net.laneVpn[cue].slice(2); net.setDrive(vpn,150); let pre;
  for(let t=0;t<680;t++) {
    if(t===50) net.resetCounts();
    if(t===240) { net.setDrive(vpn,0); agent.decide(net);
      pre={rates:[...agent.rates],f:[...agent.f],routed:agent.activeHead,lanes:[...agent.laneCounts]};
      if(message!==null) DN.forEach((id,i)=>{if(codes[message][i]) net.setDrive([id],300);}); }
    if(t===280) net.setDrive(DN,0);
    net.step();
    if(t>=240 && t<280) for(let i=0;i<net.nSpikes;i++) {
      if(net.spikes[i]===DN[0]) counts[0]++; else if(net.spikes[i]===DN[1]) counts[1]++; }
  }
  return {cue,repeat,message,key,...pre,counts,decoded:decode(counts),
    weightsUnchanged:before===studentWeights(agent) && net.m.every(x=>x===1)};
}
const receiverRows=[];
for(let cue=0;cue<3;cue++) for(let repeat=0;repeat<6;repeat++)
  for(const message of [null,0,1,2]) receiverRows.push(receiverProbe(cue,repeat,message));
const receiverConditions=[0,1,2].flatMap(cue=>{
  const none=receiverRows.filter(x=>x.cue===cue && x.message===null);
  return [null,0,1,2].map(message=>{
    const x=receiverRows.filter(r=>r.cue===cue && r.message===message);
    const decoded=x.filter(r=>r.decoded===(message??-1)).length;
    const routed=x.filter(r=>r.routed===cue).length;
    const intended=x.every(r=>message===null?r.counts.every(n=>n===0):
      codes[message].every((v,i)=>v?r.counts[i]>0:r.counts[i]===0));
    const paired=x.filter(r=>{const n=none.find(v=>v.repeat===r.repeat);
      return JSON.stringify([r.rates,r.f,r.routed])===JSON.stringify([n.rates,n.f,n.routed]);}).length;
    return {cue,message,decoded,routed,intended,paired,
      passed:x.length===6 && decoded===6 && routed===6 && intended && paired===6 && x.every(r=>r.weightsUnchanged)};
  });
});
artifact.receiver={rows:receiverRows,conditions:receiverConditions,
  summary:{passed:receiverConditions.every(x=>x.passed),conditions:receiverConditions}};
if (!artifact.receiver.summary.passed) { await finish('stopped_receiver'); process.exit(0); }

const synthetic=student(), syntheticAgent=synthetic.agent;
syntheticAgent.activeHead=0; syntheticAgent.W=syntheticAgent.heads[0].W;
syntheticAgent.b=syntheticAgent.heads[0].b; syntheticAgent.f=[1,0,0];
const decoded=decode(receiverRows.find(x=>x.message===0).counts);
const margin=()=>{const q=syntheticAgent.logits(syntheticAgent.f);
  return q[decoded]-Math.max(...q.filter((_,a)=>a!==decoded));};
const initialMargin=margin(), beforeSynthetic=studentWeights(syntheticAgent);
syntheticAgent.imitate(decoded,false);
const disabledUnchanged=beforeSynthetic===studentWeights(syntheticAgent);
syntheticAgent.imitate(decoded,true);
const syntheticPassed=disabledUnchanged && margin()>initialMargin;
artifact.synthetic={initialMargin,updatedMargin:margin(),disabledUnchanged,passed:syntheticPassed};
if (!syntheticPassed) { await finish('stopped_synthetic'); process.exit(0); }

function trainArm(name,targets=null) {
  const s=student(), initialWeights=studentWeights(s.agent), points=[{attempt:0,...frozen('student',s)}], rows=[];
  for(let i=0;i<144;i++) { rows.push(studentRound(s,trainingCases[i],i,targets?targets[i]:null,true));
    if(checkpoints.includes(i+1)) points.push({attempt:i+1,...frozen('student',s)}); }
  return {name,initialWeights,finalWeights:studentWeights(s.agent),training:{...summarize(rows),rows},
    checkpoints:points,neuralWeightsUnchanged:s.net.m.every(x=>x===1)};
}
const alone=trainArm('alone'), aloneFirst=alone.checkpoints[0], aloneFinal=alone.checkpoints.at(-1);
const independentPassed=aloneFinal.meanExpectedReward>=.20 &&
  aloneFinal.meanExpectedReward-aloneFirst.meanExpectedReward>=.10 &&
  aloneFinal.successes>=22 && aloneFinal.actions.every(n=>n>=4) && alone.neuralWeightsUnchanged;
artifact.independent={arm:alone,summary:{passed:independentPassed,initialExpected:aloneFirst.meanExpectedReward,
  finalExpected:aloneFinal.meanExpectedReward,finalSuccesses:aloneFinal.successes,finalActions:aloneFinal.actions}};
if (!independentPassed) { await finish('stopped_independent'); process.exit(0); }

const random=rng(c.generic), targets={
  trained:trainedSource.table.map(x=>x.preference),
  untrained:untrainedSource.table.map(x=>x.preference),
  shuffled:trainedSource.table.map((_,i)=>trainedSource.table[(i+37)%144].preference),
  generic:Array.from({length:144},()=>Math.floor(3*random())) };
function controlQuality(tape) {
  return {aligned:tape.filter((a,i)=>a===best[trainingCases[i].cue]).length,
    actions:[0,1,2].map(a=>tape.filter(x=>x===a).length),
    byCue:[0,1,2].map(cue=>({cue,n:trainingCases.filter(x=>x.cue===cue).length,
      aligned:tape.filter((a,i)=>trainingCases[i].cue===cue && a===best[cue]).length}))};
}
const controls={shuffled:controlQuality(targets.shuffled),generic:controlQuality(targets.generic)};
const controlPassed=Object.values(controls).every(x=>x.aligned<=72 && x.actions.every(n=>n>0));
artifact.comparison={controls,targets,byArm:{alone},summary:{controlPassed,passed:false}};
if (!controlPassed) { await finish('stopped_control_validity'); process.exit(0); }
for(const name of ['untrained','shuffled','generic','trained'])
  artifact.comparison.byArm[name]=trainArm(name,targets[name]);
const byArm=artifact.comparison.byArm;
assert.ok(arms.every(name=>byArm[name].initialWeights===alone.initialWeights &&
  byArm[name].training.n===144 && byArm[name].checkpoints.length===6 && byArm[name].neuralWeightsUnchanged));
assert.ok(arms.every(name=>byArm[name].checkpoints.every(x=>x.disconnected && x.explorationOff && x.weightsUnchanged)));
assert.ok(arms.filter(x=>x!=='alone').every(name=>byArm[name].training.rows.every(x=>x.decoded===x.target && x.imitation===1)));
const first20=Object.fromEntries(arms.map(name=>[name,
  byArm[name].checkpoints.find(x=>x.meanExpectedReward>=.20)?.attempt??null]));
const early=Object.fromEntries(arms.map(name=>[name,
  [6,12,24,48].reduce((s,n)=>s+byArm[name].checkpoints.find(x=>x.attempt===n).meanExpectedReward,0)/4]));
const final=Object.fromEntries(arms.map(name=>[name,{
  expected:byArm[name].checkpoints.at(-1).meanExpectedReward,
  successes:byArm[name].checkpoints.at(-1).successes}]));
const passed=first20.trained!==null && first20.trained<=48 &&
  arms.filter(name=>name!=='trained').every(name=>early.trained-early[name]>=.06 &&
    (first20[name]===null || first20.trained<first20[name])) &&
  final.trained.expected>=.20 && final.trained.successes>=22;
artifact.comparison.summary={controlPassed,passed,first20,early,final};
await finish(passed?'passed':'failed');
