// One locked synthetic recovery comparison; no RPS or demo modules are modified.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { access, open, readFile } from 'node:fs/promises';
import { Network, rng, wiring } from '../js/sim.js';
import { RpsContextualBandit } from '../js/rps.js';
import { RecoveryLinUcb } from './package_recovery_linucb.mjs';

const stage=process.argv[2];
if(!['initial','confirmation'].includes(stage)) throw Error('Use initial or confirmation');
const root=new URL('../',import.meta.url), read=async path=>JSON.parse(await readFile(new URL(path,root)));
const output=new URL(`data/package_recovery_uncertainty_${stage}.json`,root);
try {await access(output);throw Error('Write-once uncertainty result already exists');}
catch(e){if(e.code!=='ENOENT')throw e;}
if(stage==='confirmation' && (await read('data/package_recovery_uncertainty_initial.json')).status!=='passed')
  throw Error('Initial comparison did not pass');
const protocolSha256=createHash('sha256').update(await readFile(new URL('docs/package-recovery-uncertainty-protocol.md',root))).digest('hex');
const sub=await read('data/subgraph.json');
const {featureMean,featureStd}=(await read('data/rps_contextual_bandit.json')).design;
const conf=stage==='initial'?{neuralSeed:15001,trainSeed:2026110821,evalSeed:2026110822}:
  {neuralSeed:15002,trainSeed:2026110823,evalSeed:2026110824};
const cueProb=[[.70,.20,.10],[.20,.70,.10],[.10,.10,.80],[.20,.60,.20]];
const successProb=[[.78,.35,.20],[.32,.78,.22],[.12,.15,.82],[.36,.62,.30]];
const delays=[2,4,8], checkpoints=[0,24,48,96,144];
const artifact={stage,protocolSha256,conf,controller:{ridge:1,bonus:.5},
  task:{cueProb,successProb,delays},status:'started',validity:null,arms:null,summary:null};
async function finish(status){artifact.status=status;const file=await open(output,'wx');
  try{await file.writeFile(JSON.stringify(artifact,null,2)+'\n');}finally{await file.close();}
  console.log(JSON.stringify({stage,status,validity:artifact.validity,summary:artifact.summary}));}
const expected=(mode,a)=>successProb[mode][a]*1.4-.4-.03*delays[a]-(a===2?.2:0);
function makeCases(seed,perMode){
  const random=rng(seed), rows=[];
  for(let mode=0;mode<4;mode++)for(let j=0;j<perMode;j++){
    const u=random();let cue=0,sum=0;
    while(cue<2 && u>=(sum+=cueProb[mode][cue]))cue++;
    const outcomes=successProb[mode].map(p=>Number(random()<p));
    rows.push({id:`${seed}-${mode}-${j}`,mode,cue,outcomes});
  }
  for(let i=rows.length-1;i>0;i--){const j=Math.floor(random()*(i+1));[rows[i],rows[j]]=[rows[j],rows[i]];}
  return rows;
}
const trainingCases=makeCases(conf.trainSeed,36),evaluationCases=makeCases(conf.evalSeed,9);
artifact.task.trainingCases=trainingCases;artifact.task.evaluationCases=evaluationCases;
const makeNet=()=>new Network(sub,wiring(sub,'real',conf.neuralSeed),{seed:conf.neuralSeed});
const snapshot=(name,agent)=>name==='original'?{
  W:agent.W.map(w=>[...w]),b:[...agent.b],baseline:agent.baseline}:agent.core.snapshot();
function actor(name,frozen=false,source=null){
  const net=makeNet(),opts={seed:conf.neuralSeed,featureMean,featureStd,readoutLearn:!frozen};
  const agent=name==='original'?new RpsContextualBandit(net,opts):new RecoveryLinUcb(net,opts);
  if(source){const saved=snapshot(name,source.agent);
    if(name==='original'){saved.W.forEach((w,a)=>agent.W[a].set(w));agent.b.set(saved.b);agent.baseline=saved.baseline;}
    else agent.core.restore(saved);}
  agent.explore=!frozen;return {net,agent};
}
function outcome(spec,action){
  const success=spec.outcomes[action],delay=delays[action],escalation=action===2?.2:0;
  return {success,failed:1-success,delay,escalation,
    reward:(success?1:-.4)-.03*delay-escalation,expectedReward:expected(spec.mode,action)};
}
function round(s,spec,index,learn,evaluation=false){
  const {net,agent}=s,key=conf.neuralSeed*10000000+(evaluation?900000:0)+index*1000;
  net.resetState();net.rand=rng(key+1);agent.rand=rng(key+501);
  const vpn=net.laneVpn[spec.cue].slice(2);net.setDrive(vpn,150);let row;
  for(let t=0;t<400;t++){
    if(t===50)net.resetCounts();
    if(t===240){net.setDrive(vpn,0);const action=agent.decide(net),result=outcome(spec,action);
      row={index,id:spec.id,cue:spec.cue,mode:spec.mode,action,
        rates:[...agent.rates],f:[...agent.f],scores:[...agent.scores],
        epsilonExplored:agent.kind==='rps_contextual_bandit' && learn ? rng(key+501)()<agent.epsilon : null,
        uncertainty:agent.uncertainty?[...agent.uncertainty]:null,
        bonus:agent.bonuses?[...agent.bonuses]:null,
        selectionScores:agent.selectionScores?[...agent.selectionScores]:null,
        ...result};
      agent.reward(net,result.reward);net.resetCounts();}
    if(t===340)agent.afterDopamine(net,learn);
    net.step();
  }
  assert.ok(row && net.m.every(x=>x===1));return row;
}
function summary(rows){
  const n=rows.length;
  return {n,successes:rows.reduce((s,r)=>s+r.success,0),
    failures:rows.reduce((s,r)=>s+r.failed,0),delay:rows.reduce((s,r)=>s+r.delay,0),
    escalations:rows.filter(r=>r.escalation>0).length,
    totalReward:rows.reduce((s,r)=>s+r.reward,0),
    meanReward:rows.reduce((s,r)=>s+r.reward,0)/n,
    meanExpectedReward:rows.reduce((s,r)=>s+r.expectedReward,0)/n,
    actions:[0,1,2].map(a=>rows.filter(r=>r.action===a).length),
    byCue:[0,1,2].map(cue=>{const selected=rows.filter(r=>r.cue===cue);
      return {cue,n:selected.length,successes:selected.reduce((s,r)=>s+r.success,0),
        actions:[0,1,2].map(a=>selected.filter(r=>r.action===a).length),
        expected:selected.reduce((s,r)=>s+r.expectedReward,0)/selected.length};})};
}
function frozen(name,source){const s=actor(name,true,source),before=JSON.stringify(snapshot(name,s.agent));
  const rows=evaluationCases.map((spec,i)=>round(s,spec,i,false,true));
  assert.equal(JSON.stringify(snapshot(name,s.agent)),before);
  assert.ok(s.net.m.every(x=>x===1));
  if(name==='uncertainty')assert.ok(rows.every(r=>r.bonus.every(v=>v===0) &&
    r.action===r.scores.indexOf(Math.max(...r.scores))));
  return {...summary(rows),rows,explorationOff:true,weightsUnchanged:true,neuralWeightsUnchanged:true};}
function train(name){const s=actor(name),initial=JSON.stringify(snapshot(name,s.agent));
  const points=[{attempt:0,...frozen(name,s)}],rows=[];
  for(let i=0;i<trainingCases.length;i++){
    rows.push(round(s,trainingCases[i],i,true));
    if(checkpoints.includes(i+1))points.push({attempt:i+1,...frozen(name,s)});
  }
  return {name,initialWeights:JSON.parse(initial),finalWeights:snapshot(name,s.agent),
    training:{...summary(rows),rows},
    trainingCheckpoints:checkpoints.slice(1).map(n=>({attempt:n,...summary(rows.slice(0,n))})),
    frozenCheckpoints:points,neuralWeightsUnchanged:s.net.m.every(x=>x===1)};
}
artifact.arms={original:train('original'),uncertainty:train('uncertainty')};
const original=artifact.arms.original,uncertainty=artifact.arms.uncertainty;
const sameInputs=original.training.rows.every((r,i)=>r.id===uncertainty.training.rows[i].id &&
  JSON.stringify(r.f)===JSON.stringify(uncertainty.training.rows[i].f)) &&
  original.frozenCheckpoints.every((p,j)=>p.rows.every((r,i)=>r.id===uncertainty.frozenCheckpoints[j].rows[i].id &&
    JSON.stringify(r.f)===JSON.stringify(uncertainty.frozenCheckpoints[j].rows[i].f)));
const initialZero=original.initialWeights.W.every(w=>w.every(v=>v===0)) &&
  original.initialWeights.b.every(v=>v===0) && uncertainty.initialWeights.b.every(b=>b.every(v=>v===0));
const caseIntegrity=trainingCases.length===144 && evaluationCases.length===36 &&
  new Set([...trainingCases,...evaluationCases].map(r=>r.id)).size===180 &&
  [trainingCases,evaluationCases].every(cases=>[0,1,2,3].every(m=>cases.filter(r=>r.mode===m).length===cases.length/4));
const updates=uncertainty.finalWeights.n.reduce((s,n)=>s+n,0)===144;
const frozenIntegrity=[original,uncertainty].every(arm=>arm.frozenCheckpoints.length===5 &&
  arm.frozenCheckpoints.every(p=>p.rows.length===36 && p.explorationOff && p.weightsUnchanged) &&
  arm.neuralWeightsUnchanged);
artifact.validity={passed:sameInputs && initialZero && caseIntegrity && updates && frozenIntegrity,
  sameInputs,initialZero,caseIntegrity,updates,frozenIntegrity};
if(!artifact.validity.passed){await finish('stopped_validity');process.exit(0);}
const o0=original.frozenCheckpoints[0],oF=original.frozenCheckpoints.at(-1);
const u0=uncertainty.frozenCheckpoints[0],uF=uncertainty.frozenCheckpoints.at(-1);
const early=arm=>[24,48,96].reduce((s,n)=>s+arm.frozenCheckpoints.find(p=>p.attempt===n).meanExpectedReward,0)/3;
const cue2=uF.byCue[2],cue2Clear=cue2.actions[2];
const teacherGate=uF.meanExpectedReward>=.20 && uF.meanExpectedReward-u0.meanExpectedReward>=.10 &&
  uF.meanExpectedReward-oF.meanExpectedReward>=.05 && uF.successes>=22 &&
  uF.actions.every(n=>n>=4) && cue2Clear/cue2.n>=.60;
const earlyAdvantage=early(uncertainty)-early(original);
const failureReduction=original.training.failures-uncertainty.training.failures;
const rewardAdvantage=uncertainty.training.meanReward-original.training.meanReward;
const speedGate=earlyAdvantage>=.04 && failureReduction>=5 && rewardAdvantage>=.03;
artifact.summary={passed:teacherGate && speedGate,teacherGate,speedGate,
  initial:{original:o0.meanExpectedReward,uncertainty:u0.meanExpectedReward},
  final:{original:{expected:oF.meanExpectedReward,successes:oF.successes,actions:oF.actions},
    uncertainty:{expected:uF.meanExpectedReward,successes:uF.successes,actions:uF.actions,
      cue2Clear, cue2N:cue2.n}},early:{original:early(original),uncertainty:early(uncertainty),advantage:earlyAdvantage},
  training:{originalFailures:original.training.failures,uncertaintyFailures:uncertainty.training.failures,
    failureReduction,originalMeanReward:original.training.meanReward,
    uncertaintyMeanReward:uncertainty.training.meanReward,rewardAdvantage}};
await finish(artifact.summary.passed?'passed':'failed');
