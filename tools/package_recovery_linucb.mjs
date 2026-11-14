// Isolated synthetic recovery readout. The original RPS controllers are untouched.
import { RpsContextualBandit } from '../js/rps.js';

const DIM = 4;
export const standardizeMbonRates = (rates, mean, std) => {
  if (rates.length !== 3 || mean.length !== 3 || std.length !== 3 ||
      rates.some(v => !Number.isFinite(v) || v < 0) ||
      std.some(v => !Number.isFinite(v) || v <= 0)) throw Error('Invalid MBON features');
  return rates.map((rate,j) => (Math.log1p(rate) - mean[j]) / std[j]);
};
function solve(A, rhs) {
  const m=A.map((row,i)=>[...row,rhs[i]]);
  for(let k=0;k<DIM;k++) {
    let pivot=k;
    for(let i=k+1;i<DIM;i++) if(Math.abs(m[i][k])>Math.abs(m[pivot][k])) pivot=i;
    if(!Number.isFinite(m[pivot][k]) || Math.abs(m[pivot][k])<1e-12) throw Error('Singular LinUCB matrix');
    [m[k],m[pivot]]=[m[pivot],m[k]];
    const scale=m[k][k]; for(let j=k;j<=DIM;j++) m[k][j]/=scale;
    for(let i=0;i<DIM;i++) if(i!==k) {
      const factor=m[i][k]; for(let j=k;j<=DIM;j++) m[i][j]-=factor*m[k][j];
    }
  }
  return m.map(row=>row[DIM]);
}
const dot=(a,b)=>a.reduce((sum,value,j)=>sum+value*b[j],0);
export class LinUcbCore {
  constructor({ridge=1,bonus=.5}={}) {
    if(ridge!==1 || bonus!==.5) throw Error('Locked LinUCB configuration');
    this.ridge=ridge; this.bonus=bonus;
    this.A=[0,1,2].map(()=>[0,1,2,3].map(i=>[0,1,2,3].map(j=>i===j?ridge:0)));
    this.b=[0,1,2].map(()=>[0,0,0,0]);
    this.n=[0,0,0]; this.last=null;
  }
  predict(f) {
    if(f.length!==3 || f.some(v=>!Number.isFinite(v))) throw Error('Invalid neural features');
    const x=[1,...f];
    const means=[], uncertainty=[];
    for(let a=0;a<3;a++) {
      const theta=solve(this.A[a],this.b[a]), variance=dot(x,solve(this.A[a],x));
      if(!Number.isFinite(variance) || variance<0) throw Error('Invalid uncertainty');
      means.push(dot(x,theta)); uncertainty.push(Math.sqrt(variance));
    }
    return {x,means,uncertainty};
  }
  choose(f,training=true) {
    const {x,means,uncertainty}=this.predict(f);
    const bonuses=uncertainty.map(s=>training?this.bonus*s:0);
    const selectionScores=means.map((v,a)=>v+bonuses[a]);
    const action=selectionScores.indexOf(Math.max(...selectionScores));
    this.last={x,action};
    return {action,means,uncertainty,bonuses,selectionScores};
  }
  update(reward) {
    if(!this.last || !Number.isFinite(reward)) throw Error('Missing decision or invalid reward');
    const {x,action:a}=this.last;
    for(let i=0;i<DIM;i++) {
      this.b[a][i]+=reward*x[i];
      for(let j=0;j<DIM;j++) this.A[a][i][j]+=x[i]*x[j];
    }
    this.n[a]++; this.last=null;
  }
  snapshot() { return {A:this.A.map(m=>m.map(row=>[...row])),b:this.b.map(row=>[...row]),n:[...this.n]}; }
  restore(s) {
    if(s.A?.length!==3 || s.b?.length!==3 || s.n?.length!==3) throw Error('Invalid state');
    this.A=s.A.map(m=>m.map(row=>[...row])); this.b=s.b.map(row=>[...row]); this.n=[...s.n];
    this.last=null;
    for(let a=0;a<3;a++) solve(this.A[a],this.b[a]);
  }
}
export class RecoveryLinUcb extends RpsContextualBandit {
  constructor(net,opts) {
    super(net,opts);
    this.core=new LinUcbCore(); this.kind='package_recovery_linucb_opt_in';
  }
  decide(net) {
    this.rates=net.groupMbon.map(ids=>ids.reduce((sum,id)=>sum+net.count[id],0)/ids.length);
    this.f=standardizeMbonRates(this.rates,this.featureMean,this.featureStd);
    const d=this.core.choose(this.f,this.explore);
    this.scores=d.means; this.uncertainty=d.uncertainty; this.bonuses=d.bonuses;
    this.selectionScores=d.selectionScores; this.action=d.action;
    this.V=Math.tanh(this.rates[this.action]/this.o.cRef);
    this.elig=Float64Array.from(this.plasticPre[this.action],p=>net.trace[p]);
    return this.action;
  }
  afterDopamine(net,learn) {
    super.afterDopamine(net,false);
    if(learn && this.readoutLearn) this.core.update(this.r);
  }
}
