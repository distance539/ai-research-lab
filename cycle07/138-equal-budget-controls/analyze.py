"""Registered full-budget control; confirmation outputs never loaded."""
import hashlib,json,math,time,resource,sys
from pathlib import Path
import numpy as np
from metrics import read,fit,sigmoid
R=Path(__file__).resolve().parent
def put(n,v): (R/n).write_text(json.dumps(v,indent=2)+'\n')
def evaluate(rows,ts):
 out=[]
 for r in rows:
  t=ts[int(r['retained_tokens']>256)];x=(r['logits'][1]-r['logits'][0])/t;p=sigmoid(x)
  out.append(dict(id=r['id'],loss=max(x,0)+math.log1p(math.exp(-abs(x)))-r['label']*x,brier=2*(p-r['label'])**2,confidence=max(p,1-p),correct=int((x>0)==r['label'])))
 order=sorted(out,key=lambda r:(-r['confidence'],r['id']));errors=0;prefix=[]
 for k,r in enumerate(order,1):errors+=1-r['correct'];prefix.append(errors/k)
 bins=[[0,0,0.] for _ in range(10)]
 for r in out:
  v=bins[min(9,int(r['confidence']*10))];v[0]+=1;v[1]+=r['correct'];v[2]+=r['confidence']
 return dict(nll=sum(r['loss'] for r in out)/len(out),brier=sum(r['brier'] for r in out)/len(out),hits=sum(r['correct'] for r in out),ece10=sum(abs(v[1]-v[2]) for v in bins)/len(out),risk80=prefix[799],errors80=round(prefix[799]*800),prefix=prefix,order=[r['id'] for r in order],rows=out)
def main():
 assert hashlib.sha256((R/'protocol.json').read_bytes()).hexdigest()==(R/'protocol.sha256').read_text().strip()
 c,e,p=read('calibration.json'),read('exploration.json'),read('pools.json')
 for rows,key in [(c,'calibration'),(e,'exploration')]:assert [r['id'] for r in rows]==[r['id'] for r in p[key]]
 assert not {r['id'] for r in c}&{r['id'] for r in e}
 groups=[[r for r in c if int(r['retained_tokens']>256)==g] for g in [0,1]]
 start=time.perf_counter();global_fit=fit(c);global_s=time.perf_counter()-start
 start=time.perf_counter();gs=[fit(g) for g in groups] if min(map(len,groups))>=50 else [global_fit]*2;group_s=time.perf_counter()-start
 fits=dict(global_fit=global_fit,group_fits=gs,group_sizes=list(map(len,groups)),global_seconds=global_s,group_seconds=group_s,global_gradient_row_evaluations=102*len(c),group_gradient_row_evaluations=102*sum(map(len,groups)),scope='full1000 only; no ablation or confirmation')
 a=evaluate(e,[global_fit['temperature']]*2);b=evaluate(e,[g['temperature'] for g in gs])
 delta=np.array([y['loss']-x['loss'] for x,y in zip(a['rows'],b['rows'])]);rng=np.random.default_rng(137)
 boot=np.array([delta[rng.integers(0,1000,1000)].mean() for _ in range(10000)])
 result=dict(global_metrics={k:v for k,v in a.items() if k not in ['rows','order','prefix']},group_metrics={k:v for k,v in b.items() if k not in ['rows','order','prefix']},delta_nll=float(delta.mean()),exploratory_ci=np.quantile(boot,[.025,.975]).tolist(),delta_risk80=b['risk80']-a['risk80'],rank_changes=sum(x!=y for x,y in zip(a['order'],b['order'])),accepted_set_changes=len(set(a['order'][:800])^set(b['order'][:800])),confirmation_accessed=False)
 put('fits.json',fits);put('predictions.json',dict(global_method=a,group_method=b));put('results.json',result)
 print(json.dumps(result,indent=2));print('fit seconds',global_s,group_s)
if __name__=='__main__': main()
