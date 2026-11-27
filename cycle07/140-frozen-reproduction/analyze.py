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
 fits={};pred={};results={};deltas={}
 for n in [128,1000]:
  rows=c[:n];groups=[[r for r in rows if int(r['retained_tokens']>256)==g] for g in [0,1]]
  start=time.perf_counter();f=fit(rows);gs=[fit(g) for g in groups] if min(map(len,groups))>=50 else [f]*2
  fits[str(n)]=dict(global_fit=f,group_fits=gs,group_sizes=list(map(len,groups)),fallback=min(map(len,groups))<50,fit_seconds=time.perf_counter()-start,global_gradient_rows=102*n,additional_group_gradient_rows=0 if min(map(len,groups))<50 else 102*n)
  for name,ts in [('global',[f['temperature']]*2),('group',[x['temperature'] for x in gs])]:
   key=f'{n}_{name}';pred[key]=evaluate(e,ts);results[key]={k:v for k,v in pred[key].items() if k not in ['rows','order','prefix']}
  deltas[str(n)]=np.array([b['loss']-a['loss'] for a,b in zip(pred[f'{n}_global']['rows'],pred[f'{n}_group']['rows'])])
 vectors={'128':deltas['128'],'1000':deltas['1000'],'interaction':deltas['1000']-deltas['128']}
 rng=np.random.default_rng(137);boot={k:[] for k in vectors}
 for _ in range(10000):
  idx=rng.integers(0,1000,1000)
  for k,v in vectors.items():boot[k].append(float(v[idx].mean()))
 contrasts={k:dict(mean=float(v.mean()),exploratory_ci=np.quantile(boot[k],[.025,.975]).tolist()) for k,v in vectors.items()}
 put('fits.json',fits);put('predictions.json',pred);put('results.json',dict(cells=results,contrasts=contrasts,confirmation_accessed=False))
 print(json.dumps(read('results.json'),indent=2))
if __name__=='__main__':main()
