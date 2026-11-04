import json,hashlib,math,time,platform,resource
from pathlib import Path
import numpy as np
import sklearn
from sklearn.isotonic import IsotonicRegression
from model_run import fit
C=Path(__file__).resolve().parent
def save(n,x):(C/n).write_text(json.dumps(x,indent=2,allow_nan=False))
def sigmoid(x):return 1/(1+math.exp(-x)) if x>=0 else math.exp(x)/(1+math.exp(x))
def metrics(rows,ps,eps):
 y=[r['label'] for r in rows];true=[p if t else 1-p for p,t in zip(ps,y)]
 return {'n':len(y),'correct':sum(int(p>.5)==t for p,t in zip(ps,y)),'brier':sum(2*(p-t)**2 for p,t in zip(ps,y))/len(y),'nll_clipped':sum(-math.log(max(eps,min(1-eps,v))) for v in true)/len(y),'infinite_nll_rows':sum(v==0 for v in true),'boundary_rows':sum(p in (0,1) for p in ps)}
def main():
 start=time.perf_counter();q=json.loads((C/'protocol.json').read_text())
 for f,h in q['inputs'].items():assert hashlib.sha256((C/f).read_bytes()).hexdigest()==h
 cal=json.loads((C/'calibration_predictions.json').read_text());aud=json.loads((C/'audit_predictions.json').read_text());out=[]
 order=sorted(cal,key=lambda r:hashlib.sha256(('119:'+str(r['idx'])).encode()).hexdigest())
 for n in q['sizes']:
  rows=order[:n];t=fit(rows)['temperature'];x=[sigmoid(r['logits'][1]-r['logits'][0]) for r in rows]
  iso=IsotonicRegression(increasing=True,out_of_bounds='clip').fit(x,[r['label'] for r in rows])
  run={'n':n,'ids':[r['idx'] for r in rows],'temperature':t,'x_thresholds':iso.X_thresholds_.tolist(),'y_thresholds':iso.y_thresholds_.tolist(),'evaluations':{}}
  for split,rr in [('calibration',rows),('audit',aud)]:
   raw=[sigmoid(r['logits'][1]-r['logits'][0]) for r in rr];ps={'raw':raw,'temperature':[sigmoid((r['logits'][1]-r['logits'][0])/t) for r in rr],'isotonic':iso.predict(raw).tolist()}
   run['evaluations'][split]={m:{'p_positive':v,'metrics':metrics(rr,v,q['nll_epsilon'])} for m,v in ps.items()}
  # Additional numerical diagnostic; no change to preregistered metrics.
  for split,rr in [('calibration',rows),('audit',aud)]:
   for method,tt in [('raw',1),('temperature',t)]:
    vals=[]
    for r in rr:
     v=(1-2*r['label'])*(r['logits'][1]-r['logits'][0])/tt
     vals.append(max(v,0)+math.log1p(math.exp(-abs(v))))
    run['evaluations'][split][method]['metrics']['stable_unclipped_nll']=sum(vals)/len(vals)
  out.append(run)
 save('results.json',out);save('analysis_environment.json',{'python':platform.python_version(),'platform':platform.platform(),'sklearn':sklearn.__version__,'numpy':np.__version__,'seconds':time.perf_counter()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
 print(json.dumps([{ 'n':r['n'],'T':r['temperature'],'metrics':{s:{m:v['metrics'] for m,v in a.items()} for s,a in r['evaluations'].items()}} for r in out],indent=2))
if __name__=='__main__':main()
