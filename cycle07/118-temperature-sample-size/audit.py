"""Independent stdlib golden-section optimizer and softplus evaluation."""
import math,json,hashlib,statistics
from pathlib import Path
C=Path(__file__).resolve().parent
p=json.loads((C/'protocol.json').read_text());cal=json.loads((C/'calibration_predictions.json').read_text());test=json.loads((C/'audit_predictions.json').read_text());runs=json.loads((C/'runs.json').read_text())
def loss(rows,b):
 return math.fsum(max(b*(r['logits'][1]-r['logits'][0]),0)+math.log1p(math.exp(-abs(b*(r['logits'][1]-r['logits'][0]))))-r['label']*b*(r['logits'][1]-r['logits'][0]) for r in rows)/len(rows)
maxerr=0;maxopt=0
assert len(runs)==250
for seed in p['seeds']:
 order=sorted(cal,key=lambda r:hashlib.sha256(('118:%d:%d'%(seed,r['idx'])).encode()).hexdigest())
 for n in p['sizes']:
  r=next(r for r in runs if r['seed']==seed and r['n']==n);rows=order[:n];assert r['ids']==[x['idx'] for x in rows];assert not set(r['ids'])&{x['idx'] for x in test}
  lo,hi=.05,20.;phi=(math.sqrt(5)-1)/2
  for _ in range(100):
   x=hi-phi*(hi-lo);y=lo+phi*(hi-lo)
   if loss(rows,x)<loss(rows,y):hi=y
   else:lo=x
  opt=min([.05,20.,(lo+hi)/2],key=lambda b:loss(rows,b));gap=abs(loss(rows,opt)-loss(rows,r['beta']));maxopt=max(maxopt,gap);assert gap<1e-10
  assert r['cal_errors']==sum(int(x['logits'][1]>x['logits'][0])!=x['label'] for x in rows)
  assert r['audit_correct']==330
  e=abs(loss(test,r['beta'])-r['audit_nll']);maxerr=max(maxerr,e);assert e<1e-10
s=json.loads((C/'summary.json').read_text())
for n in p['sizes']:
 rr=[r for r in runs if r['n']==n];v=[r['T'] for r in rr];mean=math.fsum(v)/50;sd=math.sqrt(math.fsum((x-mean)**2 for x in v)/49);assert abs(sd-s[str(n)]['T_sd'])<1e-10
result={'passed':True,'rows':250,'max_nll_error':maxerr,'max_optimal_loss_gap':maxopt,'inputs':{f:hashlib.sha256((C/f).read_bytes()).hexdigest() for f in ['protocol.json','runs.json','calibration_predictions.json','audit_predictions.json','audit.py']}}
(C/'audit_result.json').write_text(json.dumps(result,indent=2));print(result)
