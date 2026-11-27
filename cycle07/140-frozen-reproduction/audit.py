"""Independent golden-section loss optimizer and direct two-class softmax."""
import json,math,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
read=lambda n:json.loads((R/n).read_text())
def loss(rows,b):
 return sum(math.log(math.exp((r['logits'][0]-max(r['logits']))*b)+math.exp((r['logits'][1]-max(r['logits']))*b))+(max(r['logits'])-r['logits'][r['label']])*b for r in rows)/len(rows)
def opt(rows):
 lo,hi=.05,20.;q=(math.sqrt(5)-1)/2
 for _ in range(120):
  a=hi-q*(hi-lo);b=lo+q*(hi-lo)
  if loss(rows,a)<loss(rows,b):hi=b
  else:lo=a
 return (lo+hi)/2
c,e,f=read('calibration.json'),read('exploration.json'),read('fits.json');pred=read('predictions.json');checks=[]
for n in [128,1000]:
 ff=f[str(n)];rows=c[:n];groups=[[r for r in rows if int(r['retained_tokens']>256)==g] for g in [0,1]]
 assert ff['group_sizes']==list(map(len,groups))
 assert ff['fallback']==(min(map(len,groups))<50)
 for rs,fit in [(rows,ff['global_fit'])]+list(zip(groups,ff['group_fits'])):
  if ff['fallback'] and len(rs)!=n:continue
  b=opt(rs);assert abs(loss(rs,b)-loss(rs,fit['beta']))<1e-12;checks.append(dict(n=len(rs),loss_gap=loss(rs,b)-loss(rs,fit['beta'])))
 for name,ts in [('global',[ff['global_fit']['temperature']]*2),('group',[v['temperature'] for v in ff['group_fits']])]:
  saved=pred[f'{n}_{name}'];check=[];losses=[]
  for r,s in zip(e,saved['rows']):
   z=[v/ts[int(r['retained_tokens']>256)] for v in r['logits']];m=max(z);w=[math.exp(v-m) for v in z];p=[v/sum(w) for v in w];ok=int(z.index(max(z))==r['label']);v=-math.log(p[r['label']]);losses.append(v)
   assert s['id']==r['id'] and abs(v-s['loss'])<1e-12 and s['correct']==ok
   assert abs(sum((p[k]-int(k==r['label']))**2 for k in [0,1])-s['brier'])<1e-12
   check.append((-max(p),r['id'],1-ok))
  check.sort();assert [r[1] for r in check]==saved['order'];assert sum(r[2] for r in check[:800])==saved['errors80'];assert abs(sum(losses)/len(e)-saved['nll'])<1e-12
res=read('results.json');d={str(n):sum(b['loss']-a['loss'] for a,b in zip(pred[f'{n}_global']['rows'],pred[f'{n}_group']['rows']))/1000 for n in [128,1000]}
assert abs(d['1000']-d['128']-res['contrasts']['interaction']['mean'])<1e-12
result=dict(status='passed',checked_predictions=4000,optimizer_checks=checks,interaction=d['1000']-d['128'],inputs={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ['calibration.json','exploration.json','protocol.json','fits.json','predictions.json']})
(R/'audit_result.json').write_text(json.dumps(result,indent=2));print(result)
