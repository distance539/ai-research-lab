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
c,e,f=read('calibration.json'),read('exploration.json'),read('fits.json');pred=read('predictions.json');out=[]
for rows,fit in [(c,f['global_fit'])]+[([r for r in c if int(r['retained_tokens']>256)==g],f['group_fits'][g]) for g in [0,1]]:
 b=opt(rows);assert abs(loss(rows,b)-loss(rows,fit['beta']))<1e-12;out.append(dict(n=len(rows),independent_beta=b,loss_gap=loss(rows,b)-loss(rows,fit['beta'])))
for name,ts in [('global_method',[f['global_fit']['temperature']]*2),('group_method',[v['temperature'] for v in f['group_fits']])]:
 check=[]
 for r,saved in zip(e,pred[name]['rows']):
  t=ts[int(r['retained_tokens']>256)];z=[v/t for v in r['logits']];m=max(z);w=[math.exp(v-m) for v in z];p=[v/sum(w) for v in w];correct=int(z.index(max(z))==r['label'])
  assert abs(-math.log(p[r['label']])-saved['loss'])<1e-12
  assert abs(sum((p[k]-int(k==r['label']))**2 for k in [0,1])-saved['brier'])<1e-12
  assert correct==saved['correct'];check.append((-max(p),r['id'],1-correct))
 check.sort();assert [r[1] for r in check]==pred[name]['order'];assert sum(r[2] for r in check[:800])==pred[name]['errors80']
result=dict(status='passed',independent_optimizers=out,checked_predictions=2000,inputs={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ['calibration.json','exploration.json','protocol.json','fits.json','predictions.json']})
(R/'audit_result.json').write_text(json.dumps(result,indent=2));print(result)
