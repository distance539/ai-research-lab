import json,math,hashlib
from pathlib import Path
C=Path(__file__).resolve().parent
q=json.loads((C/'protocol.json').read_text());cal=json.loads((C/'calibration_predictions.json').read_text());aud=json.loads((C/'audit_predictions.json').read_text());runs=json.loads((C/'results.json').read_text());mx=0
for f,h in q['inputs'].items():assert hashlib.sha256((C/f).read_bytes()).hexdigest()==h
assert len(runs)==60
for r in runs:
 ids=[]
 for y,n in [(0,100-r['positive']),(1,r['positive'])]:ids += [v['idx'] for v in sorted((v for v in cal if v['label']==y),key=lambda v:hashlib.sha256(f"{r['seed']}:{v['idx']}".encode()).hexdigest())[:n]]
 assert ids==r['ids'] and len(set(ids))==100
 rows=[v for v in cal if v['idx'] in ids]
 def loss(beta,rr):
  return sum(math.log1p(math.exp(-abs(beta*(v['logits'][1]-v['logits'][0]))))+max((1-2*v['label'])*beta*(v['logits'][1]-v['logits'][0]),0) for v in rr)/len(rr)
 lo,hi=.05,20;g=(math.sqrt(5)-1)/2
 for _ in range(120):
  a=hi-g*(hi-lo);b=lo+g*(hi-lo)
  if loss(a,rows)<loss(b,rows):hi=b
  else:lo=a
 beta=(hi+lo)/2;assert abs(loss(beta,rows)-loss(r['fit']['beta'],rows))<1e-12
 for rr,m in [(rows,r['calibration']),(aud,r['audit'])]+[([v for v in aud if v['label']==y],r['by_label'][str(y)]) for y in [0,1]]:
  err=abs(loss(r['fit']['beta'],rr)-m['nll']);mx=max(mx,err);assert err<1e-12
  pp=[1/(1+math.exp(-r['fit']['beta']*(v['logits'][1]-v['logits'][0]))) for v in rr]
  assert abs(sum(2*(p-v['label'])**2 for p,v in zip(pp,rr))/len(rr)-m['brier'])<1e-12
  assert m['correct']==sum(int(v['logits'][1]>v['logits'][0])==v['label'] for v in rr)
result={'passed':True,'fits_checked':60,'method':'independent golden-section minimization and scalar softplus/Brier reconstruction','max_metric_error':mx,'input_hashes':{f:hashlib.sha256((C/f).read_bytes()).hexdigest() for f in ['audit.py','protocol.json','results.json','calibration_predictions.json','audit_predictions.json']}}
(C/'audit_result.json').write_text(json.dumps(result,indent=2));print(result)
