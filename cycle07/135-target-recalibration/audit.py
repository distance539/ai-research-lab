import json,math,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
rd=lambda n:json.loads((R/n).read_text())
cal=rd('formal.json');dev=rd('exploration.json');result=rd('results.json');cfg=rd('protocol.json')
def loss(rows,b):
 return sum(math.log(math.exp((r['logits'][0]-max(r['logits']))*b)+math.exp((r['logits'][1]-max(r['logits']))*b))+(max(r['logits'])-r['logits'][r['label']])*b for r in rows)/len(rows)
checks=[]
for v in result['budgets']:
 rows=cal[:v['n']];a,b=cfg['beta_bounds'];ratio=(math.sqrt(5)-1)/2
 for _ in range(120):
  c=b-ratio*(b-a);d=a+ratio*(b-a)
  if loss(rows,c)<loss(rows,d):b=d
  else:a=c
 beta=(a+b)/2;assert abs(loss(rows,beta)-loss(rows,v['beta']))<1e-10
 assert abs(loss(dev,v['beta'])-v['exploration']['nll'])<1e-12
 ps=[]
 for r in dev:
  z=[x*v['beta'] for x in r['logits']];m=max(z);e=[math.exp(x-m) for x in z];ps.append([x/sum(e) for x in e])
 hits=sum(int(p[1]>p[0])==r['label'] for r,p in zip(dev,ps));br=sum(sum((p[k]-int(k==r['label']))**2 for k in range(2)) for r,p in zip(dev,ps))/len(dev)
 assert hits==v['exploration']['hits'] and abs(br-v['exploration']['brier'])<1e-12
 checks.append(dict(n=v['n'],independent_beta=beta,nll=loss(dev,beta)))
for group,name in [('calibration','formal.json'),('exploration','exploration.json')]:assert [r['id'] for r in rd(name)]==[r['id'] for r in rd('pools.json')[group]]
assert not set(x['id'] for x in cal)&set(x['id'] for x in dev)
(R/'audit_result.json').write_text(json.dumps(dict(passed=True,checks=checks,inputs={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ['formal.json','exploration.json','results.json','audit.py']}),indent=2));print('independent golden-search loss / softmax / argmax / Brier / pool identity passed')
