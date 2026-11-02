"""Independent scalar recomputation; does not import experiment code."""
import json,math,hashlib
from pathlib import Path
C=Path(__file__).resolve().parent
load=lambda n:json.loads((C/n).read_text())
r=load('results.json');f=load('fit.json');m=load('split_manifest.json');errors=[]
for pool,key in [('calibration','calibration'),('audit','protocol_audit')]:
 rows=load(pool+'_predictions.json');assert [x['idx'] for x in rows]==m[key]
 for state,t in [('raw',1),('scaled',f['temperature'])]:
  ps=[];ys=[];ns=[];cs=[];correct=[];bins=[[] for _ in range(10)]
  for row in rows:
   d=(row['logits'][1]-row['logits'][0])/t;p=1/(1+math.exp(-d));y=row['label'];nll=max(d,0)-y*d+math.log1p(math.exp(-abs(d)));conf=max(p,1-p);ok=int((d>0)==y)
   ps.append(p);ys.append(y);ns.append(nll);cs.append(conf);correct.append(ok);bins[min(int(conf*10),9)].append((conf,ok))
  n=len(rows);high=[i for i,c in enumerate(cs) if c>=.9]
  values={'n':n,'correct':sum(correct),'accuracy':sum(correct)/n,'mean_confidence':sum(cs)/n,'signed_gap':(sum(cs)-sum(correct))/n,'nll':sum(ns)/n,'brier_two_class_sum':sum(2*(p-y)**2 for p,y in zip(ps,ys))/n,'ece10':sum(abs(sum(c-o for c,o in b)) for b in bins)/n}
  for k,v in values.items():errors.append(abs(v-r[pool][state][k]));assert abs(v-r[pool][state][k])<1e-12,(pool,state,k)
  h=r[pool][state]['high_confidence'];assert h['n']==len(high) and h['correct']==sum(correct[i] for i in high)
  for i,b in enumerate(bins):
   expected=r[pool][state]['bins'][i];assert expected['n']==len(b)
   if b:
    assert abs(expected['confidence']-sum(x[0] for x in b)/len(b))<1e-12
    assert expected['accuracy']==sum(x[1] for x in b)/len(b)
cal=load('calibration_predictions.json')
def loss(t):
 vals=[]
 for x in cal:
  d=(x['logits'][1]-x['logits'][0])/t;vals.append(max(d,0)-x['label']*d+math.log1p(math.exp(-abs(d))))
 return sum(vals)/len(vals)
# Different optimizer: golden-section search directly over positive T.
lo,hi=.05,20.;g=(5**.5-1)/2
for _ in range(100):
 a=hi-g*(hi-lo);b=lo+g*(hi-lo)
 if loss(a)<loss(b):hi=b
 else:lo=a
independent_t=(lo+hi)/2;assert abs(independent_t-f['temperature'])<1e-6
assert hashlib.sha256((C/'calibration_predictions.json').read_bytes()).hexdigest()==f['calibration_sha256']
assert not set(m['calibration'])&set(m['protocol_audit'])
result={'passed':True,'max_metric_error':max(errors),'independent_temperature':independent_t,'temperature_difference':independent_t-f['temperature'],'inputs':{n:hashlib.sha256((C/n).read_bytes()).hexdigest() for n in ['calibration_predictions.json','audit_predictions.json','fit.json','results.json','split_manifest.json']}}
(C/'audit_result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
