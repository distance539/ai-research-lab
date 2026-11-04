"""Independent min-max characterization of isotonic least squares, not PAVA."""
import json,math,hashlib,bisect
from pathlib import Path
C=Path(__file__).resolve().parent
cal=json.loads((C/'calibration_predictions.json').read_text());aud=json.loads((C/'audit_predictions.json').read_text());runs=json.loads((C/'results.json').read_text());q=json.loads((C/'protocol.json').read_text());mx=0
for f,h in q['inputs'].items():assert hashlib.sha256((C/f).read_bytes()).hexdigest()==h
for run in runs:
 rows=sorted(cal,key=lambda r:hashlib.sha256(('119:'+str(r['idx'])).encode()).hexdigest())[:run['n']];assert run['ids']==[r['idx'] for r in rows]
 pairs=sorted((1/(1+math.exp(r['logits'][0]-r['logits'][1])),r['label']) for r in rows);xx=sorted(set(x for x,y in pairs));ys=[[y for x,y in pairs if x==v] for v in xx];s=[0];w=[0]
 for a in ys:s.append(s[-1]+sum(a));w.append(w[-1]+len(a))
 fitted=[max(min((s[b+1]-s[a])/(w[b+1]-w[a]) for b in range(i,len(xx))) for a in range(i+1)) for i in range(len(xx))]
 def interp(x):
  if x<=xx[0]:return fitted[0]
  if x>=xx[-1]:return fitted[-1]
  j=bisect.bisect_right(xx,x);return fitted[j-1]+(fitted[j]-fitted[j-1])*(x-xx[j-1])/(xx[j]-xx[j-1])
 for split,rr in [('calibration',rows),('audit',aud)]:
  for method,v in run['evaluations'][split].items():
   ps=[]
   for r in rr:
    d=r['logits'][1]-r['logits'][0];p=1/(1+math.exp(-d/(run['temperature'] if method=='temperature' else 1)))
    ps.append(interp(p) if method=='isotonic' else p)
   err=max(abs(a-b) for a,b in zip(ps,v['p_positive']));mx=max(mx,err);assert err<1e-10
   y=[r['label'] for r in rr];m=v['metrics'];assert m['correct']==sum(int(p>.5)==t for p,t in zip(ps,y))
   assert abs(m['brier']-2*sum((p-t)**2 for p,t in zip(ps,y))/len(y))<1e-10
   assert abs(m['nll_clipped']+sum(math.log(max(1e-6,min(1-1e-6,p if t else 1-p))) for p,t in zip(ps,y))/len(y))<1e-10
   assert m['infinite_nll_rows']==sum((p if t else 1-p)==0 for p,t in zip(ps,y))
   if method in ('raw','temperature'):
    tt=run['temperature'] if method=='temperature' else 1
    vv=[(1-2*r['label'])*(r['logits'][1]-r['logits'][0])/tt for r in rr]
    independent=sum(math.log1p(math.exp(v)) if v<0 else v+math.log1p(math.exp(-v)) for v in vv)/len(vv)
    assert abs(independent-m['stable_unclipped_nll'])<1e-10
result={'passed':True,'method':'independent weighted min-max block averages and interpolation','max_probability_error':mx,'inputs':{f:hashlib.sha256((C/f).read_bytes()).hexdigest() for f in ['protocol.json','results.json','audit.py','calibration_predictions.json','audit_predictions.json']}}
(C/'audit_result.json').write_text(json.dumps(result,indent=2));print(result)
