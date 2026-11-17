"""Independent vector audit: uses two-column softmax rather than sigmoid."""
import json, hashlib
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent
def read(n):return json.loads((D/n).read_bytes())
def sha(n):return hashlib.sha256((D/n).read_bytes()).hexdigest()
pools=read('pools.json'); assert sha('pools.json')=='d472a0676f5f9b58aea434ce694ddd40ea2d5b4ab50f9c3aa28aec896d982764'
for k in ('movie','id','text'): assert len({r[k] for g in pools.values() for r in g})==3000
assert [r['id'] for r in read('formal.json')]==[r['id'] for r in pools['exploration']]
for s,f in zip(read('smoke.json'),read('formal.json')[:8]):
 assert s['id']==f['id'] and s['label']==f['label'] and np.allclose(s['logits'],f['logits'],rtol=0,atol=1e-5)
for domain,n,src in [('SST2',360,'audit_predictions.json'),('IMDb',1000,'formal.json')]:
 r=read(src);assert len(r)==n
 z=np.array([x['logits'] for x in r]);y=np.array([x['label'] for x in r]);hot=np.eye(2)[y];ok=z.argmax(1)==y
 for mode,t in [('raw',1),('frozen',read('fit.json')['temperature'])]:
  w=np.exp((z-z.max(1,keepdims=True))/t);p=w/w.sum(1,keepdims=True);c=p.max(1);a=c>=.9
  b=np.minimum((10*c).astype(int),9);rows=[[int(sum(b==i)),int(ok[b==i].sum()),float(c[b==i].sum())] for i in range(10)]
  s=read('results.json')[domain][mode];assert np.allclose(rows,s['bins'],rtol=0,atol=1e-8)
  for k,v in dict(n=n,hits=ok.sum(),nll=-np.log(p[np.arange(n),y]).mean(),brier=((p-hot)**2).sum(1).mean(),ece=sum(abs(i-j) for _,i,j in rows)/n,accepted=a.sum(),errors=(a&~ok).sum()).items():assert abs(s[k]-v)<1e-10,(domain,mode,k)
print('PASS: 3000 identities, 8 smoke, 1360 paired decisions, 40 bins;', {n:sha(n) for n in ('audit_predictions.json','fit.json','formal.json','protocol.json','pools.json','results.json')})
