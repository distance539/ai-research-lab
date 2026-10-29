"""Independent audit from logits; does not import analyze.py."""
import json,math,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
load=lambda f:json.loads((P/f).read_text())
def ece(rows,m):
 total=0
 for j in range(m):
  items=[r for r in rows if j/m<=r[0] and (r[0]<(j+1)/m or j==m-1)]
  total+=abs(math.fsum(r[0]-r[1] for r in items))
 return total/len(rows)
def quant(v,q):
 v=sorted(v); a=(len(v)-1)*q; lo=math.floor(a);hi=math.ceil(a);return v[lo]*(1-(a-lo))+v[hi]*(a-lo)
r=load('formal_predictions.json');cfg=load('analysis_config.json');out=load('analysis_results.json');boot=load('bootstrap_ece.json');draws=load('bootstrap_draws.json')
assert hashlib.sha256((P/'formal_predictions.json').read_bytes()).hexdigest()==cfg['input_sha256']
rows=[]
for x in r:
 z=x['logits']; p=1/(1+math.exp(-abs(z[0]-z[1])));y=int((0 if z[0]>=z[1] else 1)==x['label']);assert abs(p-x['confidence'])<1e-12 and y==x['correct'];rows.append((p,y))
for k,m in enumerate(cfg['bins']):
 c=out['conditions'][k];assert abs(ece(rows,m)-c['ece'])<1e-12
 assert sum(b['n'] for b in c['per_bin'])==len(rows)
 for b in c['per_bin']:
  j=b['bin'];rr=[x for x in rows if j/m<=x[0] and (x[0]<(j+1)/m or j==m-1)];assert len(rr)==b['n']
  if rr:
   a=sum(x[1] for x in rr)/len(rr);assert abs(a-b['accuracy'])<1e-12
   for endpoint in b['accuracy_wilson95']:
    # Wilson endpoints satisfy score-test equality.
    assert abs((a-endpoint)**2-1.959963984540054**2*endpoint*(1-endpoint)/len(rr))<1e-12
  else: assert b['accuracy'] is None and b['contribution']==0
 for i,ids in enumerate(draws):assert abs(ece([rows[j] for j in ids],m)-boot[i][k])<1e-12
 assert all(abs(quant([b[k] for b in boot],q)-v)<1e-12 for q,v in zip([.025,.975],c['bootstrap95']))
assert all(abs(quant([b[-1]-b[0] for b in boot],q)-v)<1e-12 for q,v in zip([.025,.975],out['paired_delta20_5']['interval']))
checks={'logits_probabilities_labels':True,'all_bins_and_wilson':True,'all_8000_bootstrap_metrics':True,'percentile_endpoints':True,'input_identity':True}
(P/'analysis_audit.json').write_text(json.dumps(dict(passed=True,checks=checks,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_sha256=cfg['input_sha256']),indent=2));print(checks)
