"""Independent audit in logit-margin space; never imports the analysis."""
import hashlib,json,math,statistics
from collections import Counter
from pathlib import Path
p=Path(__file__).resolve().parent
def j(n):return json.loads((p/n).read_bytes())
for f,v in j('lock.json').items():assert hashlib.sha256((p/f).read_bytes()).hexdigest()==v
def margin(f):
    rows=j(f)
    for r in rows:
        assert int(r['logits'][1]>r['logits'][0])==r['prediction'] and (r['prediction']==r['label'])==r['correct']
    return {r['idx']:r['logits'][1-r['label']]-r['logits'][r['label']] for r in rows}
c=margin('calibration_predictions.json'); t=margin('audit_predictions.json')
rows=j('splits.json'); assert len(rows)==200
counts={i:[0,0] for i in c}; hs=[];ts=[]
for s,row in enumerate(rows,12800):
    ids=sorted(c,key=lambda i:(hashlib.sha256(('%d:%d'%(s,i)).encode('ascii')).hexdigest(),i))
    a,b=ids[:128],ids[128:]; assert row['seed']==s and a==row['fit'] and b==row['test']
    cut=sorted(map(c.get,a),reverse=True)[11]
    assert abs(row['q']-1/(1+math.exp(-cut)))<1e-12
    bad=[i for i in b if c[i]>cut]; outside=[i for i in t if t[i]>cut]
    assert bad==row['missed'] and outside==row['audit_missed'] and row['hits']==128-len(bad)
    hs.append(128-len(bad));ts.append(360-len(outside))
    for i in b:counts[i][0]+=1;counts[i][1]+=i not in bad
assert {str(i):v for i,v in counts.items()}==j('by_id.json')
r=j('results.json')
for name,data,den in [('held',hs,128),('fixed',ts,360)]:
    assert r[name]==dict(den=den,mean=sum(data)/(200*den),min=min(data),max=max(data),below=sum(10*h<9*den for h in data),counts={str(i):v for i,v in Counter(data).items()})
assert r['always']==sum(a==b for a,b in counts.values()) and r['never']==sum(b==0 for a,b in counts.values())
print('PASS: locked inputs, 200 thresholds, 97600 membership decisions, by-ID counts and coverage summaries.')
