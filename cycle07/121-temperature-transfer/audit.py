"""Independent direct objective search, validation of identities and audit cells."""
import hashlib
import json
import math
from pathlib import Path

def load(p):
    return json.loads(Path(p).read_text())

def loss(rows,t):
    return sum(math.log1p(math.exp((z[1-y]-z[y])/t)) for z,y in rows)/len(rows)

def optimum(rows):
    lo,hi=.05,20
    for _ in range(110):
        u,v=(2*lo+hi)/3,(lo+2*hi)/3
        if loss(rows,u)<loss(rows,v):
            hi=v
        else:
            lo=u
    return (lo+hi)/2

cfg=load('protocol.json')
for p,h in cfg['input_sha256'].items():
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
formal=load('formal.json')
assert hashlib.sha256(Path('formal.json').read_bytes()).hexdigest()==json.loads(Path('formal.log').read_text().splitlines()[-1])['result']
assert all(r in formal for r in load('smoke.json'))
results=load('result.json')
temps={'unit':1}
splits={}
for source in ['A','B','C']:
    for split,n in [('calibration',256),('audit',360)]:
        base=load(split+'_predictions.json')
        if source=='C':
            rows=[(r['logits'],r['label']) for r in base]
        else:
            records=[r for r in formal if r['split']==split and r['source']==source]
            assert [(r['idx'],r['label']) for r in records]==[(b['idx'],b['label']) for b in base]
            rows=[(r['scores'],r['label']) for r in records]
        assert len(rows)==n
        splits[source,split]=rows
    t=optimum(splits[source,'calibration'])
    assert abs(t-results['temperature'][source])<2e-5
    temps[source]=t
for source in ['A','B','C']:
    rows=splits[source,'audit']
    for param,t in temps.items():
        cell=results['audit'][source][param]
        assert abs(loss(rows,t)-cell['nll'])<1e-6
        pred=[max(range(2),key=lambda j:z[j]/t) for z,y in rows]
        assert sum(pred)==cell['positive'] and sum(p==r[1] for p,r in zip(pred,rows))==cell['correct']
print(json.dumps(dict(passed=True,temperatures=temps,verified_cells=12,formal_sha256=hashlib.sha256(Path('formal.json').read_bytes()).hexdigest(),input_sha256=cfg['input_sha256'])))
