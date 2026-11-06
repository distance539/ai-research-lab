"""Fit positive inverse temperatures using calibration rows only."""
import json
import math
import hashlib
from pathlib import Path

def read(name):
    return json.loads(Path(name).read_text())

def probability(x):
    return 1/(1+math.exp(-x))

def fit(rows, bounds):
    lo, hi = bounds
    for _ in range(90):
        mid = (lo+hi)/2
        grad = sum(d*(probability(mid*d)-y) for d,y in rows)
        if grad>0:
            hi = mid
        else:
            lo = mid
    return (lo+hi)/2

def metrics(rows, beta):
    loss = 0
    correct = positive = 0
    for d,y in rows:
        margin = (1-2*y)*d*beta
        loss += max(margin,0)+math.log1p(math.exp(-abs(margin)))
        positive += d>0
        correct += (d>0)==y
    return dict(n=len(rows),nll=loss/len(rows),correct=correct,positive=positive)

def main():
    cfg = read('protocol.json')
    for file, expected in cfg['input_sha256'].items():
        assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==expected
    formal = read('formal.json')
    assert len(formal)==1232
    identity = read('split_manifest.json')
    data = {}
    for split, key in [('calibration','calibration'),('audit','protocol_audit')]:
        base = read(split+'_predictions.json')
        assert [r['idx'] for r in base]==identity[key]
        data[split] = {'C':[(r['logits'][1]-r['logits'][0],r['label']) for r in base]}
        for source in ['A','B']:
            rows = [r for r in formal if r['source']==source and r['split']==split]
            assert [(r['idx'],r['label']) for r in rows]==[(r['idx'],r['label']) for r in base]
            data[split][source] = [(r['scores'][1]-r['scores'][0],r['label']) for r in rows]
    beta = {s:fit(v,cfg['beta_bounds']) for s,v in data['calibration'].items()}
    result = dict(temperature={s:1/b for s,b in beta.items()},audit={})
    for target, rows in data['audit'].items():
        result['audit'][target]={s:metrics(rows,b) for s,b in {'unit':1,**beta}.items()}
    Path('result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':
    main()
