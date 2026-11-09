"""Rebuild prefixes using bootstrap multiplicities and logit margins."""
import hashlib
import json
import math
import random
from collections import Counter
from itertools import accumulate
from pathlib import Path

root = Path(__file__).resolve().parent
read = lambda f: json.loads((root/f).read_text())
rows = read('audit_predictions.json')
assert hashlib.sha256((root/'audit_predictions.json').read_bytes()).hexdigest()==read('lock.json')['audit_predictions.json']
n = len(rows)
err = [int(int(r['logits'][1]>r['logits'][0])!=r['label']) for r in rows]
orders = [sorted(range(n),key=lambda i:-abs(rows[i]['logits'][1]-rows[i]['logits'][0])),
          sorted(range(n),key=lambda i:hashlib.sha256(('122:'+str(rows[i]['idx'])).encode()).hexdigest())]
points = read('points.json')
assert len(points)==361 and points[0]['gain'] is None and points[0]['ci'] is None
for key,order in zip(('confidence_errors','shuffle_errors'),orders):
    assert list(accumulate(err[i] for i in order))==[p[key] for p in points[1:]]
rng=random.Random(124)
hist=[Counter() for _ in range(n)]
primary=read('primary_counts.json')
assert len(primary)==4000
for expected in primary:
    counts=Counter(rng.randrange(n) for _ in range(n))
    curves=[accumulate(err[i] for i in order for _ in range(counts[i])) for order in orders]
    diffs=[b-a for a,b in zip(*curves)]
    assert expected==diffs[269]
    for h,delta in zip(hist,diffs):
        h[delta]+=1
for k,h in enumerate(hist,1):
    values=sorted(h.elements())
    for q,observed in zip((.025,.975),points[k]['ci']):
        pos=q*(len(values)-1)
        lo=math.floor(pos)
        estimate=(values[lo]+(pos-lo)*(values[lo+1]-values[lo]))/k
        assert math.isclose(estimate,observed,abs_tol=1e-14)
    p=points[k]
    assert p['gain']==(p['shuffle_errors']-p['confidence_errors'])/k
print('PASS 4000 shared draws, 360 nonempty prefix pairs and 720 endpoints, empty risk undefined')
