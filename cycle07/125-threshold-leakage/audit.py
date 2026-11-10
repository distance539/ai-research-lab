"""Independent reconstruction from logits and official labels, using another probability formula."""
import hashlib
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parent
def load(name):
    return json.loads((root/name).read_bytes())

selection = load('selected.json')
result = load('results.json')
identities = set()
for file, points, chosen in [
    ('calibration_predictions.json', selection['calibration'], selection['threshold']),
    ('audit_predictions.json', result['audit_grid'], result['leaked']['threshold']),
]:
    records = load(file)
    assert len(points)==8 and [p['threshold'] for p in points]==[.5,.9,.95,.97,.98,.99,.995,.999]
    candidates = []
    for p in points:
        n = errors = 0
        for row in records:
            a,b = row['logits']
            score = 1/(1+math.exp(-abs(a-b)))
            assert abs(score-row['confidence']) < 1e-12
            if score >= p['threshold']:
                n += 1
                errors += int(int(b>a) != row['label'])
        assert (n,errors,len(records),n/len(records),errors/n if n else None)==tuple(p[k] for k in ('accepted','errors','total','coverage','risk'))
        if n and errors*20<=n:
            candidates.append((-n,p['threshold']))
    assert (min(candidates)[1] if candidates else None)==chosen
    for row in records:
        assert row['idx'] not in identities
        identities.add(row['idx'])
    print(file, hashlib.sha256((root/file).read_bytes()).hexdigest(),len(records))
for p in (result['locked'],result['leaked']):
    if p['threshold'] is None:
        assert p==dict(threshold=None,accepted=0,errors=0,total=360,coverage=0,risk=None)
    else:
        assert p in result['audit_grid']
assert result['locked']['threshold']==selection['threshold'] and len(identities)==616
print('PASS: 616 identities, 16 candidate rows, both choices and audit reports')
