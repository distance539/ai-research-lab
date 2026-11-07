"""Independent scalar reconstruction. Uses no primary analysis helpers."""
import hashlib
import json
import math
from pathlib import Path

base = Path(__file__).resolve().parent
def load(name):
    return json.loads((base / name).read_bytes())

records = load('audit_predictions.json')
errors, scores = {}, {}
for r in records:
    i = r['idx']
    left, right = r['logits']
    scores[i] = 1 / (1 + math.exp(-abs(left - right)))
    errors[i] = int(int(right > left) != r['label'])
    assert abs(scores[i] - r['confidence']) < 1e-12
    assert r['correct'] == 1 - errors[i]
assert len(errors) == 360 and sum(errors.values()) == 30
curve = load('curve.json')
for name, order in [('ranked', sorted(errors, key=lambda i: (-scores[i], i))),
                    ('random', sorted(errors, key=lambda i: hashlib.sha256(('122:'+str(i)).encode()).hexdigest()))]:
    assert curve[name+'_ids'] == order
    assert len(curve[name]) == 361
    for k, point in enumerate(curve[name]):
        count = sum(errors[i] for i in order[:k])
        assert point == dict(k=k, coverage=k/360, errors=count, risk=count/k if k else None)
for threshold, point in load('thresholds.json').items():
    chosen = [i for i in scores if scores[i] >= float(threshold)]
    assert point == curve['ranked'][len(chosen)]
    assert point['errors'] == sum(errors[i] for i in chosen)
assert curve['uniform_expected_risk'] == sum(errors.values())/360
print('PASS: 360 logits, 722 prefix points, 4 thresholds; input SHA256',
      hashlib.sha256((base/'audit_predictions.json').read_bytes()).hexdigest())
