"""Independent decimal reconstruction, with direct counting on every accepted set."""
from decimal import Decimal, localcontext
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent
source = (root / 'audit_predictions.json').read_bytes()
result = json.loads((root / 'ranking.json').read_bytes())
data = json.loads(source)
error_by_id = {}
expected_scores = {}
with localcontext() as ctx:
    ctx.prec = 60
    for row in data:
        a, b = map(Decimal.from_float, row['logits'])
        d = abs(a-b)
        small = 1 / (1+d.exp())
        big = 1-small
        expected_scores[row['idx']] = (big, -big*big.ln()-small*small.ln(), d)
        error_by_id[row['idx']] = int(int(b>a) != row['label'])
        assert row['correct'] == 1-error_by_id[row['idx']]
names = ['maximum','entropy','gap']
assert len(data) == len(result['scores']) == len(expected_scores) == 360
for actual, original in zip(result['scores'], data):
    idx = original['idx']
    assert actual['idx'] == idx and actual['error'] == error_by_id[idx]
    for name, expected in zip(names, expected_scores[idx]):
        assert abs(actual[name]-float(expected)) < 1e-14
for col, name in enumerate(names):
    direction = 1 if name == 'entropy' else -1
    ordered = sorted(expected_scores, key=lambda i: (direction*expected_scores[i][col], i))
    assert ordered == result['orders'][name]
    assert result['duplicate_scores'][name] == len(data)-len({r[name] for r in result['scores']})
    assert len(result['curves'][name]) == 361
    for k, point in enumerate(result['curves'][name]):
        errors = sum(error_by_id[i] for i in ordered[:k])
        assert point == dict(k=k, errors=errors, risk=errors/k if k else None)
assert result['orders']['maximum'] == result['orders']['entropy'] == result['orders']['gap']
assert result['first_different_rank'] is None
print('Verified 1080 scores, 3 complete orders, 1083 prefixes, input SHA256',
      hashlib.sha256(source).hexdigest())
