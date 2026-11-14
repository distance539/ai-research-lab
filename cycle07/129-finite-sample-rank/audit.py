"""Check memberships via margin ranks without softmax or quantile selection."""
from pathlib import Path
import json
import hashlib

root = Path(__file__).resolve().parent
def read(name):
    return json.loads((root/name).read_bytes())

def margin(row, y):
    return row['logits'][1-y] - row['logits'][y]

for name, expected in read('lock.json').items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected
calibration = read('calibration_predictions.json')
audit = read('audit_predictions.json')
observed = read('memberships.json')
results = read('results.json')
assert len(observed) == len(results) == 2*len(read('protocol.json')['sizes'])
for block, summary in zip(observed, results):
    n = summary['n']
    rank = (9*(n+1)+9)//10
    assert summary['rank'] == rank
    cutoff = rank if block['rule']=='conformal' else min(rank,n)
    smaller = [margin(r,r['label']) for r in calibration[:n]]
    hist = [0,0,0]
    hits = 0
    assert len(block['decisions']) == len(audit) == 360
    for row, (idx, labels) in zip(audit, block['decisions']):
        expected = [y for y in range(2) if sum(m < margin(row,y) for m in smaller) < cutoff]
        assert idx == row['idx'] and labels == expected
        hist[len(expected)] += 1
        hits += row['label'] in expected
    assert hist == summary['sizes'] and hits == summary['hits']
    assert summary['mean_size'] == (hist[1]+2*hist[2])/360
print('PASS: 22 conditions, 7920 sets and 15840 binary inclusion decisions independently reconstructed')
