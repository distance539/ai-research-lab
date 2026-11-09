"""Pointwise paired bootstrap over sentence identities."""
import hashlib
import json
import random
from itertools import accumulate
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def curve(draw, rank, errors):
    ordered = sorted(draw, key=rank.__getitem__)
    return list(accumulate(errors[i] for i in ordered))


def main():
    for name, digest in json.loads((ROOT / 'lock.json').read_text()).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    rows = json.loads((ROOT / 'audit_predictions.json').read_text())
    assert [r['idx'] for r in rows] == json.loads((ROOT / 'split_manifest.json').read_text())['protocol_audit']
    n = len(rows)
    assert n == len({r['idx'] for r in rows}) == 360
    errors = [1-r['correct'] for r in rows]
    confidence = [(-r['confidence'], r['idx']) for r in rows]
    shuffle = [hashlib.sha256(f"122:{r['idx']}".encode()).digest() for r in rows]
    orig = [curve(range(n), rank, errors) for rank in (confidence, shuffle)]
    rng = random.Random(124)
    deltas = []
    for _ in range(4000):
        draw = [rng.randrange(n) for _ in range(n)]
        left, right = [curve(draw, rank, errors) for rank in (confidence, shuffle)]
        deltas.append([b-a for a,b in zip(left,right)])
    bounds = np.quantile(deltas, [.025,.975], axis=0, method='linear')
    points = [dict(k=0, confidence_errors=0, shuffle_errors=0, gain=None, ci=None)]
    for k,(a,b) in enumerate(zip(*orig),1):
        points.append(dict(k=k,confidence_errors=a,shuffle_errors=b,
                           gain=(b-a)/k,ci=(bounds[:,k-1]/k).tolist()))
    (ROOT/'points.json').write_text(json.dumps(points,indent=2))
    (ROOT/'primary_counts.json').write_text(json.dumps([row[269] for row in deltas]))
    for k in (1,18,36,90,180,270,324,360):
        print(points[k])


if __name__ == '__main__':
    main()
