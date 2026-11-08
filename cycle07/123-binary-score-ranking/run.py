"""Rank the frozen binary predictions; retain sample identities at every position."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def scores(row):
    a, b = row['logits']
    weights = [math.exp(x - max(a, b)) for x in (a, b)]
    probabilities = [w / sum(weights) for w in weights]
    entropy = -sum(p * math.log(p) for p in probabilities if p)
    return dict(idx=row['idx'], error=int((b > a) != row['label']),
                maximum=max(probabilities), entropy=entropy, gap=abs(b - a))


def main():
    for name, sha in json.loads((ROOT / 'lock.json').read_text()).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name
    inputs = json.loads((ROOT / 'audit_predictions.json').read_text())
    rows = [scores(row) for row in inputs]
    ids = [r['idx'] for r in rows]
    assert ids == json.loads((ROOT / 'split_manifest.json').read_text())['protocol_audit']
    assert len(set(ids)) == len(ids) == 360
    orders = {}
    curves = {}
    ties = {}
    for name, direction in [('maximum', -1), ('entropy', 1), ('gap', -1)]:
        ordered = sorted(rows, key=lambda r: (direction * r[name], r['idx']))
        orders[name] = [r['idx'] for r in ordered]
        errors = 0
        points = [dict(k=0, errors=0, risk=None)]
        for k, row in enumerate(ordered, 1):
            errors += row['error']
            points.append(dict(k=k, errors=errors, risk=errors / k))
        curves[name] = points
        ties[name] = len(ordered) - len({r[name] for r in ordered})
    differences = [k for k, ids in enumerate(zip(*orders.values()), 1) if len(set(ids)) > 1]
    result = dict(scores=rows, orders=orders, curves=curves, duplicate_scores=ties,
                  first_different_rank=next(iter(differences), None))
    (ROOT / 'ranking.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print('different ranks', len(differences), 'redundant scores', ties,
          'k=270', {n: v[270] for n, v in curves.items()})


if __name__ == '__main__':
    main()
