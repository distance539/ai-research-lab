"""Compute coverage and conditional error on locked model predictions."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def read(name):
    return json.loads((ROOT / name).read_text())

def write(name, data):
    (ROOT / name).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')

def measure(count, errors, total):
    return dict(k=count, coverage=count / total, errors=errors,
                risk=errors / count if count else None)

def prefix(rows):
    n = len(rows)
    points = [measure(0, 0, n)]
    errors = 0
    for k, row in enumerate(rows, start=1):
        errors += 1 - row['correct']
        points.append(measure(k, errors, n))
    return points

def main():
    for name, digest in read('lock.json').items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    rows = read('audit_predictions.json')
    ids = [row['idx'] for row in rows]
    assert ids == read('split_manifest.json')['protocol_audit']
    assert len(set(ids)) == len(rows) == 360
    ranked = sorted(rows, key=lambda row: (-row['confidence'], row['idx']))
    shuffled = sorted(rows, key=lambda row: hashlib.sha256(
        f"122:{row['idx']}".encode()).digest())
    curve = dict(ranked=prefix(ranked), random=prefix(shuffled),
                 ranked_ids=[row['idx'] for row in ranked],
                 random_ids=[row['idx'] for row in shuffled])
    curve['uniform_expected_risk'] = curve['ranked'][-1]['risk']
    write('curve.json', curve)
    summary = {}
    for threshold in (0.5, 0.90, 0.99, 1.0):
        k = sum(row['confidence'] >= threshold for row in rows)
        summary[str(threshold)] = curve['ranked'][k]
    write('thresholds.json', summary)
    print(summary)
    for k in (180, 270, 324, 360):
        print(k, curve['ranked'][k], curve['random'][k])

if __name__ == '__main__':
    main()
