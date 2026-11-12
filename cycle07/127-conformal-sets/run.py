"""Frozen split conformal experiment. Lock the rank before reading audit outputs."""
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def read(name):
    return json.loads((ROOT / name).read_bytes())

def write(name, obj):
    (ROOT / name).write_text(json.dumps(obj, indent=2) + '\n')

def probs(row):
    a, b = row['logits']
    weights = [math.exp(v - max(a,b)) for v in (a,b)]
    return [v / sum(weights) for v in weights]

def main():
    config = read('protocol.json')
    manifest = read('split_manifest.json')
    cal = read('calibration_predictions.json')
    ids = [r['idx'] for r in cal]
    assert ids == manifest['calibration'] and len(set(ids)) == 256
    scores = sorted(1 - probs(r)[r['label']] for r in cal)
    k = math.ceil((len(cal)+1)*(1-Fraction(config['alpha'])))
    q = scores[k-1] if k <= len(cal) else math.inf
    write('threshold.json', dict(rank=k, threshold=q, sorted_scores=scores))
    audit = read('run_predictions.json')
    aids = [r['idx'] for r in audit]
    assert aids == manifest['protocol_audit'] and len(set(aids)) == 360 and not set(ids).intersection(aids)
    out = []
    for row in audit:
        p = probs(row)
        candidates = [y for y in (0,1) if 1-p[y] <= q]
        out.append(dict(idx=row['idx'], label=row['label'], probabilities=p, labels=candidates,
                        covered=row['label'] in candidates, top1=p.index(max(p))))
    write('sets.json', out)
    sizes = [sum(len(r['labels']) == s for r in out) for s in range(3)]
    covered = sum(r['covered'] for r in out)
    summary = dict(n=len(out), covered=covered, coverage=covered/len(out), size_counts=sizes,
                   mean_size=sum(len(r['labels']) for r in out)/len(out),
                   misses_by_size=[sum(len(r['labels'])==s and not r['covered'] for r in out) for s in range(3)],
                   baseline_correct=sum(r['label']==r['top1'] for r in out), rank=k, threshold=q)
    write('results.json', summary)
    inputs = ['config.json','protocol.json','resources.lock.json','split_manifest.json','baseline.py',
              'model_run.py','run.py','audit.py','calibration_predictions.json','run_predictions.json','audit_predictions.json']
    write('lock.json',{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in inputs})
    print(json.dumps(summary))

if __name__ == '__main__':
    main()
