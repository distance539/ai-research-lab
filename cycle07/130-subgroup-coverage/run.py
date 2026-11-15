"""Apply a single calibrated LAC threshold, then describe fixed strata."""
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def read(name):
    return json.loads((ROOT / name).read_bytes())

def save(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def scores(row):
    z = row['logits']
    weights = [math.exp(v-max(z)) for v in z]
    return [1-w/sum(weights) for w in weights]

def main():
    cfg = read('protocol.json')
    cal = read('calibration_predictions.json')
    audit = read('run_predictions.json')
    manifest = read('split_manifest.json')
    assert [r['idx'] for r in cal] == manifest['calibration']
    assert [r['idx'] for r in audit] == manifest['protocol_audit']
    assert len(set(manifest['calibration'] + manifest['protocol_audit'])) == 616
    rank = math.ceil((len(cal)+1)*(1-Fraction(cfg['alpha'])))
    cutoff = sorted(scores(r)[r['label']] for r in cal)[rank-1]
    bins = ['short','medium','long']
    groups = ['all'] + ['label'+str(y) for y in cfg['label_groups']] + bins
    groups += [f'label{y}/{b}' for y in cfg['label_groups'] for b in bins]
    table = {g:dict(n=0, hits=0, doubles=0, empty=0) for g in groups}
    members = []
    for row in audit:
        labels = [i for i,s in enumerate(scores(row)) if s <= cutoff]
        size = len(labels)
        bucket = bins[sum(row['retained_tokens'] > t for t in cfg['length_edges'])]
        label = 'label'+str(row['label'])
        for key in ['all', label, bucket, label+'/'+bucket]:
            entry = table[key]
            entry['n'] += 1
            entry['hits'] += row['label'] in labels
            entry['doubles'] += size == 2
            entry['empty'] += size == 0
        members.append(dict(idx=row['idx'], group=bucket, labels=labels))
    save('memberships.json', members)
    save('results.json', dict(rank=rank, cutoff=cutoff, groups=table))
    files = 'protocol.json config.json resources.lock.json split_manifest.json calibration_predictions.json audit_predictions.json run_predictions.json run.py audit.py baseline.py model_run.py'.split()
    save('lock.json', {f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files})
    print(rank,cutoff,table)

if __name__ == '__main__':
    main()
