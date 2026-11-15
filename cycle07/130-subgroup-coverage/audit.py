"""Independent membership by calibration margin rank, aggregate by predicates."""
import json
from pathlib import Path
from collections import Counter

def read(name):
    return json.loads(Path(__file__).with_name(name).read_bytes())

def margin(row, label):
    z = row['logits']
    return z[1-label]-z[label]

def main():
    calibration = read('calibration_predictions.json')
    audit = read('audit_predictions.json')
    members = read('memberships.json')
    result = read('results.json')
    assert len(members)==len(audit)==360 and len(calibration)==256
    assert result['rank']==232 and max(r['retained_tokens'] for r in audit)<512
    ranks = [margin(r,r['label']) for r in calibration]
    counts = Counter()
    for r,m in zip(audit,members):
        size = r['retained_tokens']
        group = 'short' if size<=16 else 'long' if size>32 else 'medium'
        labels=[i for i in (0,1) if sum(v < margin(r,i) for v in ranks)<232]
        assert m == dict(idx=r['idx'], group=group, labels=labels)
        for g in ('all','label%d'%r['label'],group,'label%d/%s'%(r['label'],group)):
            for k,v in dict(n=1,hits=r['label'] in labels,empty=not labels,doubles=len(labels)==2).items():
                counts[g,k]+=v
    assert len(result['groups'])==12
    assert all(counts[g,k]==v for g,d in result['groups'].items() for k,v in d.items())
    print('720 inclusion decisions, 12 subgroup rows and counts independently verified')

if __name__ == '__main__':
    main()
