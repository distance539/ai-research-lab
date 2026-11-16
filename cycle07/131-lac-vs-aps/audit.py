"""Independent score and membership reconstruction, sharing only the softmax primitive."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import torch

root = Path(__file__).resolve().parent
def load(name):
    return json.loads((root/name).read_text())

cal = load('calibration_predictions.json')
audit = load('run_predictions.json')
pair = load('memberships.json')
summary = load('results.json')
assert len(cal)==256 and len(audit)==len(pair)==360
k = (9*(len(cal)+1)+9)//10
assert k==232
def score(row, method):
    a,b = torch.tensor(row['logits'],dtype=torch.float64).softmax(0).tolist()
    assert a != b
    if method=='LAC':
        return (1-a,1-b)
    last = 1.0 if method=='normalized_endpoint' else a+b
    return (a,last) if a>b else (last,b)

report = {'rank':k,'inputs':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['calibration_predictions.json','run_predictions.json','memberships.json','results.json']}}
for method in ['LAC','APS','normalized_endpoint']:
    true = sorted(score(r,method)[r['label']] for r in cal)
    q = true[k-1]
    sizes = [0,0,0]
    hits = 0
    for row, claimed in zip(audit,pair):
        assert (row['idx'],row['label']) == (claimed['idx'],claimed['label'])
        selected = [i for i,s in enumerate(score(row,method)) if s<=q]
        hits += row['label'] in selected
        sizes[len(selected)] += 1
        if method!='normalized_endpoint':
            assert selected == claimed[method]
    result = dict(q=q,sizes=sizes,hits=hits,slots=sum(i*n for i,n in enumerate(sizes)))
    if method!='normalized_endpoint':
        assert summary[method]==result
    report[method]=result
counts=Counter(str((p['label'] in p['LAC'],p['label'] in p['APS'])) for p in pair)
assert counts == summary['paired_hits']
assert sum(r['LAC']!=r['APS'] for r in pair)==summary['different']
report['calibration_errors']=sum(not r['correct'] for r in cal)
report['audit_last_mass_hex']=dict(Counter(max(score(r,'APS')).hex() for r in audit))
print(json.dumps(report,indent=2))
