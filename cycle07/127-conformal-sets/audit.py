"""Rebuild the inclusion decisions from calibrated logit margins, without probabilities."""
import hashlib
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parent
def read(n):
    return json.loads((root/n).read_bytes())
def margin(r,y):
    return r['logits'][y]-r['logits'][1-y]

for name,h in read('lock.json').items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h
cal = read('calibration_predictions.json')
rows = read('audit_predictions.json')
observed = read('sets.json')
assert len(cal)==256 and len(rows)==len(observed)==360
ordered = sorted((margin(r,r['label']) for r in cal),reverse=True)
cutoff = ordered[((len(cal)+1)*9+9)//10-1]
s = read('results.json')
assert abs(1/(1+math.exp(cutoff))-s['threshold']) < 1e-14 and s['rank']==232
sizes,misses = [0]*3,[0]*3
correct = 0
for r,o in zip(rows,observed):
    labels=[y for y in range(2) if margin(r,y)>=cutoff]
    assert (r['idx'],r['label'],labels)==(o['idx'],o['label'],o['labels'])
    assert o['covered']==(r['label'] in labels)
    sizes[len(labels)]+=1
    misses[len(labels)]+=int(r['label'] not in labels)
    correct+=int(max(range(2),key=lambda y:margin(r,y))==r['label'])
assert sizes==s['size_counts'] and misses==s['misses_by_size']
assert s['covered']==360-sum(misses) and s['coverage']==s['covered']/360
assert s['mean_size']==(sizes[1]+2*sizes[2])/360 and s['baseline_correct']==correct
print('PASS: rank, 720 membership decisions, 360 identities and labels, histogram, coverage, size, baseline and input hashes')
