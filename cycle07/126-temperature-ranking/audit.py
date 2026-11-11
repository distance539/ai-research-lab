"""Independent high-precision binary-margin audit of stored outputs."""
import hashlib
import json
from decimal import Decimal as D, getcontext
from pathlib import Path
getcontext().prec = 50
def load(p):
    return json.loads(Path(p).read_bytes(), parse_float=D)
fit = load('fit.json')
cal = load('calibration_predictions.json')
audit = load('audit_predictions.json')
manifest = load('split_manifest.json')
assert hashlib.sha256(Path('calibration_predictions.json').read_bytes()).hexdigest() == fit['calibration_sha256']
assert [r['idx'] for r in audit] == manifest['protocol_audit']
assert len(cal)==256 and len(audit)==len({r['idx'] for r in audit})==360
assert not {r['idx'] for r in cal}.intersection(r['idx'] for r in audit)
def margin(row):
    return row['logits'][1] - row['logits'][0]
def sigmoid(x):
    return 1/(1+(-x).exp())
gradient=sum(margin(r)*(sigmoid(margin(r)/fit['temperature'])-r['label']) for r in cal)/len(cal)
assert abs(gradient)<D('1e-12')
ordered=sorted(audit,key=lambda r:(-abs(margin(r)),r['idx']))
details=load('paired.json')
result=load('results.json')
for name,t in [('raw',D(1)),('scaled',fit['temperature'])]:
    records=details[name]
    errors=0
    calibration=[D(0)]*10
    loss=D(0)
    area=D(0)
    accepted=0
    assert records['curve'][0]==dict(k=0,errors=0,risk=None)
    assert len(records['scores'])==360 and len(records['curve'])==361
    for k,(row,saved) in enumerate(zip(ordered,records['scores']),1):
        confidence=sigmoid(abs(margin(row))/t)
        wrong=int(int(margin(row)>0)!=row['label'])
        assert saved['idx']==row['idx'] and saved['wrong']==wrong
        assert abs(saved['confidence']-confidence)<D('1e-14')
        errors+=wrong
        loss-=(1-confidence if wrong else confidence).ln()
        calibration[min(9,int(confidence*10))]+=confidence-(1-wrong)
        point=records['curve'][k]
        assert point['errors']==errors and point['k']==k and abs(point['risk']-D(errors)/k)<D('1e-14')
        area+=D(errors)/k
        accepted+=confidence>=D('.99')
    for field,value in [('ece',sum(map(abs,calibration))),('nll',loss),('aurc',area)]:
        assert abs(result[name][field]-value/360)<D('1e-12')
    assert result[name]['fixed_threshold']==records['curve'][accepted]
    assert result[name]['fixed_budget']==records['curve'][270]
print('PASS: 720 scores, 722 prefix records, NLL/ECE/AURC and fitted calibration gradient',gradient)
