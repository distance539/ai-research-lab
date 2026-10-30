import json,math,hashlib
from pathlib import Path
p=Path(__file__).resolve().parent
read=lambda n:json.loads((p/n).read_text())
raw=read('formal_predictions.json'); cfg=read('analysis_config.json'); saved=read('temperature_predictions.json'); sums=read('temperature_results.json')
assert hashlib.sha256((p/'formal_predictions.json').read_bytes()).hexdigest()==cfg['input_sha256']
assert len(saved)==1024 and len(sums)==4
errors=[]
for k,T in enumerate(cfg['temperatures']):
 nll=[];bs=[];cs=[]
 for i,r in enumerate(raw):
  d=(r['logits'][1]-r['logits'][0])/T; prob=1/(1+math.exp(-d)); loss=max(d,0)+math.log1p(math.exp(-abs(d)))-r['label']*d
  b=2*(prob-r['label'])**2; correct=int(int(d>0)==r['label']); x=saved[k*len(raw)+i]
  assert (x['idx'],x['temperature'],x['prediction'],x['correct'])==(r['idx'],T,int(d>0),correct)
  errors.extend([abs(x['nll']-loss),abs(x['brier']-b),abs(x['probabilities'][1]-prob)])
  nll.append(loss);bs.append(b);cs.append(correct)
 s=sums[k];assert s['temperature']==T and s['n']==256 and s['correct']==233
 for name,v in [('nll',sum(nll)/256),('brier',sum(bs)/256),('accuracy',sum(cs)/256),('nll_correct_contribution',sum(v for v,c in zip(nll,cs) if c)/256),('nll_error_contribution',sum(v for v,c in zip(nll,cs) if not c)/256)]:errors.append(abs(s[name]-v))
assert max(errors)<1e-12
result=dict(passed=True,rows=1024,max_absolute_error=max(errors),input_sha256=cfg['input_sha256'],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),method='independent binary sigmoid, softplus and 2*(p-y)^2; no analyzer imports')
(p/'temperature_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
