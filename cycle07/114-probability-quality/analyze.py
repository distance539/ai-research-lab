import json, math, time, platform, resource, hashlib
from pathlib import Path
p=Path(__file__).resolve().parent
read=lambda n:json.loads((p/n).read_text())
t=time.perf_counter(); cfg=read('analysis_config.json')
assert hashlib.sha256((p/'formal_predictions.json').read_bytes()).hexdigest()==cfg['input_sha256']
rows=read('formal_predictions.json'); out=[]; summary=[]
for T in cfg['temperatures']:
 assert T>0
 group=[]
 for r in rows:
  z=[v/T for v in r['logits']]; m=max(z); e=[math.exp(v-m) for v in z]; den=sum(e); probs=[v/den for v in e]
  pred=max(range(2),key=lambda j:z[j]); nll=m+math.log(den)-z[r['label']]
  item=dict(idx=r['idx'],temperature=T,probabilities=probs,prediction=pred,correct=int(pred==r['label']),nll=nll,brier=sum((probs[j]-int(j==r['label']))**2 for j in range(2)))
  assert pred==r['prediction']; group.append(item);out.append(item)
 summary.append(dict(temperature=T,n=len(group),correct=sum(x['correct'] for x in group),accuracy=sum(x['correct'] for x in group)/len(group),nll=sum(x['nll'] for x in group)/len(group),brier=sum(x['brier'] for x in group)/len(group),nll_correct_contribution=sum(x['nll'] for x in group if x['correct'])/len(group),nll_error_contribution=sum(x['nll'] for x in group if not x['correct'])/len(group)))
for name,obj in [('temperature_predictions.json',out),('temperature_results.json',summary),('analysis_environment.json',dict(python=platform.python_version(),os=platform.platform(),seconds=time.perf_counter()-t,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,device='CPU',memory_scope='process high-water mark on macOS'))]:
 (p/name).write_text(json.dumps(obj,indent=2)+'\n')
print(json.dumps(summary,indent=2))
