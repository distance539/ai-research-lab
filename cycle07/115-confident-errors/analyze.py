"""Registered descriptive error audit; no fitting or semantic annotations."""
import json,hashlib,time,resource,sys,platform
from pathlib import Path
P=Path(__file__).resolve().parent
start=time.perf_counter()
read=lambda n:json.loads((P/n).read_text())
cfg=read('analysis_config.json'); raw=(P/'formal_predictions.json').read_bytes()
assert hashlib.sha256(raw).hexdigest()==cfg['prediction_sha256']
rows=json.loads(raw)
def length(r): return 'short' if r['raw_tokens']<=16 else 'medium' if r['raw_tokens']<=32 else 'long'
def summary(name,rr):
 h=[r for r in rr if r['confidence']>=cfg['threshold']]; e=[r for r in h if not r['correct']]
 return dict(group=name,n=len(rr),errors=sum(1-r['correct'] for r in rr),high_n=len(h),high_errors=len(e),high_error_rate=len(e)/len(h) if h else None,high_error_ids=[r['idx'] for r in e])
out=[summary('all',rows)]
for g in ['short','medium','long']: out.append(summary(g,[r for r in rows if length(r)==g]))
for y in [0,1]: out.append(summary('label'+str(y),[r for r in rows if r['label']==y]))
for g in ['short','medium','long']:
 for y in [0,1]: out.append(summary(g+'/label'+str(y),[r for r in rows if length(r)==g and r['label']==y]))
top=sorted([r for r in rows if not r['correct']],key=lambda r:(-r['confidence'],r['idx']))[:cfg['top_k']]
result={'groups':out,'top_errors':top,'prediction_sha256':cfg['prediction_sha256']}
(P/'group_results.json').write_text(json.dumps(result,indent=2)+'\n')
(P/'analysis_environment.json').write_text(json.dumps({'seconds':time.perf_counter()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),'python':platform.python_version(),'platform':platform.platform()},indent=2)+'\n')
print(json.dumps(result,indent=2))
