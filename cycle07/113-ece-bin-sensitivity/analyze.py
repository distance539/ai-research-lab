"""Fixed predictions, paired bootstrap, no fitting or selection."""
import json,hashlib,random,time,platform,resource,sys
from pathlib import Path
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def measure(rows,m):
 bins=[]
 for j in range(m):
  r=[x for x in rows if min(int(x['confidence']*m),m-1)==j]; n=len(r)
  acc=sum(x['correct'] for x in r)/n if n else None
  conf=sum(x['confidence'] for x in r)/n if n else None
  ci=None
  if n:
   z=1.959963984540054; d=1+z*z/n; c=(acc+z*z/(2*n))/d; h=z*((acc*(1-acc)/n+z*z/(4*n*n))**.5)/d; ci=[c-h,c+h]
  bins.append(dict(bin=j,n=n,accuracy=acc,confidence=conf,accuracy_wilson95=ci,contribution=n/len(rows)*abs(conf-acc) if n else 0))
 return dict(bins=m,ece=sum(b['contribution'] for b in bins),per_bin=bins)
def quantile(v,q):
 v=sorted(v); t=(len(v)-1)*q; i=int(t); return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(t-i)
def main():
 start=time.perf_counter(); cfg=json.loads((P/'analysis_config.json').read_text()); rows=json.loads((P/'formal_predictions.json').read_text()); assert sha(P/'formal_predictions.json')==cfg['input_sha256']
 rng=random.Random(cfg['seed']); draws=[[rng.randrange(len(rows)) for _ in rows] for _ in range(cfg['resamples'])]
 results=[measure(rows,m) for m in cfg['bins']]; boot=[]
 for ids in draws:
  rr=[rows[i] for i in ids];boot.append([measure(rr,m)['ece'] for m in cfg['bins']])
 for k,r in enumerate(results): r['bootstrap95']=[quantile([v[k] for v in boot],q) for q in [.025,.975]]
 out=dict(input_sha256=cfg['input_sha256'],n=len(rows),correct=sum(r['correct'] for r in rows),conditions=results,paired_delta20_5=dict(value=results[-1]['ece']-results[0]['ece'],interval=[quantile([v[-1]-v[0] for v in boot],q) for q in [.025,.975]]),boundary_hits={str(m):sum(any(x['confidence']==j/m for j in range(1,m)) for x in rows) for m in cfg['bins']})
 for f,v in [('analysis_results.json',out),('bootstrap_draws.json',draws),('bootstrap_ece.json',boot)]: (P/f).write_text(json.dumps(v,indent=2)+'\n')
 (P/'analysis_environment.json').write_text(json.dumps(dict(python=sys.version,platform=platform.platform(),elapsed_seconds=time.perf_counter()-start,peak_rss_native=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,peak_rss_unit='bytes on macOS; KiB on Linux',input_sha256=cfg['input_sha256'],script_sha256=sha(Path(__file__)),protocol_sha256=sha(P/'PROTOCOL.md')),indent=2))
 print(json.dumps({str(x['bins']):[x['ece'],x['bootstrap95'],[b['n'] for b in x['per_bin']]] for x in results},indent=2))
if __name__=='__main__':main()
