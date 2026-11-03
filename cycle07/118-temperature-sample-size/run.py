"""Registered nested subsampling of real frozen classifier logits."""
import json,hashlib,time,statistics,platform,resource,sys
from pathlib import Path
from model_run import fit,scaled
C=Path(__file__).resolve().parent
def save(n,x):(C/n).write_text(json.dumps(x,indent=2))
def main():
 start=time.perf_counter();p=json.loads((C/'protocol.json').read_text())
 for f,h in p['inputs'].items():assert hashlib.sha256((C/f).read_bytes()).hexdigest()==h,f
 cal=json.loads((C/'calibration_predictions.json').read_text());audit=json.loads((C/'audit_predictions.json').read_text());runs=[]
 for seed in p['seeds']:
  order=sorted(cal,key=lambda r:hashlib.sha256(f"118:{seed}:{r['idx']}".encode()).hexdigest())
  for n in p['sizes']:
   rows=order[:n];f=fit(rows);t=f['temperature'];a=scaled(audit,t)
   runs.append({'seed':seed,'n':n,'ids':[r['idx'] for r in rows],'T':t,'beta':f['beta'],'boundary':f['boundary'],'gradient':f['gradient'],'cal_errors':sum(not r['correct'] for r in rows),'cal_nll':statistics.mean(r['nll'] for r in scaled(rows,t)),'audit_nll':statistics.mean(r['nll'] for r in a),'audit_correct':sum(r['correct'] for r in a)})
 save('runs.json',runs);s={}
 for n in p['sizes']:
  rr=[r for r in runs if r['n']==n];ts=[r['T'] for r in rr];ls=[r['audit_nll'] for r in rr]
  s[str(n)]={'T_mean':statistics.mean(ts),'T_sd':statistics.stdev(ts),'T_median':statistics.median(ts),'T_min':min(ts),'T_max':max(ts),'boundary':sum(r['boundary'] for r in rr),'zero_errors':sum(r['cal_errors']==0 for r in rr),'audit_nll_median':statistics.median(ls),'audit_nll_min':min(ls),'audit_nll_max':max(ls),'audit_worse_than_raw':sum(v>statistics.mean(r['nll'] for r in audit) for v in ls)}
 save('summary.json',s);save('analysis_environment.json',{'python':platform.python_version(),'platform':platform.platform(),'seconds':time.perf_counter()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),'protocol_sha256':hashlib.sha256((C/'protocol.json').read_bytes()).hexdigest()});print(json.dumps(s,indent=2))
if __name__=='__main__':main()
