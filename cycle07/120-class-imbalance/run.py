import json,hashlib,math,time,resource,platform,sys,statistics
from pathlib import Path
from model_run import fit
C=Path(__file__).resolve().parent
def metrics(rows,t):
 vals=[];bs=[];correct=0
 for r in rows:
  d=(r['logits'][1]-r['logits'][0])/t;y=r['label'];v=(1-2*y)*d;p=1/(1+math.exp(-d)) if d>=0 else math.exp(d)/(1+math.exp(d))
  vals.append(max(v,0)+math.log1p(math.exp(-abs(v))));bs.append(2*(p-y)**2);correct+=int(d>0)==y
 return {'n':len(rows),'nll':statistics.mean(vals),'brier':statistics.mean(bs),'correct':correct}
def main():
 start=time.perf_counter();q=json.loads((C/'protocol.json').read_text())
 for f,h in q['inputs'].items():assert hashlib.sha256((C/f).read_bytes()).hexdigest()==h
 cal=json.loads((C/'calibration_predictions.json').read_text());aud=json.loads((C/'audit_predictions.json').read_text());out=[]
 for seed in q['seeds']:
  groups={y:sorted([r for r in cal if r['label']==y],key=lambda r:hashlib.sha256((str(seed)+':'+str(r['idx'])).encode()).hexdigest()) for y in [0,1]}
  for pos in q['positive_counts']:
   rows=groups[0][:100-pos]+groups[1][:pos];assert len(rows)==100
   f=fit(rows);t=f['temperature'];out.append({'seed':seed,'positive':pos,'ids':[r['idx'] for r in rows],'fit':f,'calibration':metrics(rows,t),'audit':metrics(aud,t),'by_label':{str(y):metrics([r for r in aud if r['label']==y],t) for y in [0,1]}})
 (C/'results.json').write_text(json.dumps(out,indent=2));summary={}
 for pos in q['positive_counts']:
  rr=[r for r in out if r['positive']==pos];summary[pos]={k:{'mean':statistics.mean(v),'sd':statistics.stdev(v),'min':min(v),'max':max(v)} for k,v in {'T':[r['fit']['temperature'] for r in rr],'nll':[r['audit']['nll'] for r in rr],'brier':[r['audit']['brier'] for r in rr]}.items()}
 summary['raw']=metrics(aud,1);(C/'summary.json').write_text(json.dumps(summary,indent=2));(C/'analysis_environment.json').write_text(json.dumps({'python':platform.python_version(),'os':platform.platform(),'seconds':time.perf_counter()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)},indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
