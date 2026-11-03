"""Real frozen classifier inference, isolated fit, then audit. No audit-based selection."""
import argparse, json, math, hashlib, time, platform, resource, sys
from pathlib import Path
import baseline as b
C=Path(__file__).resolve().parent

def scaled(rows,t):
 out=[]
 for r in rows:
  z=[v/t for v in r['logits']];m=max(z);lse=m+math.log(sum(math.exp(v-m) for v in z));p=[math.exp(v-lse) for v in z];pred=int(p[1]>p[0])
  out.append(dict(r,probabilities=p,prediction=pred,confidence=max(p),correct=int(pred==r['label']),nll=lse-z[r['label']]))
 return out

def derivative(rows,beta):
 total=0.
 for r in rows:
  d=r['logits'][1]-r['logits'][0];x=beta*d;p=1/(1+math.exp(-x)) if x>=0 else math.exp(x)/(1+math.exp(x));total+=d*(p-r['label'])
 return total/len(rows)

def fit(rows):
 lo,hi=.05,20.;trace=[]
 if derivative(rows,lo)>=0:beta=lo;boundary=True
 elif derivative(rows,hi)<=0:beta=hi;boundary=True
 else:
  boundary=False
  for i in range(100):
   mid=(lo+hi)/2;g=derivative(rows,mid);trace.append({'iteration':i,'beta':mid,'gradient':g})
   if g>0:hi=mid
   else:lo=mid
  beta=(lo+hi)/2
 return {'temperature':1/beta,'beta':beta,'boundary':boundary,'gradient':derivative(rows,beta),'trace':trace}

def main():
 a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','smoke','run']);a.add_argument('--cache',type=Path,required=True);args=a.parse_args();cfg=json.loads((C/'config.json').read_text())
 if args.action=='prepare':b.prepare(args.cache,cfg);return
 import torch,transformers,numpy,pyarrow.parquet as pq
 from transformers import AutoTokenizer,AutoModelForSequenceClassification
 start=time.perf_counter();torch.set_num_threads(4);torch.manual_seed(117);torch.use_deterministic_algorithms(True)
 lock=json.loads((C/'resources.lock.json').read_text());modeldir=args.cache/cfg['model_revision'];assert b.digest(args.cache/'validation.parquet')==lock['dataset']['sha256']
 for f in lock['files']:assert b.digest(modeldir/f['name'])==f['sha256']
 manifest=json.loads((C/'split_manifest.json').read_text());allrows={r['idx']:r for r in pq.read_table(args.cache/'validation.parquet').to_pylist()}
 assert not set(manifest['calibration'])&set(manifest['protocol_audit'])
 tok=AutoTokenizer.from_pretrained(modeldir,local_files_only=True);model=AutoModelForSequenceClassification.from_pretrained(modeldir,local_files_only=True,attn_implementation='eager').eval()
 assert model.config.id2label=={0:'NEGATIVE',1:'POSITIVE'}
 shapes=[];timings={}
 def infer(ids):
  t=time.perf_counter();out=[]
  for i in range(0,len(ids),16):
   rows=[allrows[k] for k in ids[i:i+16]];texts=[r['sentence'] for r in rows];x=tok(texts,padding=True,truncation=True,max_length=512,return_tensors='pt')
   with torch.inference_mode():z=model(**x).logits
   assert list(z.shape)==[len(rows),2] and torch.isfinite(z).all();shapes.append({'input':list(x['input_ids'].shape),'logits':list(z.shape)})
   for j,r in enumerate(rows):out.append({'idx':r['idx'],'label':r['label'],'text_sha256':hashlib.sha256(r['sentence'].encode()).hexdigest(),'logits':z[j].tolist(),'retained_tokens':int(x['attention_mask'][j].sum())})
   assert time.perf_counter()-start<1800
  return scaled(out,1),time.perf_counter()-t
 prefix='smoke' if args.action=='smoke' else 'calibration'
 b.save(C/(prefix+'_start.json'),{'protocol_sha256':b.digest(C/'PROTOCOL.md'),'code_sha256':b.digest(C/'model_run.py'),'config_sha256':b.digest(C/'config.json'),'split_sha256':b.digest(C/'split_manifest.json'),'started_at':time.time(),'cache_identity':lock})
 cal,timings[prefix]=infer(manifest['calibration'][:8] if args.action=='smoke' else manifest['calibration']);b.save(C/(prefix+'_predictions.json'),cal)
 if args.action=='run':
  f=fit(cal);f['calibration_sha256']=b.digest(C/'calibration_predictions.json');b.save(C/'fit.json',f)
  audit,timings['audit']=infer(manifest['protocol_audit']);b.save(C/'audit_predictions.json',audit)
  result={'fit':{k:v for k,v in f.items() if k!='trace'},'calibration':{'raw':b.metrics(cal),'scaled':b.metrics(scaled(cal,f['temperature']))},'audit':{'raw':b.metrics(audit),'scaled':b.metrics(scaled(audit,f['temperature']))}}
  b.save(C/'results.json',result);print(json.dumps(result['fit']),flush=True)
 timings['total_after_imports']=time.perf_counter()-start
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024);assert rss<=6442450944
 b.save(C/(args.action+'_environment.json'),{'python':platform.python_version(),'platform':platform.platform(),'torch':torch.__version__,'transformers':transformers.__version__,'numpy':numpy.__version__,'device':'cpu','dtype':'float32','timings':timings,'peak_rss_bytes':rss,'shapes':shapes});print(timings)
if __name__=='__main__':main()
