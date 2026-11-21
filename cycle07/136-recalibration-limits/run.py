"""Run from this directory. Register three pools before any model forward."""
import argparse, hashlib, html, json, math, os, platform, re, resource, sys, tarfile, time, unicodedata
from pathlib import Path
from urllib.request import urlretrieve
ROOT=Path(__file__).resolve().parent
SHA='c40f74a18d3b61f90feba1e17730e0d38e8b97c05fde7008942e91923d1658fe'
MIRROR='https://huggingface.co/datasets/NolanChai/aclImdb_v1/resolve/ab375201f670381f90723112b08fb578776f542a/aclImdb_v1.tar.gz'
def h(b): return hashlib.sha256(b).hexdigest()
def read(n): return json.loads((ROOT/n).read_bytes())
def put(n,v): (ROOT/n).write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def freeze(n,v):
 if (ROOT/n).exists(): assert read(n)==v, 'Existing lock mismatch'
 else: put(n,v)
def norm(s): return ' '.join(unicodedata.normalize('NFKC',html.unescape(re.sub(r'<br\s*/?>',' ',s,flags=re.I))).casefold().split())
def scan(archive,register,allow=None):
 assert h(archive.read_bytes())==SHA
 pool={}; urls={}
 with tarfile.open(archive,'r|gz') as tf:
  for m in tf:
   match=re.fullmatch(r'aclImdb/test/(pos|neg)/(\d+)_(\d+)\.txt',m.name)
   if register and m.name in ['aclImdb/test/urls_neg.txt','aclImdb/test/urls_pos.txt']:
    urls[m.name[-7:-4]]=tf.extractfile(m).read().decode().splitlines()
   if not match: continue
   key=h(m.name.encode())
   if not register and key not in allow: continue
   raw=tf.extractfile(m).read().decode()
   pol,idx,rating=match.groups()
   if register: pool[key]=(pol,int(idx),h(norm(raw).encode()))
   else: pool[key]=(raw,int(pol=='pos'))
 if not register: return pool
 assert len(pool)==25000
 movies=set(); texts=set(); selected=[]
 for key in sorted(pool,key=lambda k:h(('cycle7:132:'+k).encode())):
  pol,idx,dig=pool[key]; movie=re.search(r'/title/(tt\d+)',urls[pol][idx]).group(1)
  if movie in movies or dig in texts: continue
  movies.add(movie);texts.add(dig);selected.append(dict(id=key,movie=h(movie.encode()),text=dig))
 assert len(selected)>=3000
 return {k:selected[i*1000:(i+1)*1000] for i,k in enumerate(('exploration','calibration','confirmation'))}
def score(rows,t):
 bins=[[0,0,0.] for _ in range(10)];hits=accepted=errors=0;loss=brier=0
 for r in rows:
  a,b=r['logits']; y=r['label']; diff=(b-a)/t;p=1/(1+math.exp(-diff)); c=max(p,1-p); ok=int((diff>0)==y)
  hits+=ok; loss+=math.log1p(math.exp(diff)) - y*diff;brier+=2*(p-y)**2
  v=bins[min(9,int(10*c))];v[0]+=1;v[1]+=ok;v[2]+=c
  if c>=.9:accepted+=1;errors+=1-ok
 n=len(rows)
 return dict(n=n,hits=hits,nll=loss/n,brier=brier/n,ece=sum(abs(k-s) for _,k,s in bins)/n,bins=bins,accepted=accepted,errors=errors)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['fetch','smoke','formal']);ap.add_argument('--archive',type=Path,default=Path('aclImdb_v1.tar.gz'));ap.add_argument('--cache',type=Path,default=Path('cache'));a=ap.parse_args()
 cfg=read('protocol.json'); lock=read('resources.lock.json');md=a.cache/cfg['model_revision'];start=time.monotonic()
 if a.action=='fetch':
  md.mkdir(parents=True,exist_ok=True)
  for dest,url,sha in [(a.archive,MIRROR,SHA)]+[(md/i['name'],i['url'],i['sha256']) for i in lock['files']]:
   if not dest.exists():urlretrieve(url,dest)
   assert h(dest.read_bytes())==sha
 elif a.action=='register':
  raise RuntimeError('Pools are frozen; do not re-register')
  pools=scan(a.archive,True);freeze('pools.json',pools);print('pool sizes',*[len(v) for v in pools.values()],'lock',h((ROOT/'pools.json').read_bytes()))
 elif a.action=='analyze':
  out={}
  for domain,name in [('SST2','audit_predictions.json'),('IMDb','formal.json')]:
   out[domain]={k:score(read(name),t) for k,t in [('raw',1),('frozen',cfg['fixed_T'])]}
  put('results.json',out);print({k:[v[x]['nll'] for x in v] for k,v in out.items()})
 else:
  import torch,transformers
  from transformers import AutoTokenizer,AutoModelForSequenceClassification
  for item in lock['files']:assert h((md/item['name']).read_bytes())==item['sha256']
  pools=read('pools.json')
  for field in ('id','movie','text'):assert len({x[field] for group in pools.values() for x in group})==3000
  rows=pools['exploration'][:8 if a.action=='smoke' else 1000]; corpus=scan(a.archive,False,{x['id'] for x in rows});assert len(corpus)==len(rows)
  if a.action=='formal':
   smoke=read('smoke_env.json');assert smoke['forward_s']*125<1800 and smoke['peak_rss_bytes']<6*2**30
  torch.manual_seed(cfg['seed']);torch.set_num_threads(cfg['threads']);torch.use_deterministic_algorithms(True)
  tok=AutoTokenizer.from_pretrained(md,local_files_only=True);model=AutoModelForSequenceClassification.from_pretrained(md,local_files_only=True,attn_implementation='eager').eval()
  assert model.config.id2label=={0:'NEGATIVE',1:'POSITIVE'} and tok.truncation_side=='right'
  out=[];shapes=[];t=time.monotonic()
  for j in range(0,len(rows),16):
   part=rows[j:j+16];xs=[corpus[r['id']][0] for r in part];assert all(h(norm(s).encode())==r['text'] for s,r in zip(xs,part))
   inp=tok(xs,padding=True,truncation=True,max_length=512,return_tensors='pt')
   with torch.inference_mode():z=model(**inp).logits
   assert z.shape==(len(xs),2) and torch.isfinite(z).all()
   shapes.append([list(inp.input_ids.shape),list(z.shape)])
   out.extend(dict(id=r['id'],label=corpus[r['id']][1],retained_tokens=l,logits=v) for r,v,l in zip(part,z.tolist(),inp.attention_mask.sum(1).tolist()))
  key=h(''.join([SHA,*[h((ROOT/n).read_bytes()) for n in ('run.py','resources.lock.json','protocol.json','pools.json')],a.action]).encode())
  rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
  put(a.action+'.json',out);put(a.action+'_env.json',dict(shapes=shapes,forward_s=time.monotonic()-t,wall_s=time.monotonic()-start,peak_rss_bytes=rss,cache_key=key,python=sys.version,platform=platform.platform(),torch=torch.__version__,transformers=transformers.__version__));print(a.action,len(out),'seconds',time.monotonic()-t,'peak RSS',rss)
if __name__=='__main__': main()
