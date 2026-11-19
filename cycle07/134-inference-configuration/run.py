from common import *
def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['smoke','formal']);ap.add_argument('--cache',type=Path,default=Path('cache'));ap.add_argument('--archive',type=Path,default=Path('aclImdb_v1.tar.gz'));a=ap.parse_args()
 import torch,transformers
 from transformers import AutoTokenizer,AutoModelForSequenceClassification
 cfg=read('protocol.json');md=a.cache/cfg['model_revision'];start=time.monotonic()
 for f in read('resources.lock.json')['files']:assert h((md/f['name']).read_bytes())==f['sha256']
 pools=read('pools.json');assert h((ROOT/'pools.json').read_bytes())=='d472a0676f5f9b58aea434ce694ddd40ea2d5b4ab50f9c3aa28aec896d982764'
 rows=pools['exploration'][:8 if a.action=='smoke' else 1000];corpus=scan(a.archive,False,{x['id'] for x in rows})
 if a.action=='formal':assert sum(x['seconds'] for x in read('smoke_env.json')['conditions']) *125<1800
 torch.manual_seed(134);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
 tok=AutoTokenizer.from_pretrained(md,local_files_only=True)
 assert tok.truncation_side=='right'
 full={r['id']:tok(corpus[r['id']][0],add_special_tokens=False,truncation=False)['input_ids'] for r in rows};out=[];conditions=[]
 for condition in cfg['conditions']:
  length=512
  model=AutoModelForSequenceClassification.from_pretrained(md,local_files_only=True,attn_implementation=condition).eval()
  assert model.config._attn_implementation==condition
  t=time.monotonic();shapes=[];padded=0
  for j in range(0,len(rows),16):
   part=rows[j:j+16];xs=[corpus[r['id']][0] for r in part];assert all(h(norm(s).encode())==r['text'] for s,r in zip(xs,part))
   inp=tok(xs,padding=True,truncation=True,max_length=length,return_tensors='pt')
   with torch.inference_mode():z=model(**inp).logits
   assert z.shape==(len(part),2) and torch.isfinite(z).all();shapes.append(list(inp.input_ids.shape));padded+=inp.input_ids.numel()
   for r,v,ids,n in zip(part,z.tolist(),inp.input_ids.tolist(),inp.attention_mask.sum(1).tolist()):
    original=full[r['id']];assert ids[1:n-1]==original[:length-2]
    out.append(dict(id=r['id'],label=corpus[r['id']][1],condition=condition,length=length,full_tokens=len(original),retained_tokens=n,dropped=max(0,len(original)-(length-2)),input_sha256=h(json.dumps(ids[:n]).encode()),logits=v))
  conditions.append(dict(condition=condition,length=length,seconds=time.monotonic()-t,padded_tokens=padded,shapes=shapes));print(conditions[-1]['length'],conditions[-1]['seconds'],flush=True)
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
 put(a.action+'.json',out);put(a.action+'_env.json',dict(conditions=conditions,wall_s=time.monotonic()-start,peak_rss_bytes=rss,python=sys.version,torch=torch.__version__,transformers=transformers.__version__,platform=platform.platform(),cache_key=h(''.join(h((ROOT/n).read_bytes()) for n in ['run.py','common.py','pools.json','protocol.json','resources.lock.json']).encode())))
if __name__=='__main__':main()
