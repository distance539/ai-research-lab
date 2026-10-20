"""Frozen-head transfer through official complete GPT2LMHeadModel forward."""
import argparse,json,time,platform,resource,sys,hashlib
from pathlib import Path
from prepare import prepare,sha
ROOT=Path(__file__).resolve().parent
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cache',required=True);ap.add_argument('--out',required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=False);t=time.perf_counter()
 stages=dict(download='pending',preprocessing='pending',inference='pending',evaluation='pending',training='not applicable');write(out/'stages.json',stages)
 cfg=json.loads((ROOT/'config.json').read_text());plan=json.loads((ROOT/'plan.json').read_text())
 for f,h in plan['files'].items():assert sha(ROOT/f)==h,f
 prepare(a.cache);stages['download']='passed';write(out/'stages.json',stages)
 import torch,transformers,numpy
 from transformers import AutoTokenizer,AutoModelForCausalLM
 torch.set_num_threads(cfg['threads']);torch.manual_seed(cfg['seed']);torch.use_deterministic_algorithms(True)
 tok=AutoTokenizer.from_pretrained(a.cache,local_files_only=True)
 m=AutoModelForCausalLM.from_pretrained(a.cache,local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval()
 rows=json.loads((ROOT/'samples.json').read_text());rows=[r for r in rows if r['id'] in cfg['smoke_ids']] if a.smoke else rows
 heads=cfg['heads'];layers=sorted({l for l,h in heads.values()});records=[]
 def forward(x):return m(x,use_cache=False).logits[:, -1, :]
 def cache(x):
  z={};hs=[]
  for l in layers:
   def capture(mod,args,l=l):z[l]=args[0].detach().clone()
   hs.append(m.transformer.h[l].attn.c_proj.register_forward_pre_hook(capture))
  try:v=forward(x)
  finally:
   for h in hs:h.remove()
  return v,{k:z[l][0,-1,h*64:(h+1)*64].clone() for k,(l,h) in heads.items()}
 def intervene(x,repl,projected=False):
  hs=[]
  for l in layers:
   keys=[k for k in repl if heads[k][0]==l]
   if not keys:continue
   if projected:
    def hook(mod,args,y,keys=keys):
     q=y.clone()
     for k in keys:
      h=heads[k][1];sl=slice(h*64,(h+1)*64);q[0,-1]+=(repl[k]-args[0][0,-1,sl])@mod.weight[sl,:]
     return q
    hs.append(m.transformer.h[l].attn.c_proj.register_forward_hook(hook))
   else:
    def hook(mod,args,keys=keys):
     q=args[0].clone();mask=torch.zeros_like(q,dtype=torch.bool)
     for k in keys:
      h=heads[k][1];sl=slice(h*64,(h+1)*64);q[0,-1,sl]=repl[k];mask[0,-1,sl]=True
     assert torch.equal(q[~mask],args[0][~mask]);return (q,)
    hs.append(m.transformer.h[l].attn.c_proj.register_forward_pre_hook(hook))
  try:return forward(x)
  finally:
   for h in hs:h.remove()
 def measure(v,io,s):
  il,sl=v[0,io].item(),v[0,s].item()
  return dict(io_logit=il,s_logit=sl,diff=il-sl,p_io=v.softmax(-1)[0,io].item(),top1_id=int(v[0].argmax()))
 setup=time.perf_counter()-t
 with torch.inference_mode():
  for row in rows:
   rt=time.perf_counter();ids=tok.encode(row['prompt'],add_special_tokens=False);io=tok.encode(row['answer'],add_special_tokens=False);s=tok.encode(row['distractor'],add_special_tokens=False)
   assert len(io)==len(s)==1;io,s=io[0],s[0];assert ids.count(io)==1 and ids.count(s)==2 and len(ids)<=cfg['max_tokens']
   for suffix,q in [(row['answer'],io),(row['distractor'],s)]:assert tok.encode(row['prompt']+suffix,add_special_tokens=False)==ids+[q]
   pos=[j for j,q in enumerate(ids) if q==s][-1];ci=ids.copy();ci[pos]=io
   assert tok.encode(tok.decode(ci),add_special_tokens=False)==ci
   x=torch.tensor([ids]);cx=torch.tensor([ci]);v,z=cache(x);cv,cz=cache(cx)
   zero={k:torch.zeros_like(q) for k,q in z.items()};zv=intervene(x,zero);rv=intervene(cx,z)
   selferr=(intervene(x,z)-v).abs().max().item();cerr=(intervene(cx,cz)-cv).abs().max().item();empty=(intervene(x,{})-v).abs().max().item();assert selferr==cerr==empty==0
   pezero=(intervene(x,zero,True)-zv).abs().max().item();perestore=(intervene(cx,z,True)-rv).abs().max().item();assert max(pezero,perestore)<0.001
   r=dict(row,ids=ids,corrupt_ids=ci,corrupt_prompt=tok.decode(ci),io_id=io,s_id=s,s2_position=pos,end_position=len(ids)-1,z_shape=[1,len(ids),12,64],logits_shape=[1,len(ids),50257],clean=measure(v,io,s),zero=measure(zv,io,s),corrupt=measure(cv,io,s),restored=measure(rv,io,s),clean_z={k:q.tolist() for k,q in z.items()},controls=dict(self_error=selferr,corrupt_self_error=cerr,empty_error=empty,projected_zero_error=pezero,projected_restore_error=perestore),seconds=time.perf_counter()-rt)
   records.append(r)
   if len(records)%16==0 or a.smoke:print(row['group'],len(records),'rows',time.perf_counter()-t,flush=True);write(out/'records.json',records)
   assert time.perf_counter()-t<cfg['budget_seconds'],'time budget'
 elapsed=time.perf_counter()-t;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024);assert rss<cfg['budget_rss_bytes']
 runtime=dict(elapsed_seconds_after_imports=elapsed,setup_seconds=setup,peak_rss_bytes=rss,projected_full_seconds=setup+(elapsed-setup)*256/len(rows),rows=len(rows),target_conditions=len(rows)*4,control_forwards=len(rows)*5)
 if a.smoke:assert runtime['projected_full_seconds']<cfg['budget_seconds']
 write(out/'runtime.json',runtime);write(out/'records.json',records)
 write(out/'environment.json',dict(os=platform.platform(),python=sys.version,torch=torch.__version__,transformers=transformers.__version__,numpy=numpy.__version__,device='cpu',dtype='float32',threads=cfg['threads'],parameters=sum(p.numel() for p in m.parameters()),batch_size=1))
 write(out/'identity.json',dict(plan_sha256=sha(ROOT/'plan.json'),resources_sha256=sha(ROOT/'resources.lock.json'),mode='smoke' if a.smoke else 'formal',sample_ids=[r['id'] for r in rows],cache_key=hashlib.sha256((sha(ROOT/'plan.json')+sha(ROOT/'resources.lock.json')+str(a.smoke)).encode()).hexdigest(),reuse_output_cache=False))
 from analyze import analyze
 write(out/'summary.json',analyze(records));stages.update(preprocessing='passed',inference='passed',evaluation='passed');write(out/'stages.json',stages);print(json.dumps(runtime),flush=True)
if __name__=='__main__':main()
