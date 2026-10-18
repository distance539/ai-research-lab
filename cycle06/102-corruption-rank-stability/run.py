"""Controlled GPT-2 head-output patching through official Transformers forward hooks."""
import argparse,json,time,platform,resource,sys,hashlib
from pathlib import Path
from prepare import sha,prepare
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
 # Last-position readout only. Core transformer is unchanged; a separate official full LM forward verifies this readout.
 def forward(x):return m.lm_head(m.transformer(x,use_cache=False).last_hidden_state[:,-1,:])
 def cache(x):
  z={};emb={};hs=[]
  for l,block in enumerate(m.transformer.h):
   def capture(mod,args,l=l):z[l]=args[0].detach().clone()
   hs.append(block.attn.c_proj.register_forward_pre_hook(capture))
  hs.append(m.transformer.drop.register_forward_hook(lambda mod,args,y:emb.update(x=y.detach().clone())))
  try:v=forward(x)
  finally:
   for h in hs:h.remove()
  return v,z,emb['x']
 def measure(v,io,s):
  return {'io_logit':v[io].item(),'s_logit':v[s].item(),'diff':(v[io]-v[s]).item(),'p_io':v.softmax(-1)[io].item(),'top1_id':int(v.argmax())}
 def patch(x,z,l,indices):
  # Each batch row is one isolated intervention on the same corrupted sentence.
  def hook(mod,args):
   y=args[0].clone().view(len(indices),x.shape[1],12,64)
   for j,(p,h) in enumerate(indices):y[j,p,h]=z[l].view(1,x.shape[1],12,64)[0,p,h]
   return (y.reshape(len(indices),x.shape[1],768),)
  handle=m.transformer.h[l].attn.c_proj.register_forward_pre_hook(hook)
  try:return forward(x.expand(len(indices),-1))
  finally:handle.remove()
 records=[];patches=[];setup=time.perf_counter()-t
 with torch.inference_mode():
  for row,condition in [(r,k) for r in rows for k in cfg['corruption']]:
   rt=time.perf_counter();ids=tok.encode(row['prompt'],add_special_tokens=False);io=tok.encode(row['answer'],add_special_tokens=False);s=tok.encode(row['distractor'],add_special_tokens=False)
   assert len(io)==len(s)==1;io,s=io[0],s[0];assert ids.count(io)==1 and ids.count(s)==2
   assert tok.encode(row['prompt']+row['answer'],add_special_tokens=False)==ids+[io]
   assert tok.encode(row['prompt']+row['distractor'],add_special_tokens=False)==ids+[s]
   p2=[i for i,q in enumerate(ids) if q==s][-1];bad=ids.copy()
   if condition=='s2_flip':bad[p2]=io
   elif condition=='global_swap':bad=[s if q==io else io if q==s else q for q in ids]
   else:raise ValueError(condition)
   changed=[i for i,(u,v) in enumerate(zip(ids,bad)) if u!=v]
   assert len(ids)<=cfg['max_tokens'] and len(changed)==(1 if condition=='s2_flip' else 3)
   assert tok.encode(tok.decode(bad),add_special_tokens=False)==bad
   x=torch.tensor([ids]);xc=torch.tensor([bad]);clean,cz,ce=cache(x);corrupt,bz,be=cache(xc)
   official=m(x,use_cache=False).logits[0,-1];readout_error=(official-clean[0]).abs().max().item();assert readout_error<0.001
   h=m.transformer.drop.register_forward_hook(lambda mod,args,y:ce.clone())
   try:full=forward(xc)
   finally:h.remove()
   full_error=(full-clean).abs().max().item();assert full_error==0
   hs=[]
   for l,b in enumerate(m.transformer.h):hs.append(b.attn.c_proj.register_forward_pre_hook(lambda mod,args,l=l:(bz[l].clone(),)))
   try:selfv=forward(xc)
   finally:
    for h in hs:h.remove()
   self_error=(selfv-corrupt).abs().max().item();assert self_error==0
   # Independent algebraic implementation at a preselected node: projected delta, rather than a preprojection overwrite.
   l,h,p=9,9,len(ids)-1;sl=slice(h*64,(h+1)*64)
   def projected(mod,args,y):
    q=y.clone();q[0,p]+=(cz[l][0,p,sl]-args[0][0,p,sl])@mod.weight[sl,:];return q
   handle=m.transformer.h[l].attn.c_proj.register_forward_hook(projected)
   try:alt=forward(xc)
   finally:handle.remove()
   direct=patch(xc,cz,l,[(p,h)]);alt_error=(alt-direct).abs().max().item();assert alt_error<0.001
   # Identical batch shape for baselines used by all scan rows; tolerate only floating-point implementation differences.
   bs=cfg['batch_size'];cb=forward(x.expand(bs,-1));bb=forward(xc.expand(bs,-1));batcherr=max((cb-clean).abs().max().item(),(bb-corrupt).abs().max().item());assert batcherr<0.001
   r=dict(row,condition=condition,changed_positions=changed,clean_ids=ids,corrupt_ids=bad,corrupt_prompt=tok.decode(bad),tokens=tok.convert_ids_to_tokens(ids),io_id=io,s_id=s,s2_position=p2,io_position=ids.index(io),end_position=len(ids)-1,
    clean=measure(clean[0],io,s),corrupt=measure(corrupt[0],io,s),full=measure(full[0],io,s),self=measure(selfv[0],io,s),batch_clean=measure(cb[0],io,s),batch_corrupt=measure(bb[0],io,s),
    official_readout_max_error=readout_error,full_max_error=full_error,self_max_error=self_error,projected_delta_max_error=alt_error,batch_max_error=batcherr,z_shape=[1,len(ids),12,64],cache_bytes=sum(z.numel()*z.element_size() for z in cz.values())+ce.numel()*ce.element_size())
   for l in range(12):
    grid=[(p,h) for p in range(len(ids)) for h in range(12)]
    for start in range(0,len(grid),bs):
     inds=grid[start:start+bs];actual=len(inds)
     # Pad the final intervention batch so all scan rows have the same numerical batch shape.
     padded=inds+[inds[-1]]*(bs-actual);vs=patch(xc,cz,l,padded)
     for (pos,head),v in zip(inds,vs[:actual]):
      q=measure(v,io,s);q.update(id=row['id'],condition=condition,layer=l,head=head,position=pos,gain=q['diff']-r['batch_corrupt']['diff']);patches.append(q)
     assert time.perf_counter()-t<cfg['budget_seconds'],'time budget'
    print('row',row['id'],condition,'layer',l,'seconds',round(time.perf_counter()-rt,3),flush=True)
   r['seconds']=time.perf_counter()-rt;records.append(r)
   write(out/'records.json',records);write(out/'patches.json',patches)
   stages['preprocessing']='passed';stages['inference']='running';write(out/'stages.json',stages)
 elapsed=time.perf_counter()-t;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024);assert rss<cfg['budget_rss_bytes']
 write(out/'runtime.json',{'elapsed_seconds_after_imports':elapsed,'setup_seconds':setup,'peak_rss_bytes':rss,'projected_full_seconds':setup+(elapsed-setup)*16/len(rows),'rows':len(records),'samples':len(rows),'patches':len(patches)})
 if a.smoke:assert setup+(elapsed-setup)*16/len(rows)<cfg['budget_seconds'],'smoke resource projection'
 write(out/'environment.json',{'os':platform.platform(),'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__,'numpy':numpy.__version__,'device':'cpu','dtype':'float32','threads':4,'parameters':sum(p.numel() for p in m.parameters()),'batch_size':cfg['batch_size']})
 write(out/'identity.json',{'plan_sha256':sha(ROOT/'plan.json'),'resources_sha256':sha(ROOT/'resources.lock.json'),'mode':'smoke' if a.smoke else 'formal','sample_ids':[r['id'] for r in rows],'cache_key':hashlib.sha256((sha(ROOT/'plan.json')+sha(ROOT/'resources.lock.json')+str(a.smoke)).encode()).hexdigest(),'reuse_output_cache':False})
 from analyze import analyze
 write(out/'summary.json',analyze(records,patches));stages.update(inference='passed',evaluation='passed');write(out/'stages.json',stages);print(json.dumps(json.loads((out/'runtime.json').read_text())),flush=True)
if __name__=='__main__':main()
