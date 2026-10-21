"""Local faithfulness/completeness of three END nodes, official GPT-2 forward."""
import argparse,json,time,platform,resource,sys,hashlib,itertools
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
 allrows=json.loads((ROOT/'samples.json').read_text());rows=[r for r in allrows if r['id'] in cfg['smoke_ids']] if a.smoke else allrows
 universe={(l,h) for l in range(12) for h in range(12)};heads={k:tuple(v) for k,v in cfg['heads'].items()};C=set(heads.values())
 def forward(x):return m(x,use_cache=False).logits[:, -1, :]
 def capture(x):
  z={};hooks=[]
  for l in range(12):
   def hook(mod,args,l=l):z[l]=args[0][0,-1].clone()
   hooks.append(m.transformer.h[l].attn.c_proj.register_forward_pre_hook(hook))
  try:v=forward(x)
  finally:
   for h in hooks:h.remove()
  return v,torch.stack([z[l] for l in range(12)])
 def intervene(x,removed,replacements,projected=False):
  hooks=[]
  for l in range(12):
   hs=sorted(h for ll,h in removed if ll==l)
   if not hs:continue
   if projected:
    def hook(mod,args,y,l=l,hs=hs):
     delta=torch.zeros_like(args[0])
     for h in hs:delta[0,-1,h*64:(h+1)*64]=replacements[l,h*64:(h+1)*64]-args[0][0,-1,h*64:(h+1)*64]
     return y+delta@mod.weight
    hooks.append(m.transformer.h[l].attn.c_proj.register_forward_hook(hook))
   else:
    def hook(mod,args,l=l,hs=hs):
     q=args[0].clone();mask=torch.zeros_like(q,dtype=torch.bool)
     for h in hs:q[0,-1,h*64:(h+1)*64]=replacements[l,h*64:(h+1)*64];mask[0,-1,h*64:(h+1)*64]=True
     assert torch.equal(q[~mask],args[0][~mask]);return(q,)
    hooks.append(m.transformer.h[l].attn.c_proj.register_forward_pre_hook(hook))
  try:return forward(x)
  finally:
   for h in hooks:h.remove()
 def measure(v,io,s):
  il,sl=v[0,io].item(),v[0,s].item()
  return dict(io_logit=il,s_logit=sl,diff=il-sl,p_io=v.softmax(-1)[0,io].item(),top1_id=int(v[0].argmax()))
 bank={};refs=[];records=[]
 with torch.inference_mode():
  for ti in range(4):
   row=next(r for r in allrows if r['template_index']==ti);ids=tok.encode(row['prompt'],add_special_tokens=False)
   candidate_ids=[tok.encode(row[k],add_special_tokens=False)[0] for k in ['answer','distractor']]
   positions=[i for i,v in enumerate(ids) if v in candidate_ids];assert len(positions)==3
   names=[tok.encode(' '+n,add_special_tokens=False) for n in cfg['reference_names']];assert all(len(q)==1 for q in names)
   zs=[]
   for perm in itertools.permutations([q[0] for q in names]):
    xx=ids.copy()
    for pos,val in zip(positions,perm):xx[pos]=val
    assert len(set(xx[pos] for pos in positions))==3 and tok.encode(tok.decode(xx),add_special_tokens=False)==xx
    _,z=capture(torch.tensor([xx]));zs.append(z);refs.append(dict(template_index=ti,ids=xx,prompt=tok.decode(xx),name_positions=positions))
   bank[ti]=torch.stack(zs).mean(0)
  write(out/'references.json',refs);write(out/'means.json',{str(k):v.tolist() for k,v in bank.items()});setup=time.perf_counter()-t
  stages['preprocessing']='passed';stages['inference']='running';write(out/'stages.json',stages)
  for row in rows:
   rt=time.perf_counter();ids=tok.encode(row['prompt'],add_special_tokens=False);io=tok.encode(row['answer'],add_special_tokens=False);s=tok.encode(row['distractor'],add_special_tokens=False)
   assert len(io)==len(s)==1;io,s=io[0],s[0];assert ids.count(io)==1 and ids.count(s)==2 and len(ids)<=cfg['max_tokens']
   for suffix,q in [(row['answer'],io),(row['distractor'],s)]:assert tok.encode(row['prompt']+suffix,add_special_tokens=False)==ids+[q]
   x=torch.tensor([ids]);v,z=capture(x)
   se=(intervene(x,universe,z)-v).abs().max().item();ee=(intervene(x,set(),z)-v).abs().max().item();assert se==ee==0
   r=dict(row,ids=ids,io_id=io,s_id=s,end_position=len(ids)-1,z_shape=[1,len(ids),12,64],logits_shape=[1,len(ids),50257],clean=measure(v,io,s),conditions={},controls=dict(all_head_self_error=se,empty_error=ee,projected_errors={}))
   for mode in cfg['modes']:
    repl=torch.zeros_like(z) if mode=='zero' else bank[row['template_index']];r['conditions'][mode]={'full':{},'retained':{}}
    for subset in cfg['subsets']:
     K={heads[k] for k in subset}
     for system in ['full','retained']:
      removed=K if system=='full' else (universe-C)|K
      q=v if not removed else intervene(x,removed,repl)
      r['conditions'][mode][system][subset]=measure(q,io,s)
      if system=='retained' and subset=='':
       pe=(q-intervene(x,removed,repl,True)).abs().max().item();assert pe<.001
       r['controls']['projected_errors'][mode]=pe
   r['seconds']=time.perf_counter()-rt;records.append(r)
   if len(records)%16==0 or a.smoke:write(out/'records.json',records);print('rows',len(records),'elapsed',time.perf_counter()-t,flush=True)
   assert time.perf_counter()-t<cfg['budget_seconds'],'time budget'
 elapsed=time.perf_counter()-t;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024);assert rss<cfg['budget_rss_bytes']
 runtime=dict(elapsed_seconds_after_imports=elapsed,setup_seconds=setup,peak_rss_bytes=rss,projected_full_seconds=setup+(elapsed-setup)*256/len(rows),rows=len(rows),unique_scored_forwards=len(rows)*31,control_forwards=len(rows)*4,reference_forwards=24)
 if a.smoke:assert runtime['projected_full_seconds']<cfg['budget_seconds']
 write(out/'runtime.json',runtime);write(out/'records.json',records)
 write(out/'environment.json',dict(os=platform.platform(),python=sys.version,torch=torch.__version__,transformers=transformers.__version__,numpy=numpy.__version__,device='cpu',dtype='float32',threads=cfg['threads'],parameters=sum(p.numel() for p in m.parameters()),batch_size=1))
 write(out/'identity.json',dict(plan_sha256=sha(ROOT/'plan.json'),resources_sha256=sha(ROOT/'resources.lock.json'),mode='smoke' if a.smoke else 'formal',sample_ids=[r['id'] for r in rows],cache_key=hashlib.sha256((sha(ROOT/'plan.json')+sha(ROOT/'resources.lock.json')+str(a.smoke)).encode()).hexdigest(),reuse_output_cache=False))
 from analyze import analyze
 write(out/'summary.json',analyze(records));stages.update(inference='passed',evaluation='passed');write(out/'stages.json',stages);print(json.dumps(runtime),flush=True)
if __name__=='__main__':main()
