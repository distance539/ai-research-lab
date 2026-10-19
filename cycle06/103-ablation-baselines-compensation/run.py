"""Real GPT-2 END-only factorial ablation; official Transformers forward."""
import argparse,json,time,platform,resource,sys,hashlib,itertools
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
 allrows=json.loads((ROOT/'samples.json').read_text());rows=[r for r in allrows if r['id'] in cfg['smoke_ids']] if a.smoke else allrows
 heads=cfg['heads'];layers=sorted(set(l for l,h in heads.values()))
 def forward(x):return m.lm_head(m.transformer(x,use_cache=False).last_hidden_state[:,-1,:])
 def cache(x):
  z={};hs=[]
  for l in layers:
   def capture(mod,args,l=l):z[l]=args[0].detach().clone()
   hs.append(m.transformer.h[l].attn.c_proj.register_forward_pre_hook(capture))
  try:v=forward(x)
  finally:
   for h in hs:h.remove()
  return v,{k:z[l][0,-1,h*64:(h+1)*64].clone() for k,(l,h) in heads.items()}
 def measure(v,io,s):
  return {'io_logit':v[0,io].item(),'s_logit':v[0,s].item(),'diff':(v[0,io]-v[0,s]).item(),'p_io':v.softmax(-1)[0,io].item(),'top1_id':int(v[0].argmax())}
 def intervene(x,replacements,subset,projected=False):
  hs=[]
  for l in layers:
   keys=[k for k in subset if heads[k][0]==l]
   if not keys:continue
   if projected:
    def hook(mod,args,y,keys=keys):
     q=y.clone()
     for k in keys:
      h=heads[k][1];sl=slice(h*64,(h+1)*64)
      q[0,-1]+=(replacements[k]-args[0][0,-1,sl])@mod.weight[sl,:]
     return q
    hs.append(m.transformer.h[l].attn.c_proj.register_forward_hook(hook))
   else:
    def hook(mod,args,keys=keys):
     q=args[0].clone();mask=torch.zeros_like(q,dtype=torch.bool)
     for k in keys:
      h=heads[k][1];sl=slice(h*64,(h+1)*64);q[0,-1,sl]=replacements[k];mask[0,-1,sl]=True
     assert torch.equal(q[~mask],args[0][~mask]);return (q,)
    hs.append(m.transformer.h[l].attn.c_proj.register_forward_pre_hook(hook))
  try:return forward(x)
  finally:
   for h in hs:h.remove()
 bank={};bank_meta=[];records=[]
 with torch.inference_mode():
  # Build bank from public template structure, without loading heldout samples.
  for ti in cfg['template_indices']:
   row=next(r for r in allrows if r['template_index']==ti);ids=tok.encode(row['prompt'],add_special_tokens=False)
   nameids=[tok.encode(' '+n,add_special_tokens=False) for n in ['Michael','Christopher','Jessica']];assert all(len(q)==1 for q in nameids);nameids=[q[0] for q in nameids]
   positions=[j for j,q in enumerate(ids) if q in nameids];assert len(positions)==3
   bank[ti]=[]
   for ix,perm in enumerate(itertools.permutations(nameids)):
    ref=ids.copy()
    for p,n in zip(positions,perm):ref[p]=n
    assert len(set(ref[p] for p in positions))==3 and tok.encode(tok.decode(ref),add_special_tokens=False)==ref
    _,z=cache(torch.tensor([ref]));bank[ti].append(z)
    bank_meta.append({'template_index':ti,'donor_index':ix,'ids':ref,'prompt':tok.decode(ref),'name_positions':positions,'z':{k:v.tolist() for k,v in z.items()}})
  write(out/'reference_bank.json',bank_meta);setup=time.perf_counter()-t
  for row in rows:
   rt=time.perf_counter();ids=tok.encode(row['prompt'],add_special_tokens=False);io=tok.encode(row['answer'],add_special_tokens=False);s=tok.encode(row['distractor'],add_special_tokens=False)
   assert len(io)==len(s)==1;io,s=io[0],s[0];assert ids.count(io)==1 and ids.count(s)==2
   assert tok.encode(row['prompt']+row['answer'],add_special_tokens=False)==ids+[io]
   assert tok.encode(row['prompt']+row['distractor'],add_special_tokens=False)==ids+[s]
   assert len(ids)<=cfg['max_tokens'];x=torch.tensor([ids]);v,z=cache(x)
   err=(m(x,use_cache=False).logits[:,-1]-v).abs().max().item();assert err<0.001
   selferr=(intervene(x,z,'ABC')-v).abs().max().item();emptyerr=(intervene(x,z,'')-v).abs().max().item();assert selferr==emptyerr==0
   refs=bank[row['template_index']];di=cfg['donor_indices'][str(row['id'])]
   repl={'zero':{k:torch.zeros_like(q) for k,q in z.items()},'mean_abc':{k:torch.stack([q[k] for q in refs]).mean(0) for k in heads},'resample_abc':refs[di]}
   r=dict(row,ids=ids,io_id=io,s_id=s,end_position=len(ids)-1,clean=measure(v,io,s),z_shape=[1,len(ids),12,64],donor_index=di,clean_z={k:q.tolist() for k,q in z.items()},replacement_z={mode:{k:q.tolist() for k,q in zz.items()} for mode,zz in repl.items()},diagnostics={},interventions={},controls={'official_readout_error':err,'self_error':selferr,'empty_error':emptyerr,'projected_errors':{}})
   for mode in cfg['modes']:
    zz=repl[mode];r['interventions'][mode]={};r['diagnostics'][mode]={k:{'clean_norm':z[k].norm().item(),'replacement_norm':zz[k].norm().item(),'distance_to_clean':(zz[k]-z[k]).norm().item()} for k in heads}
    for subset in cfg['subsets']:
     q=intervene(x,zz,subset);r['interventions'][mode][subset]=measure(q,io,s)
     if subset=='ABC':
      e=(q-intervene(x,zz,subset,True)).abs().max().item();assert e<0.001;r['controls']['projected_errors'][mode]=e
    assert time.perf_counter()-t<cfg['budget_seconds'],'time budget'
   r['seconds']=time.perf_counter()-rt;records.append(r);write(out/'records.json',records)
   stages.update(preprocessing='passed',inference='running');write(out/'stages.json',stages);print('row',row['id'],'21 interventions',r['seconds'],flush=True)
 elapsed=time.perf_counter()-t;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024);assert rss<cfg['budget_rss_bytes']
 runtime={'elapsed_seconds_after_imports':elapsed,'setup_seconds':setup,'peak_rss_bytes':rss,'projected_full_seconds':setup+(elapsed-setup)*16/len(rows),'rows':len(rows),'interventions':21*len(rows),'reference_forwards':12}
 write(out/'runtime.json',runtime)
 if a.smoke:assert runtime['projected_full_seconds']<cfg['budget_seconds']
 write(out/'environment.json',{'os':platform.platform(),'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__,'numpy':numpy.__version__,'device':'cpu','dtype':'float32','threads':4,'parameters':sum(p.numel() for p in m.parameters()),'batch_size':1})
 write(out/'identity.json',{'plan_sha256':sha(ROOT/'plan.json'),'resources_sha256':sha(ROOT/'resources.lock.json'),'mode':'smoke' if a.smoke else 'formal','sample_ids':[r['id'] for r in rows],'cache_key':hashlib.sha256((sha(ROOT/'plan.json')+sha(ROOT/'resources.lock.json')+str(a.smoke)).encode()).hexdigest(),'reuse_output_cache':False})
 from analyze import analyze
 write(out/'summary.json',analyze(records));stages.update(inference='passed',evaluation='passed');write(out/'stages.json',stages);print(json.dumps(runtime),flush=True)
if __name__=='__main__':main()
