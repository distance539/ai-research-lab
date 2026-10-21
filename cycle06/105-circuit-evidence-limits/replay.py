"""Independent dense postprojection replay of every saved model condition."""
import argparse,json,time,hashlib,itertools
from pathlib import Path
from prepare import prepare

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cache',required=True);ap.add_argument('--results',default='results');ap.add_argument('--out',required=True);a=ap.parse_args()
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 prepare(a.cache);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
 m=AutoModelForCausalLM.from_pretrained(a.cache,local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval();tok=AutoTokenizer.from_pretrained(a.cache,local_files_only=True)
 p=Path(a.results);rows=json.loads((p/'records.json').read_text());saved_means=json.loads((p/'means.json').read_text());refs=json.loads((p/'references.json').read_text());ABC=[(8,10),(8,6),(9,9)];t=time.perf_counter();err=0;perr=0;n=0;meanerr=0
 with torch.inference_mode():
  # Recompute bank independently in float64 after collecting fp32 z.
  means={}
  for ti in range(4):
   zs=[]
   for r in refs:
    if r['template_index']!=ti:continue
    assert tok.encode(r['prompt'],add_special_tokens=False)==r['ids'];z={};handles=[]
    for l in range(12):
     def capture(mod,args,l=l):z[l]=args[0][0,-1].double().clone()
     handles.append(m.transformer.h[l].attn.c_proj.register_forward_pre_hook(capture))
    m(torch.tensor([r['ids']]),use_cache=False)
    for h in handles:h.remove()
    zs.append(torch.stack([z[l] for l in range(12)]))
   check=torch.stack(zs).mean(0).float();saved=torch.tensor(saved_means[str(ti)]);e=(check-saved).abs().max().item();meanerr=max(meanerr,e);assert e<.00001;means[ti]=saved
  for row in rows:
   ids=tok.encode(row['prompt'],add_special_tokens=False);assert ids==row['ids'];assert tok.encode(row['answer'],add_special_tokens=False)==[row['io_id']] and tok.encode(row['distractor'],add_special_tokens=False)==[row['s_id']]
   tests=[('clean','full','',row['clean'])]
   for mode in ['zero','mean_abc']:
    for system in ['full','retained']:
     for k,record in row['conditions'][mode][system].items():
      if system=='full' and not k:continue
      tests.append((mode,system,k,record))
   for mode,system,k,record in tests:
    handles=[];removed={ABC['ABC'.index(ch)] for ch in k}
    if system=='retained':removed|={(l,h) for l in range(12) for h in range(12)}-set(ABC)
    for l in range(12):
     selected=[h for ll,h in removed if ll==l]
     if not selected:continue
     def patch(mod,args,out,l=l,selected=selected):
      delta=torch.zeros_like(args[0])
      for h in selected:
       sl=slice(h*64,(h+1)*64);replacement=0 if mode=='zero' else means[row['template_index']][l,sl]
       delta[0,-1,sl]=replacement-args[0][0,-1,sl]
      return out+delta@mod.weight
     handles.append(m.transformer.h[l].attn.c_proj.register_forward_hook(patch))
    try:v=m(torch.tensor([ids]),use_cache=False).logits[0,-1]
    finally:
     for h in handles:h.remove()
    assert int(v.argmax())==record['top1_id'],(row['id'],mode,system,k)
    ee=max(abs(v[row['io_id']].item()-record['io_logit']),abs(v[row['s_id']].item()-record['s_logit']));pp=abs(v.softmax(-1)[row['io_id']].item()-record['p_io']);err=max(err,ee);perr=max(perr,pp);assert ee<.001 and pp<.0001;n+=1
   if (row['id']+1)%32==0:print('replayed',row['id']+1,flush=True)
 result={'status':'passed','conditions':n,'all_argmax_equal':True,'max_candidate_logit_error':err,'max_probability_error':perr,'max_float64_mean_error':meanerr,'seconds':time.perf_counter()-t,'inputs':{f:hashlib.sha256((p/f).read_bytes()).hexdigest() for f in ['records.json','means.json','references.json']}}
 Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
