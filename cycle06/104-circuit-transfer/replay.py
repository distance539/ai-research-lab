"""Independent full-vocabulary replay via postprojection output deltas."""
import argparse,json,hashlib,time
from pathlib import Path

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cache',required=True);ap.add_argument('--results',default='results');ap.add_argument('--out',required=True);a=ap.parse_args()
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 from prepare import prepare
 prepare(a.cache);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
 model=AutoModelForCausalLM.from_pretrained(a.cache,local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval();tok=AutoTokenizer.from_pretrained(a.cache,local_files_only=True)
 p=Path(a.results);rows=json.loads((p/'records.json').read_text());heads=[(8,10),(8,6),(9,9)];err=0;prob=0;count=0;failed_clean=[];t=time.perf_counter()
 with torch.inference_mode():
  for r in rows:
   ids=tok.encode(r['prompt'],add_special_tokens=False);assert ids==r['ids'];clean_z={};hooks=[]
   for layer in [8,9]:
    def capture(module,args,layer=layer):clean_z[layer]=args[0].clone()
    hooks.append(model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(capture))
   model(torch.tensor([ids]),use_cache=False)
   for h in hooks:h.remove()
   for cond in ['clean','zero','corrupt','restored']:
    source=ids if cond in ['clean','zero'] else r['corrupt_ids'];hooks=[]
    if cond in ['zero','restored']:
     for layer in [8,9]:
      def replace(module,args,output,layer=layer,cond=cond):
       delta=torch.zeros_like(args[0])
       for l,h in heads:
        if l==layer:
         sl=slice(h*64,(h+1)*64);desired=torch.zeros(64) if cond=='zero' else clean_z[layer][0,-1,sl]
         delta[0,-1,sl]=desired-args[0][0,-1,sl]
       return output+delta@module.weight
      hooks.append(model.transformer.h[layer].attn.c_proj.register_forward_hook(replace))
    try:v=model(torch.tensor([source]),use_cache=False).logits[0,-1]
    finally:
     for h in hooks:h.remove()
    record=r[cond];top=int(v.argmax());assert top==record['top1_id'],(r['id'],cond)
    ee=max(abs(v[r['io_id']].item()-record['io_logit']),abs(v[r['s_id']].item()-record['s_logit']));pp=abs(v.softmax(-1)[r['io_id']].item()-record['p_io']);err=max(err,ee);prob=max(prob,pp);assert ee<.001 and pp<.0001
    if cond=='clean' and top!=r['io_id']:failed_clean.append(dict(id=r['id'],group=r['group'],prompt=r['prompt'],io=r['answer'],top1=tok.decode([top]),diff=record['diff']))
    count+=1
 result=dict(status='passed',conditions=count,max_candidate_logit_error=err,max_probability_error=prob,all_argmax_equal=True,seconds=time.perf_counter()-t,records_sha256=hashlib.sha256((p/'records.json').read_bytes()).hexdigest(),failed_clean=failed_clean)
 Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='failed_clean'}))
if __name__=='__main__':main()
