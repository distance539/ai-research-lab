"""Actual GPT-2 IOI forward pass; protocol identities checked before inference."""
import argparse,json,time,platform,resource,sys,hashlib
from pathlib import Path
from prepare import sha,prepare
ROOT=Path(__file__).resolve().parent

def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cache',required=True);ap.add_argument('--out',required=True);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
 out=Path(args.out);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
 stages={'download':'pending','preprocessing':'pending','inference':'pending','evaluation':'pending','training':'not applicable'};write(out/'stages.json',stages)
 cfg=json.loads((ROOT/'config.json').read_text());plan=json.loads((ROOT/'plan.json').read_text())
 for f,h in plan['files'].items():assert sha(ROOT/f)==h,f'frozen code/input modified: {f}'
 lock=prepare(args.cache);stages['download']='passed';write(out/'stages.json',stages)
 import torch,transformers,tokenizers,safetensors,numpy
 from transformers import AutoTokenizer,AutoModelForCausalLM
 torch.set_num_threads(cfg['threads']);torch.manual_seed(cfg['seed']);torch.use_deterministic_algorithms(True)
 tok=AutoTokenizer.from_pretrained(args.cache,local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(args.cache,local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager');model.eval()
 samples=json.loads((ROOT/'samples.json').read_text());samples=samples[:2] if args.smoke else samples
 prepared=[]
 for row in samples:
  ids=tok.encode(row['prompt'],add_special_tokens=False);io=tok.encode(row['answer'],add_special_tokens=False);s=tok.encode(row['distractor'],add_special_tokens=False)
  assert len(io)==len(s)==1 and io!=s
  assert tok.encode(row['prompt']+row['answer'],add_special_tokens=False)==ids+io
  assert tok.encode(row['prompt']+row['distractor'],add_special_tokens=False)==ids+s
  assert len(ids)<=cfg['max_tokens'] and ids.count(io[0])==1 and ids.count(s[0])==2
  prepared.append((row,ids,io[0],s[0]))
 stages['preprocessing']='passed';write(out/'stages.json',stages);setup=time.perf_counter()-start
 records=[];forward_start=time.perf_counter()
 with torch.inference_mode():
  for row,ids,io,s in prepared:
   x=torch.tensor([ids]);y=model(x,use_cache=False,output_hidden_states=True);v=y.logits[0,-1,:];hidden=y.hidden_states[-1][0,-1,:]
   direct=model.lm_head(hidden);err=(direct-v).abs().max().item();assert err<1e-4
   prob=v.softmax(-1);d=(v[io]-v[s]).item();r=dict(row)
   r.update(token_ids=ids,tokens=tok.convert_ids_to_tokens(ids),io_id=io,s_id=s,io_position=ids.index(io),s_positions=[i for i,vv in enumerate(ids) if vv==s],end_position=len(ids)-1,logits_shape=list(y.logits.shape),hidden_shape=list(hidden.shape),io_logit=v[io].item(),s_logit=v[s].item(),logit_diff=d,io_prob=prob[io].item(),s_prob=prob[s].item(),pair_prob=torch.sigmoid(v[io]-v[s]).item(),top1_id=int(v.argmax()),top1_text=tok.decode([int(v.argmax())]),top1_logit=v.max().item(),io_rank=int((v>v[io]).sum())+1,readout_max_error=err,prompt_sha256=hashlib.sha256(row['prompt'].encode()).hexdigest())
   records.append(r);print('row',row['id'],'diff',round(d,6),flush=True)
   assert time.perf_counter()-start<cfg['budget_seconds'],'runtime budget exceeded'
 elapsed=time.perf_counter()-start;inference=time.perf_counter()-forward_start;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
 assert rss<=cfg['budget_rss_bytes'],'RSS budget exceeded'
 stages['inference']='passed';write(out/'records.json',records)
 def agg(rows):
  return {'n':len(rows),'mean_logit_diff':sum(r['logit_diff'] for r in rows)/len(rows),'positive':sum(r['logit_diff']>0 for r in rows),'top1_correct':sum(r['top1_id']==r['io_id'] for r in rows),'mean_io_prob':sum(r['io_prob'] for r in rows)/len(rows),'mean_s_prob':sum(r['s_prob'] for r in rows)/len(rows),'mean_pair_prob':sum(r['pair_prob'] for r in rows)/len(rows),'min_logit_diff':min(r['logit_diff'] for r in rows),'max_logit_diff':max(r['logit_diff'] for r in rows)}
 summary={'overall':agg(records),'cells':{f'{t}-{o}':agg([r for r in records if r['template_index']==t and r['order']==o]) for t,o in sorted(set((r['template_index'],r['order']) for r in records))},'setup_seconds':setup,'inference_seconds':inference,'total_seconds_after_imports':elapsed,'peak_rss_bytes':rss,'projected_full_seconds':setup+inference*64/len(records),'readout_max_error':max(r['readout_max_error'] for r in records)}
 if args.smoke:assert summary['projected_full_seconds']<=cfg['budget_seconds'],'projected budget exceeded'
 write(out/'summary.json',summary);stages['evaluation']='passed';write(out/'stages.json',stages)
 write(out/'identity.json',{'plan_sha256':sha(ROOT/'plan.json'),'resources_sha256':sha(ROOT/'resources.lock.json'),'mode':'smoke' if args.smoke else 'formal','sample_ids':[r['id'] for r in records],'cache_key':hashlib.sha256((sha(ROOT/'plan.json')+sha(ROOT/'resources.lock.json')+str(args.smoke)).encode()).hexdigest(),'cache_reused_outputs':False})
 write(out/'environment.json',{'os':platform.platform(),'machine':platform.machine(),'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__,'tokenizers':tokenizers.__version__,'safetensors':safetensors.__version__,'numpy':numpy.__version__,'device':'cpu','dtype':'float32','threads':torch.get_num_threads(),'parameters':sum(p.numel() for p in model.parameters()),'layers':model.config.n_layer,'heads':model.config.n_head,'width':model.config.n_embd,'vocab':model.config.vocab_size})
 print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
