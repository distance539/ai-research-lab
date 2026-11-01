"""Real finite-candidate scoring; no open generation correctness estimate."""
import argparse,hashlib,json,math,platform,resource,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x): (P/n).write_text(json.dumps(x,indent=2)+'\n')
def main():
 a=argparse.ArgumentParser();a.add_argument('--model-dir',type=Path,required=True);a.add_argument('--data',type=Path,required=True);a.add_argument('--smoke',action='store_true');a=a.parse_args()
 import torch,transformers,pyarrow.parquet as pq
 c=json.loads((P/'config.json').read_text());assert H(P/'formal_predictions.json')==c['input_sha256'];assert H(a.data)==json.loads((P/'resources.lock.json').read_text())['dataset']['sha256']
 torch.set_num_threads(c['threads']);torch.manual_seed(c['seed']);torch.use_deterministic_algorithms(True)
 started=time.perf_counter();tag='smoke' if a.smoke else 'formal';base=json.loads((P/'formal_predictions.json').read_text());base=base[:8] if a.smoke else base
 data={x['idx']:x for x in pq.read_table(a.data).to_pylist()}
 expected=json.loads((P/'model.lock.json').read_text());assert all(H(a.model_dir/e['name'])==e['sha256'] for e in expected['files'])
 lock={f.name:H(f) for f in a.model_dir.iterdir() if f.is_file()};save(tag+'_start.json',{'config_sha256':H(P/'config.json'),'code_sha256':H(__file__),'protocol_sha256':H(P/'PROTOCOL.md'),'model_files':lock,'n':len(base)})
 tok=transformers.AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True)
 model=transformers.AutoModelForCausalLM.from_pretrained(a.model_dir,local_files_only=True,attn_implementation='eager',torch_dtype=torch.float32).eval()
 records=[];forward=time.perf_counter()
 for b in base:
  row=data[b['idx']];assert hashlib.sha256(row['sentence'].encode()).hexdigest()==b['text_sha256'];assert row['label']==b['label']
  prompt=c['prompt'].format(sentence=row['sentence']);ctx=tok.encode(prompt,add_special_tokens=False);scores=[];parts=[];ids=[];shapes=[]
  for candidate in c['candidates']:
   full=tok.encode(prompt+candidate,add_special_tokens=False);assert full[:len(ctx)]==ctx;tail=full[len(ctx):];assert tail and len(full)<=c['max_length']
   x=torch.tensor([full[:-1]])
   with torch.inference_mode(): logits=model(x).logits
   lp=torch.log_softmax(logits[0,len(ctx)-1:,:].double(),dim=-1);vals=lp[torch.arange(len(tail)),torch.tensor(tail)].tolist()
   scores.append(sum(vals));parts.append(vals);ids.append(tail);shapes.append(list(logits.shape))
  denom=sum(math.exp(s-max(scores)) for s in scores);probs=[math.exp(s-max(scores))/denom for s in scores];pred=max(range(2),key=lambda k:probs[k])
  records.append({'idx':b['idx'],'label':b['label'],'text_sha256':b['text_sha256'],'context_tokens':len(ctx),'candidate_ids':ids,'token_logprobs':parts,'scores':scores,'probabilities':probs,'prediction':pred,'confidence':probs[pred],'correct':int(pred==b['label']),'candidate_mass':sum(math.exp(s) for s in scores) if all(len(x)==1 for x in ids) else None,'logits_shapes':shapes})
  print(tag,len(records),flush=True)
  assert time.perf_counter()-started<c['budget_seconds']
 save(tag+'_candidate_predictions.json',records)
 def metrics(rows):
  return {'n':len(rows),'correct':sum(r['correct'] for r in rows),'mean_confidence':sum(r['confidence'] for r in rows)/len(rows),'nll':sum(-math.log(r['probabilities'][r['label']]) for r in rows)/len(rows),'brier':sum(sum((p-int(k==r['label']))**2 for k,p in enumerate(r['probabilities'])) for r in rows)/len(rows)}
 result={'classifier':metrics(base),'candidate':metrics(records),'mean_candidate_mass':sum(r['candidate_mass'] for r in records)/len(records) if all(r['candidate_mass'] is not None for r in records) else None,'disagreements':sum(x['prediction']!=y['prediction'] for x,y in zip(base,records)),'forward_seconds':time.perf_counter()-forward,'total_seconds_after_imports':time.perf_counter()-started,'peak_rss_raw':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'rss_unit':'bytes' if sys.platform=='darwin' else 'KiB','python':platform.python_version(),'os':platform.platform(),'torch':torch.__version__,'transformers':transformers.__version__}
 save(tag+'_comparison.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__': main()
