"""Real frozen CrossEncoder + unchanged BEIR Rerank; controlled candidate audit."""
import argparse,csv,importlib.metadata,platform,resource,sys,time,traceback
from pathlib import Path
from common import ROOT,read,write,sha,order,metric,verify

def main(cache,out,smoke):
 started=time.perf_counter();out.mkdir(parents=True,exist_ok=True)
 status={'command':['python','run.py','--cache','<verified-cache>','--out',str(out.name)]+(['--smoke'] if smoke else []),'exit_code':None,'stages':{},'smoke':smoke}
 try:
  import numpy as np,torch,pytrec_eval
  from sentence_transformers import CrossEncoder
  from official_rerank import Rerank
  cfg=read(ROOT/'config.json');torch.set_num_threads(cfg['threads']);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
  mp=verify(cache);status['stages']['download']={'status':'passed','mode':'existing hash-verified model/data cache'}
  t=time.perf_counter();corpus={r['_id']:r for r in map(__import__('json').loads,(cache/'scifact/corpus.jsonl').read_text().splitlines())};queries={r['_id']:r['text'] for r in map(__import__('json').loads,(cache/'scifact/queries.jsonl').read_text().splitlines())}
  full=read(ROOT/'inputs/candidates.json');qids=cfg['query_ids'][:5] if smoke else cfg['query_ids'];k=10 if smoke else cfg['top_k']
  candidates={q:{d:full[q][d] for d in order(full[q])[:k]} for q in qids};gold={q:read(ROOT/'inputs/qrels.json')[q] for q in qids};assert len(corpus)==5183
  pairs=[(q,d,queries[q],(corpus[d].get('title','')+' '+corpus[d]['text']).strip()) for q in qids for d in candidates[q]]
  identity={'model':read(ROOT/'model.lock.json'),'config':cfg,'input_lock':read(ROOT/'input.lock.json'),'data':{f:h for f,h in read(ROOT/'resources.lock.json')['data_files'].items() if f in ['scifact/corpus.jsonl','scifact/queries.jsonl','scifact/qrels/train.tsv']},'code':{f.name:sha(f) for f in sorted(ROOT.glob('*.py'))},'smoke':smoke}
  import hashlib,json
  identity['cache_key']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();write(out/'identity.json',identity)
  status['stages']['preprocessing']={'status':'passed','seconds':time.perf_counter()-t,'queries':len(qids),'corpus':len(corpus),'pairs':len(pairs)}
  t=time.perf_counter();model=CrossEncoder(str(mp),max_length=cfg['max_length'],device='cpu',local_files_only=True,default_activation_function=torch.nn.Identity());model.model.requires_grad_(False);model.model.eval()
  status['stages']['model_load']={'status':'passed','seconds':time.perf_counter()-t,'parameters':sum(p.numel() for p in model.model.parameters())}
  t=time.perf_counter();token_rows=[]
  for i in range(0,len(pairs),128):
   batch=pairs[i:i+128];lens=model.tokenizer([x[2] for x in batch],[x[3] for x in batch],truncation=False,padding=False,verbose=False)['input_ids']
   token_rows.extend({'query_id':x[0],'doc_id':x[1],'raw_tokens':len(ids),'retained_tokens':min(len(ids),cfg['max_length']),'truncated':len(ids)>cfg['max_length']} for x,ids in zip(batch,lens))
  features=model.smart_batching_collate_text_only([[x[2],x[3]] for x in pairs[:2]])
  with torch.no_grad():raw=model.model(**features,output_hidden_states=True)
  check_scores=model.predict([[x[2],x[3]] for x in pairs[:2]],batch_size=2,show_progress_bar=False)
  shapes={'input_ids':list(features['input_ids'].shape),'last_hidden':list(raw.hidden_states[-1].shape),'logits':list(raw.logits.shape),'predict':list(check_scores.shape),'manual_logit_max_error':float(np.max(np.abs(raw.logits[:,0].numpy()-check_scores))),'training':model.model.training}
  assert shapes['manual_logit_max_error']<1e-6
  write(out/'shape_check.json',shapes);write(out/'token_audit.json',{'pairs':len(pairs),'raw_tokens':sum(x['raw_tokens'] for x in token_rows),'retained_tokens':sum(x['retained_tokens'] for x in token_rows),'truncated_pairs':sum(x['truncated'] for x in token_rows),'max_raw_tokens':max(x['raw_tokens'] for x in token_rows),'tokenization':'longest_first on pair; combined512; no prompt'})
  status['stages']['token_shape_audit']={'status':'passed','seconds':time.perf_counter()-t}
  t=time.perf_counter();reranked=Rerank(model,batch_size=cfg['batch_size']).rerank(corpus,{q:queries[q] for q in qids},candidates,top_k=k)
  status['stages']['inference']={'status':'passed','seconds':time.perf_counter()-t,'pairs':len(pairs),'new_inference':True};print('inference',status['stages']['inference'],flush=True)
  t=time.perf_counter();oracle={q:{d:float(len(candidates[q])-i) for i,d in enumerate(sorted(candidates[q],key=lambda d:(gold[q].get(d,0)>0,d),reverse=True))} for q in qids}
  runs={'original':candidates,'reranked':reranked,'oracle':oracle};per={};summary={}
  evaluator=pytrec_eval.RelevanceEvaluator(gold,{'ndcg_cut.10','recall.10,50'})
  for name,run in runs.items():
   assert all(set(run[q])==set(candidates[q]) for q in qids)
   per[name]=evaluator.evaluate(run)
   for q in qids:per[name][q]['mrr_10']=metric(gold[q],order(run[q]))['mrr_10']
   summary[name]={m:sum(per[name][q][m] for q in qids)/len(qids) for m in ['ndcg_cut_10','recall_10','recall_50','mrr_10']}
  cases=[];cats={'no_relevant_candidate':0,'candidate_hit_but_top10_miss':0,'top10_hit':0};partial=0
  for q in qids:
   relevant={d for d,x in gold[q].items() if x>0};found=set(candidates[q])&relevant;partial+=0<len(found)<len(relevant)
   category='no_relevant_candidate' if not found else ('candidate_hit_but_top10_miss' if not(set(order(reranked[q])[:10])&relevant) else 'top10_hit');cats[category]+=1
   a=per['original'][q]['ndcg_cut_10'];b=per['reranked'][q]['ndcg_cut_10'];o=per['oracle'][q]['ndcg_cut_10'];assert b<=o+1e-12
   cases.append({'query_id':q,'delta':b-a,'original':a,'reranked':b,'oracle':o,'candidate_loss':1-o,'ordering_loss':o-b,'category':category,'relevant_total':len(relevant),'relevant_in_candidates':len(found),'original_first_relevant':next((i for i,d in enumerate(order(candidates[q]),1) if d in relevant),None),'reranked_first_relevant':next((i for i,d in enumerate(order(reranked[q]),1) if d in relevant),None)})
  summary.update({'queries':len(qids),'pairs':len(pairs),'win_tie_loss':[sum(x['delta']>1e-12 for x in cases),sum(abs(x['delta'])<=1e-12 for x in cases),sum(x['delta']< -1e-12 for x in cases)],'failure_categories':cats,'partial_candidate_coverage':partial,'mean_candidate_loss':sum(x['candidate_loss'] for x in cases)/len(qids),'mean_ordering_loss':sum(x['ordering_loss'] for x in cases)/len(qids),'delta':summary['reranked']['ndcg_cut_10']-summary['original']['ndcg_cut_10'],'scope':'development subset, no test, no training'})
  write(out/'runs.json',runs);write(out/'per_query.json',per);write(out/'diagnostics.json',cases);write(out/'summary.json',summary)
  selected=sorted(cases,key=lambda x:(x['delta'],x['query_id']));write(out/'cases.json',{'rule':'3 lowest and3 highest delta; automatic, not human annotation','rows':selected[:3]+selected[-3:]})
  with (out/'pair_scores.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=['query_id','doc_id','raw_tokens','retained_tokens','truncated','original_rank','reranked_rank','logit','relevant']);w.writeheader()
   for r in token_rows:
    q,d=r['query_id'],r['doc_id'];w.writerow({**r,'original_rank':order(candidates[q]).index(d)+1,'reranked_rank':order(reranked[q]).index(d)+1,'logit':reranked[q][d],'relevant':int(gold[q].get(d,0)>0)})
  status['stages']['evaluation']={'status':'passed','seconds':time.perf_counter()-t,'evaluator':'pytrec_eval-terrier'};status['exit_code']=0
  write(out/'environment.json',{'python':sys.version,'platform':platform.platform(),'device':'CPU','dtype':'float32','peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},'gpu_memory':'not applicable','retrieval_cost':'not remeasured; inherited frozen real runs','timing':'includes imports, verification, load and token audit; excludes network downloads'})
  print(json.dumps(summary,indent=2),flush=True)
 except Exception:
  status['exit_code']=1;status['error']=traceback.format_exc();raise
 finally:
  status['elapsed_seconds']=time.perf_counter()-started;write(out/'status.json',status)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,default=Path('results'));p.add_argument('--smoke',action='store_true');a=p.parse_args();main(a.cache,a.out,a.smoke)
