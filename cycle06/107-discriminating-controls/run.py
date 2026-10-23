"""Four fixed conditions; qrels-assisted probes, all semantic conclusions provisional."""
import argparse,datetime,hashlib,importlib.metadata,json,platform,resource,sys,time,traceback
from pathlib import Path
from common import ROOT,read,write,sha,verify,order
from build_materials import text_of,visibility

def main(cache,out,smoke):
 start=time.perf_counter();out.mkdir(parents=True,exist_ok=False);status={'command':'python run.py --cache <verified-cache> --out '+out.name+(' --smoke' if smoke else ''),'exit_code':None,'stages':{},'semantic_review':'pending','smoke':smoke}
 try:
  import numpy as np,torch,pytrec_eval
  from sentence_transformers import CrossEncoder
  from official_rerank import Rerank
  cfg=read(ROOT/'config.json');torch.set_num_threads(cfg['threads']);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
  mp=verify(cache);status['stages']['download']={'status':'passed','mode':'existing hash-verified cache; no confirmation qrels read'}
  rows=read(ROOT/'materials.json')['rows'];corpus={x['_id']:x for x in map(json.loads,(cache/'scifact/corpus.jsonl').read_text().splitlines())};allc=read(ROOT/'inputs/candidates.json');gold={r['query_id']:read(ROOT/'inputs/qrels.json')[r['query_id']] for r in rows}
  candidates={r['query_id']:{d:allc[r['query_id']][d] for d in (list(dict.fromkeys(order(allc[r['query_id']])[:2]+[r['doc_id']])) if smoke else order(allc[r['query_id']]))} for r in rows}
  identity={'config':cfg,'model':read(ROOT/'model.lock.json'),'resources':read(ROOT/'resources.lock.json'),'input_lock':read(ROOT/'input.lock.json'),'code':{p.name:sha(p) for p in ROOT.glob('*.py')},'smoke':smoke};identity['cache_key']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();write(out/'identity.json',identity)
  status['stages']['preprocessing']={'status':'passed','queries':len(rows),'pairs_per_condition':sum(map(len,candidates.values()))}
  model=CrossEncoder(str(mp),device='cpu',max_length=512,local_files_only=True,default_activation_function=torch.nn.Identity());model.model.eval();model.model.requires_grad_(False)
  tok=model.tokenizer;assert tok.is_fast
  runs={};pairs=[];timings={};shapes=None;inference_total=0
  for cond in cfg['conditions']:
   runs[cond]={};t=time.perf_counter()
   for r in rows:
    q=r['query_id'];d=r['doc_id'];doc=r['document'];indices=r['orders'][cond];c=dict(corpus);c[d]={'title':doc['title'],'text':' '.join(doc['abstract'][i] for i in indices)}
    if cond in ['original','identity']:assert c[d]['text']==corpus[d]['text'] and c[d]['title']==corpus[d]['title']
    vis=visibility(tok,r['query'],doc,indices,r['evidence_indices']);assert vis==r['visibility'][cond]
    xs=[[r['query'],(c[k]['title']+' '+c[k]['text']).strip()] for k in candidates[q]]
    ids=tok([x[0] for x in xs],[x[1] for x in xs],truncation='longest_first',max_length=512,padding=False)['input_ids']
    raw=tok([x[0] for x in xs],[x[1] for x in xs],truncation=False,padding=False,verbose=False)['input_ids']
    if shapes is None:
     features=model.smart_batching_collate_text_only(xs[:2])
     with torch.no_grad():y=model.model(**features,output_hidden_states=True)
     pred=model.predict(xs[:2],batch_size=2,show_progress_bar=False)
     shapes={'input_ids':list(features['input_ids'].shape),'hidden':list(y.hidden_states[-1].shape),'logits':list(y.logits.shape),'manual_predict_max_error':float(np.max(np.abs(pred-y.logits[:,0].numpy()))),'training':model.model.training};assert shapes['manual_predict_max_error']<1e-6
    t0=time.perf_counter();sc=Rerank(model,batch_size=cfg['batch_size']).rerank(c,{q:r['query']},{q:candidates[q]},top_k=len(candidates[q]))[q];inference_total+=time.perf_counter()-t0
    runs[cond][q]=sc
    for k,ii,rr in zip(candidates[q],ids,raw):pairs.append({'condition':cond,'query_id':q,'doc_id':k,'score':float(sc[k]),'raw_tokens':len(rr),'retained_tokens':len(ii),'input_sha256':hashlib.sha256(json.dumps(ii).encode()).hexdigest(),'target':k==d})
    assert inference_total<cfg['budget_seconds'];assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)<cfg['peak_rss_limit_bytes']
   timings[cond]=time.perf_counter()-t;print(cond,'seconds',timings[cond],flush=True)
  status['stages']['inference']={'status':'passed','pairs':len(pairs),'seconds':inference_total,'conditions_seconds_with_token_audit':timings}
  evaluator=pytrec_eval.RelevanceEvaluator(gold,{'ndcg_cut.10','recall.50'});metrics={k:evaluator.evaluate(v) for k,v in runs.items()};pq=[]
  for r in rows:
   q,d=r['query_id'],r['doc_id'];pq.append({'query_id':q,'doc_id':d,'group':r['group'],'ndcg':{k:metrics[k][q]['ndcg_cut_10'] for k in runs},'rank':{k:order(runs[k][q]).index(d)+1 for k in runs},'target_score':{k:runs[k][q][d] for k in runs},'front_minus_tail':metrics['front'][q]['ndcg_cut_10']-metrics['tail'][q]['ndcg_cut_10']})
  group_means={g:sum(x['front_minus_tail'] for x in pq if x['group']==g)/2 for g in ['long','short']}
  identity_error=max(abs(runs['original'][q][d]-runs['identity'][q][d]) for q in candidates for d in candidates[q]);assert identity_error==0
  summary={'queries':4,'pairs':len(pairs),'group_delta':group_means,'difference_of_group_deltas':group_means['long']-group_means['short'],'identity_max_error':identity_error,'token_totals':{k:sum(x['retained_tokens'] for x in pairs if x['condition']==k) for k in runs},'semantic_review':'pending; transformed metrics provisional','confirmation_evaluated':False,'interpretation':'qrels-assisted diagnostic; short group not randomly matched; no causal or population claim'}
  write(out/'runs.json',runs);write(out/'pairs.json',pairs);write(out/'per_query.json',pq);write(out/'summary.json',summary);write(out/'shape_check.json',shapes)
  status['stages']['evaluation']={'status':'passed','engine':'pytrec_eval-terrier0.5.10','label_status':'original qrels provisionally reused, no human approval asserted'};status['exit_code']=0
  write(out/'environment.json',{'python':sys.version,'platform':platform.platform(),'device':'CPU','dtype':'float32','peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},'gpu_memory':'not applicable','retrieval_cost':'not measured; historical fixed candidates'})
  print(json.dumps(summary,indent=2),flush=True)
 except Exception:
  status['exit_code']=1;status['error']=traceback.format_exc();raise
 finally:status['elapsed_seconds']=time.perf_counter()-start;write(out/'status.json',status)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();main(a.cache,a.out,a.smoke)
