"""109: two real scoring passes, four factorial ranking conditions."""
import argparse,datetime,hashlib,importlib.metadata,json,platform,resource,sys,time,traceback
from pathlib import Path
from common import ROOT,read,write,sha,verify,order
from material_tools import text_of,visibility

def main(cache,out,smoke):
 start=time.perf_counter();out.mkdir(parents=True,exist_ok=False);status={'command':'python run.py --cache <verified-cache> --out '+out.name+(' --smoke' if smoke else ''),'exit_code':None,'stages':{},'semantic_review':'107 actual approval reused; material hash verified','smoke':smoke}
 try:
  import numpy as np,torch,pytrec_eval
  from sentence_transformers import CrossEncoder
  from official_rerank import Rerank
  cfg=read(ROOT/'config.json');torch.set_num_threads(cfg['threads']);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
  mp=verify(cache);status['stages']['download']={'status':'passed','mode':'existing hash-verified cache; no confirmation qrels read'}
  assert read(ROOT/'human_review_confirmation.json')['materials_sha256']==sha(ROOT/'materials.json')
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
  write(out/'raw_runs.json',runs);write(out/'pairs.json',pairs);write(out/'shape_check.json',shapes)
  t=time.perf_counter();cells={'00':runs['original'],'10':runs['front']};trace=[]
  for a,cond in [('0','original'),('1','front')]:
   cell=a+'1';cells[cell]={}
   for q in candidates:
    prior_rank={d:i for i,d in enumerate(order(candidates[q]),1)};ce_rank={d:i for i,d in enumerate(order(runs[cond][q]),1)}
    cells[cell][q]={d:1/(cfg['fusion_constant']+prior_rank[d])+1/(cfg['fusion_constant']+ce_rank[d]) for d in candidates[q]}
    for d in candidates[q]:trace.append({'cell':cell,'query_id':q,'doc_id':d,'prior_rank':prior_rank[d],'ce_rank':ce_rank[d],'score':cells[cell][q][d]})
    order(cells[cell][q])
  fusion_seconds=time.perf_counter()-t
  ev=pytrec_eval.RelevanceEvaluator(gold,{'ndcg_cut.10','recall.50'});metrics={k:ev.evaluate(v) for k,v in cells.items()};pq=[]
  for r in rows:
   q,d=r['query_id'],r['doc_id'];y={k:metrics[k][q]['ndcg_cut_10'] for k in cells}
   effects={'A_at_B0':y['10']-y['00'],'A_at_B1':y['11']-y['01'],'B_at_A0':y['01']-y['00'],'B_at_A1':y['11']-y['10']}
   effects['interaction']=effects['B_at_A1']-effects['B_at_A0'];effects['A_average']=(effects['A_at_B0']+effects['A_at_B1'])/2;effects['B_average']=(effects['B_at_A0']+effects['B_at_A1'])/2
   pq.append({'query_id':q,'doc_id':d,'group':r['group'],'ndcg':y,'rank':{k:order(cells[k][q]).index(d)+1 for k in cells},'recall50':{k:metrics[k][q]['recall_50'] for k in cells},'effects':effects})
  non_targets=[x for x in pairs if x['condition']=='original' and not x['target']];bykey={(x['condition'],x['query_id'],x['doc_id']):x for x in pairs};unchanged=max(abs(x['score']-bykey[('front',x['query_id'],x['doc_id'])]['score']) for x in non_targets)
  for x in pairs:
   z=bykey[('original' if x['condition']=='front' else 'front',x['query_id'],x['doc_id'])];assert x['raw_tokens']==z['raw_tokens'] and x['retained_tokens']==z['retained_tokens']
   if not x['target']:assert x['input_sha256']==z['input_sha256'] and x['score']==z['score']
  summary={'queries':4,'pairs':len(pairs),'cell_means':{k:sum(x['ndcg'][k] for x in pq)/4 for k in cells},'mean_effects':{k:sum(x['effects'][k] for x in pq)/4 for k in effects},'group_interaction':{g:sum(x['effects']['interaction'] for x in pq if x['group']==g)/2 for g in ['long','short']},'token_totals':{k:sum(x['retained_tokens'] for x in pairs if x['condition']==k) for k in runs},'unchanged_negative_max_score_error':unchanged,'confirmation_evaluated':False,'interpretation':'four previously observed nonrandom oracle probes; descriptive; no population inference or success threshold claim'}
  write(out/'cells.json',cells);write(out/'fusion_trace.json',trace);write(out/'per_query.json',pq);write(out/'summary.json',summary)
  write(out/'budget_ledger.json',{'unique_scored_pairs':len(pairs),'cell_pair_uses':2*len(pairs),'extra_pairs_for_B':0,'auxiliary_shape_pairs':4,'training_tokens':0,'token_totals':summary['token_totals'],'inference_seconds':inference_total,'fusion_seconds':fusion_seconds,'fusion_scope':'rank maps, reciprocal scores, sort and trace; excludes metrics/writes','retrieval_seconds':'not remeasured; fixed historical candidates','token_note':'same counts are verified; equal count is not identical content or FLOPs'})
  status['stages']['evaluation']={'status':'passed','engine':'pytrec_eval-terrier0.5.10','label_status':'107 actual approval reused for byte-identical original/front material'};status['exit_code']=0
  write(out/'environment.json',{'python':sys.version,'platform':platform.platform(),'device':'CPU','dtype':'float32','peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},'gpu_memory':'not applicable','retrieval_cost':'not measured; historical fixed candidates'})
  print(json.dumps(summary,indent=2),flush=True)
 except Exception:
  status['exit_code']=1;status['error']=traceback.format_exc();raise
 finally:status['elapsed_seconds']=time.perf_counter()-start;write(out/'status.json',status)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();main(a.cache,a.out,a.smoke)
