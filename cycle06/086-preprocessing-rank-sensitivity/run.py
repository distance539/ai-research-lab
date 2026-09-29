"""Real four-condition preprocessing experiment. No score-cache reuse."""
import argparse,csv,hashlib,importlib.metadata,json,platform,resource,sys,time,traceback
from pathlib import Path
from common import ROOT,read,write,sha,order,metric,verify

def main(cache,out,smoke):
 started=time.perf_counter();out.mkdir(parents=True,exist_ok=False)
 status={'command':['python','run.py','--cache','<hash-verified-cache>','--out',out.name]+(['--smoke'] if smoke else []),'exit_code':None,'stages':{},'smoke':smoke}
 try:
  import numpy as np,torch,pytrec_eval
  from sentence_transformers import CrossEncoder
  from official_rerank import Rerank
  cfg=read(ROOT/'config.json');torch.set_num_threads(cfg['threads']);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
  mp=verify(cache);status['stages']['download']={'status':'passed','mode':'existing hash-verified cache'}
  corpus={r['_id']:r for r in map(json.loads,(cache/'scifact/corpus.jsonl').read_text().splitlines())};queries={r['_id']:r['text'] for r in map(json.loads,(cache/'scifact/queries.jsonl').read_text().splitlines())};assert len(corpus)==5183
  full=read(ROOT/'inputs/candidates.json');qids=cfg['query_ids'][:5] if smoke else cfg['query_ids'];k=10 if smoke else cfg['top_k'];candidates={q:{d:full[q][d] for d in order(full[q])[:k]} for q in qids};gold={q:read(ROOT/'inputs/qrels.json')[q] for q in qids}
  identity={'model_tokenizer':read(ROOT/'model.lock.json'),'config':cfg,'inputs':read(ROOT/'input.lock.json'),'data':read(ROOT/'resources.lock.json')['data_files'],'code':{f.name:sha(f) for f in sorted(ROOT.glob('*.py'))},'requirements':sha(ROOT/'requirements.txt'),'prompt':'none','decode':'scalar Identity logit','smoke':smoke}
  identity['cache_key']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();write(out/'identity.json',identity)
  status['stages']['preprocessing']={'status':'passed','queries':len(qids),'corpus':len(corpus),'pairs_per_variant':len(qids)*k}
  model=CrossEncoder(str(mp),max_length=512,device='cpu',local_files_only=True,default_activation_function=torch.nn.Identity());model.model.requires_grad_(False);model.model.eval();assert model.tokenizer.is_fast
  runs={'rrf':candidates};tokens={};shapes={};times={};allrows=[]
  for variant in cfg['variants']:
   name=variant['name'];limit=variant['max_length'];model.max_length=limit
   altered={d:{'title':r.get('title','') if variant['title'] else '', 'text':r['text']} for d,r in corpus.items()}
   pairs=[(q,d,queries[q],(altered[d]['title']+' '+altered[d]['text']).strip(),len(altered[d]['title'])) for q in qids for d in candidates[q]]
   rows=[];start=time.perf_counter()
   for i in range(0,len(pairs),128):
    batch=pairs[i:i+128];qs=[x[2] for x in batch];ds=[x[3] for x in batch]
    raw=model.tokenizer(qs,ds,truncation=False,padding=False,verbose=False,return_offsets_mapping=True)
    cut=model.tokenizer(qs,ds,truncation='longest_first',max_length=limit,padding=False,return_offsets_mapping=True)
    for j,x in enumerate(batch):
     q,d,_,_,boundary=x;rawseq=raw.sequence_ids(j);seq=cut.sequence_ids(j);off=cut['offset_mapping'][j]
     qt=seq.count(0);dt=seq.count(1);at=sum(s==1 and (not boundary or o[1]>boundary) for s,o in zip(seq,off));special=seq.count(None)
     ids=cut['input_ids'][j];r={'variant':name,'query_id':q,'doc_id':d,'raw_tokens':len(raw['input_ids'][j]),'retained_tokens':len(ids),'query_raw':rawseq.count(0),'query_retained':qt,'doc_raw':rawseq.count(1),'doc_retained':dt,'title_retained':dt-at,'abstract_retained':at,'special_tokens':special,'truncated':int(len(ids)<len(raw['input_ids'][j])),'relevant':int(gold[q].get(d,0)>0),'input_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest()};assert qt+dt+special==len(ids);rows.append(r)
   sample=[[x[2],x[3]] for x in pairs[:2]];features=model.smart_batching_collate_text_only(sample)
   with torch.no_grad():rawcheck=model.model(**features,output_hidden_states=True)
   scores=model.predict(sample,batch_size=2,show_progress_bar=False)
   shapes[name]={'input':list(features['input_ids'].shape),'hidden':list(rawcheck.hidden_states[-1].shape),'logits':list(rawcheck.logits.shape),'predict':list(scores.shape),'manual_error':float(np.max(np.abs(rawcheck.logits[:,0].numpy()-scores)))};assert shapes[name]['manual_error']<1e-6
   for j in range(2):
    ids=features['input_ids'][j][features['attention_mask'][j].bool()].tolist();assert hashlib.sha256(json.dumps(ids).encode()).hexdigest()==rows[j]['input_sha256']
   audit_seconds=time.perf_counter()-start;start=time.perf_counter()
   runs[name]=Rerank(model,batch_size=cfg['batch_size']).rerank(altered,{q:queries[q] for q in qids},candidates,top_k=k)
   times[name]={'inference_seconds':time.perf_counter()-start,'token_shape_audit_seconds':audit_seconds}
   for r in rows:
    q,d=r['query_id'],r['doc_id'];r.update(logit=runs[name][q][d],rank=order(runs[name][q]).index(d)+1)
   allrows.extend(rows);tokens[name]={key:sum(r[key] for r in rows) for key in ['raw_tokens','retained_tokens','query_raw','query_retained','doc_raw','doc_retained','title_retained','abstract_retained','truncated']};tokens[name]['relevant_pairs']=sum(r['relevant'] for r in rows);tokens[name]['relevant_truncated']=sum(r['relevant']*r['truncated'] for r in rows);tokens[name]['relevant_unique_docs_truncated']=len({r['doc_id'] for r in rows if r['relevant'] and r['truncated']})
   write(out/'runs.json',runs);print(name,times[name],tokens[name],flush=True)
  status['stages']['inference']={'status':'passed','variants':4,'pairs_total':len(allrows),'timings':times}
  evaluator=pytrec_eval.RelevanceEvaluator(gold,{'ndcg_cut.10','recall.10,50'});per={};agg={}
  for name,run in runs.items():
   per[name]=evaluator.evaluate(run)
   for q in qids:
    assert set(run[q])==set(candidates[q]);per[name][q]['mrr_10']=metric(gold[q],order(run[q]))['mrr_10']
   agg[name]={m:sum(per[name][q][m] for q in qids)/len(qids) for m in ['ndcg_cut_10','mrr_10','recall_10','recall_50']}
  def delta(a,b):return {q:per[a][q]['ndcg_cut_10']-per[b][q]['ndcg_cut_10'] for q in qids}
  contrasts={f'{a}_minus_{b}':delta(a,b) for a,b in [('title_256','abstract_256'),('title_512','abstract_512'),('title_512','title_256'),('abstract_512','abstract_256')]+[(v['name'],'rrf') for v in cfg['variants']]}
  contrasts['interaction']={q:contrasts['title_512_minus_abstract_512'][q]-contrasts['title_256_minus_abstract_256'][q] for q in qids}
  effects={n:{'mean':sum(d.values())/len(d),'win_tie_loss':[sum(v>1e-12 for v in d.values()),sum(abs(v)<=1e-12 for v in d.values()),sum(v< -1e-12 for v in d.values())]} for n,d in contrasts.items()}
  summary={'queries':len(qids),'pairs_per_variant':len(qids)*k,'metrics':agg,'effects':effects,'tokens':tokens,'timings':times,'scope':'development100x50, 4 frozen conditions; smoke5x10 if specified'}
  write(out/'summary.json',summary);write(out/'per_query.json',per);write(out/'contrasts.json',contrasts);write(out/'shapes.json',shapes)
  chosen=sorted(qids,key=lambda q:(-abs(contrasts['title_512_minus_title_256'][q]),q))[:3]
  write(out/'cases.json',{'rule':'3largest absolute title length delta; tie queryID asc; automatic not human annotation','rows':[{'query_id':q,'metrics':{n:per[n][q] for n in runs},'relevant_ranks':{n:{d:order(runs[n][q]).index(d)+1 for d in gold[q] if d in candidates[q]} for n in runs}} for q in chosen]})
  with (out/'pairs.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
  status['stages']['evaluation']={'status':'passed','evaluator':'pytrec_eval-terrier0.5.10; independent MRR10'};status['exit_code']=0
  write(out/'environment.json',{'python':sys.version,'platform':platform.platform(),'device':'CPU','dtype':'float32','peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},'gpu_memory':'not measured; CPU only','notes':'RSS process maximum; timing excludes downloads and original retrieval; condition order fixed, no latency significance'})
 except Exception:status['exit_code']=1;status['error']=traceback.format_exc();raise
 finally:status['elapsed_seconds']=time.perf_counter()-started;write(out/'status.json',status)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();main(a.cache,a.out,a.smoke)
