"""Frozen BEIR BM25 + real MiniLM + official Pyserini RRF target transfer."""
import argparse,collections,hashlib,importlib.metadata,json,platform,resource,sys,time,traceback
from pathlib import Path
from common import ROOT,read_json as read,write_json as write,sha,ranked,metrics,verify_resources,qrels

def summarize(per):
 n=len(per['bm25']);out={}
 for name,rows in per.items():
  ds=[rows[q]['ndcg_cut_10']-per['bm25'][q]['ndcg_cut_10'] for q in rows]
  out[name]={m:sum(r[m] for r in rows.values())/n for m in ['ndcg_cut_10','recall_100','mrr_10']}
  out[name].update(delta_vs_bm25=sum(ds)/n,win_tie_loss=[sum(d>1e-12 for d in ds),sum(abs(d)<=1e-12 for d in ds),sum(d< -1e-12 for d in ds)])
 return out

def main(a):
 start=time.perf_counter();a.out.mkdir(parents=True,exist_ok=False)
 state={'command':['python','run.py','--cache','<verified-cache>','--out',a.out.name]+(['--smoke'] if a.smoke else []),'exit_code':None,'stages':{}}
 try:
  cfg=read(ROOT/'config.json');lock,upstream=verify_resources(a.cache);ml=read(ROOT/'model.lock.json');mp=a.cache/'models'/cfg['revision']
  for f,v in ml['files'].items():assert sha(mp/f)==v['sha256']
  state['stages']['download']={'status':'passed','mode':'verified cached BEIR archive, data and model; network timing in acquisition record'}
  import torch,numpy as np,pytrec_eval
  sys.path.insert(0,str(upstream.resolve()))
  from beir.datasets.data_loader import GenericDataLoader
  from beir.retrieval.search.dense import DenseRetrievalExactSearch
  from sentence_transformers import SentenceTransformer
  from compat import RunnableBM25
  from encoder import Encoder
  from fusion import official_modules
  torch.set_num_threads(cfg['threads']);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);torch.use_deterministic_algorithms(True)
  t=time.perf_counter();corpus,queries,gold=GenericDataLoader(str(a.cache/'nfcorpus')).load(split='test')
  assert len(corpus)==3633 and len(queries)==323
  queries=dict(sorted(queries.items()));original_counts={'documents':len(corpus),'queries':len(queries)}
  if a.smoke:queries=dict(list(queries.items())[:5]);corpus=dict(list(corpus.items())[:32])
  gold={q:gold[q] for q in queries};assert not set(corpus)&set(queries)
  grades=collections.Counter(s for r in gold.values() for s in r.values());assert all(s>=0 for s in grades)
  for split in ['train','dev']:assert not set(queries)&set(qrels(a.cache/f'nfcorpus/qrels/{split}.tsv'))
  counts=[sum(s>0 for s in r.values()) for r in gold.values()];assert min(counts)>0
  write(a.out/'data_audit.json',{'full':original_counts,'actual':{'documents':len(corpus),'queries':len(queries)},'label_counts':dict(grades),'positive_per_query':{'min':min(counts),'max':max(counts),'mean':sum(counts)/len(counts)},'train_dev_query_overlap':0,'query_doc_collision':0,'grades_binarized':False,'test_access':'all target test used once as frozen transfer; smoke subset overlaps but no tuning; no untouched target claim after run'})
  identity={'config':cfg,'data':lock['data_files'],'model_tokenizer':ml,'prompt':None,'decoding':None,'beir':lock['beir_commit'],'protocol':sha(ROOT/'PROTOCOL.md'),'code':{str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*.py')) if '__pycache__' not in str(p)},'requirements':sha(ROOT/'requirements.txt'),'smoke':a.smoke,'query_ids':list(queries),'corpus_ids':list(corpus)}
  key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();write(a.out/'identity.json',{'cache_key':key,**identity})
  state['stages']['preprocessing']={'status':'passed','seconds':time.perf_counter()-t}
  t=time.perf_counter();bm=RunnableBM25(index_name='cycle6-087-'+key[:12],hostname=a.hostname,initialize=False,number_of_shards=1,language='english',batch_size=64);es=bm.es.es;info=es.info()
  assert info['version']['number']==cfg['es_version'] and info['version']['build_hash']==cfg['es_build'];write(a.out/'es_version.json',info['version'])
  index=bm.es.index_name
  if es.indices.exists(index=index):
   from elasticsearch.helpers import scan
   stored={r['_id']:r['_source'] for r in scan(es,index=index)};assert set(stored)==set(corpus)
   assert all(stored[d]['title']==corpus[d]['title'] and stored[d]['txt']==corpus[d]['text'] for d in corpus)
  else:bm.es.create_index();bm.index(corpus)
  es.indices.refresh(index=index);assert es.count(index=index)['count']==len(corpus)
  write(a.out/'index_settings.json',es.indices.get(index=index))
  state['stages']['indexing']={'status':'passed','seconds':time.perf_counter()-t}
  t=time.perf_counter();raw=bm.search(corpus,queries,top_k=100);write(a.out/'bm25_raw.json',raw);assert set(raw)<=set(queries)
  missing=sorted(set(queries)-set(raw));write(a.out/'bm25_missing.json',{'missing_query_ids':missing,'handling':'retain missing queries as empty runs; denominator unchanged'})
  raw={q:raw.get(q,{}) for q in queries}
  br={q:{d:raw[q][d] for d in ranked(raw[q])[:100]} for q in queries}
  state['stages']['bm25_retrieval']={'status':'passed','seconds':time.perf_counter()-t}
  t=time.perf_counter();model=SentenceTransformer(str(mp),device='cpu',local_files_only=True,trust_remote_code=False);model.max_seq_length=256;model.eval();model.requires_grad_(False)
  texts=['A scientific claim.','Protein expression changes under treatment.'];tok=model.tokenizer(texts,padding=True,truncation=True,max_length=256,return_tensors='pt')
  with torch.inference_mode():
   h=model[0].auto_model(**tok).last_hidden_state;pool=(h*tok['attention_mask'][...,None]).sum(1)/tok['attention_mask'].sum(1)[:,None];direct=torch.nn.functional.normalize(pool,dim=1);official=model.encode(texts,convert_to_tensor=True)
  err=float((direct-official).abs().max());assert err<1e-5
  write(a.out/'shape_check.json',{'input_ids':list(tok['input_ids'].shape),'hidden':list(h.shape),'pooled':list(pool.shape),'masked_mean_max_error':err,'parameters':sum(p.numel() for p in model.parameters()),'training':model.training})
  ids={'queries':list(queries),'corpus':sorted(corpus,key=lambda d:len(corpus[d].get('title','')+corpus[d]['text']),reverse=True)}
  enc=Encoder(model,a.out,ids);enc.embedding_cache=a.cache/'embeddings'/key;enc.embedding_cache.mkdir(parents=True,exist_ok=True);write(a.out/'ids.json',ids)
  state['stages']['model_load_shape']={'status':'passed','seconds':time.perf_counter()-t}
  t=time.perf_counter();dr=DenseRetrievalExactSearch(enc,batch_size=16,corpus_chunk_size=50000,show_progress_bar=False).search(corpus,queries,top_k=100,score_function='cos_sim')
  state['stages']['dense_inference']={'status':'passed','seconds':time.perf_counter()-t};print('dense',state['stages']['dense_inference'],flush=True)
  write(a.out/'embedding_hashes.json',{n:sha(enc.embedding_cache/n) for n in ['queries.npy','corpus.npy']})
  t=time.perf_counter();trec,fuse=official_modules()
  def make(r):return trec.TrecRun.from_list([(q,'Q0',d,i,r[q][d],'frozen') for q in queries for i,d in enumerate(ranked(r[q]),1)])
  rr=fuse.reciprocal_rank_fusion([make(br),make(dr)],rrf_k=60,depth=100,k=None);full={q:{} for q in queries}
  for r in rr.run_data.itertuples(index=False):full[r.topic][r.docid]=float(r.score)
  fr={q:{d:full[q][d] for d in ranked(full[q])[:100]} for q in queries};write(a.out/'rrf_union.json',full)
  state['stages']['fusion']={'status':'passed','seconds':time.perf_counter()-t}
  t=time.perf_counter();runs={'bm25':br,'dense':dr,'rrf60':fr};write(a.out/'runs.json',runs)
  ev=pytrec_eval.RelevanceEvaluator(gold,{'ndcg_cut.10','recall.100'});per={};error=0;diagnostics={}
  for name,run in runs.items():
   per[name]=ev.evaluate(run)
   for q in queries:
    if q not in per[name]:
     assert not run[q];per[name][q]={'ndcg_cut_10':0.0,'recall_100':0.0}
   assert set(per[name])==set(queries)
   for q in queries:
    independent=metrics(gold[q],ranked(run[q]));error=max(error,max(abs(v-independent[k]) for k,v in per[name][q].items()));per[name][q]['mrr_10']=independent['mrr_10']
   diagnostics[name]={'no_relevant_top10':sum(not any(gold[q].get(d,0)>0 for d in ranked(run[q])[:10]) for q in queries),'no_relevant_top100':sum(not any(gold[q].get(d,0)>0 for d in run[q]) for q in queries)}
  assert error<1e-10;write(a.out/'per_query.json',per)
  contrasts={name:[{'query_id':q,'delta':per[name][q]['ndcg_cut_10']-per['bm25'][q]['ndcg_cut_10']} for q in queries] for name in ['dense','rrf60']};write(a.out/'contrasts.json',contrasts)
  source=read(ROOT/'inputs/source_per_query.json');summary={'scope':'SMOKE reduced corpus; no benchmark' if a.smoke else 'All323 BEIR NFCorpus test queries; full3633documents; frozen single configuration','source':summarize(source),'target':summarize(per),'independent_metric_max_error':error,'failure_counts':diagnostics,'rrf_union_rows':sum(map(len,full.values())),'target_tuning':False,'source_confirmation_used':False};write(a.out/'summary.json',summary)
  cases={n:{'rule':'three largest gains/losses automatically selected, no human annotation','losses':sorted(r,key=lambda x:(x['delta'],x['query_id']))[:3],'gains':sorted(r,key=lambda x:(-x['delta'],x['query_id']))[:3]} for n,r in contrasts.items()};write(a.out/'cases.json',cases)
  state['stages']['evaluation']={'status':'passed','seconds':time.perf_counter()-t,'metric_max_error':error};state['exit_code']=0
  # Only resource numbers are retained; machine identifiers and log paths omitted.
  node=next(iter(es.nodes.stats(metric='jvm')['nodes'].values()));write(a.out/'server_memory.json',{'heap_used_bytes':node['jvm']['mem']['heap_used_in_bytes'],'heap_max_bytes':node['jvm']['mem']['heap_max_in_bytes'],'server_peak_rss':'not measured'})
  print(json.dumps(summary,indent=2),flush=True)
 except Exception:
  state['exit_code']=1;state['error']=traceback.format_exc();raise
 finally:
  state['elapsed_seconds']=time.perf_counter()-start;write(a.out/'status.json',state)
  write(a.out/'environment.json',{'python':sys.version,'platform':platform.platform(),'device':'CPU','dtype':'float32','peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},'server_peak_rss':'not measured; separate JVM snapshot','timing':'excludes downloads and server startup; source inference reused, not timed','gpu_memory':'not applicable'})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--hostname',default='http://127.0.0.1:19282');p.add_argument('--smoke',action='store_true');main(p.parse_args())
