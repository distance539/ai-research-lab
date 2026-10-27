"""Official frozen BM25+dense+RRF on full corpus. No evaluation/tuning here."""
import argparse,json,time,sys,resource,traceback,importlib.util
from pathlib import Path
from common import ROOT,read,write,sha,order

def load_vendor():
 for name,f in [('pyserini.trectools','trectools_base.py'),('_fusion','fusion_base.py')]:
  spec=importlib.util.spec_from_file_location(name,ROOT/'vendor'/f);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
 return sys.modules['pyserini.trectools'],sys.modules['_fusion']
class Encoder:
 def __init__(self,m,out):self.model=m;self.out=out
 def encode_queries(self,qs,**kw):return self.model.encode(qs,**kw)
 def encode_corpus(self,corpus,**kw):return self.model.encode([(x.get('title','')+' '+x['text']).strip() for x in corpus],**kw)
def main(a):
 t0=time.perf_counter();a.out.mkdir(parents=True,exist_ok=False);status={'exit_code':None,'stages':{},'mode':'dev preflight' if a.smoke else 'confirmation','command':['python','retrieve.py','--cache','<verified-cache>','--out',a.out.name]+(['--smoke'] if a.smoke else [])}
 try:
  cfg=read(ROOT/'config.json');lock=read(ROOT/'resources.lock.json');up=a.cache/'upstream'/('beir-'+lock['beir_commit'])
  for f,h in lock['beir_files'].items():assert sha(up/f)==h,f
  for f,h in lock['data_files'].items():assert sha(a.cache/f)==h,f
  ml=read(ROOT/'dense_model.lock.json');mp=a.cache/'models'/ml['revision']
  for f,r in ml['files'].items():assert sha(mp/f)==r['sha256'],f
  sys.path.insert(0,str(up.resolve()))
  import torch,numpy as np
  from sentence_transformers import SentenceTransformer
  from beir.retrieval.search.dense import DenseRetrievalExactSearch
  from compat import RunnableBM25
  from elasticsearch.helpers import scan
  torch.set_num_threads(4);torch.manual_seed(111);np.random.seed(111);torch.use_deterministic_algorithms(True)
  corpus={r['_id']:r for r in map(json.loads,(a.cache/'scifact/corpus.jsonl').read_text().splitlines())};qs={r['_id']:r['text'] for r in map(json.loads,(a.cache/'scifact/queries.jsonl').read_text().splitlines())}
  ids=cfg['smoke_query_ids'] if a.smoke else cfg['query_ids'];queries={q:qs[q] for q in ids};assert len(corpus)==5183
  index='cycle6-111-'+lock['data_files']['scifact/corpus.jsonl'][:12]
  bm=RunnableBM25(index_name=index,hostname=a.hostname,initialize=False,number_of_shards=1,language='english',batch_size=64);es=bm.es.es
  info=es.info();assert info['version']['number']=='7.17.9' and info['version']['build_hash']==read(ROOT/'bm25_config.json')['model_revision'];write(a.out/'es_info.json',info)
  t=time.perf_counter();exists=es.indices.exists(index=index)
  if not exists:bm.es.create_index();bm.index(corpus)
  else:
   props=es.indices.get_mapping(index=index)[index]['mappings']['properties'];assert props['title']['analyzer']==props['txt']['analyzer']=='english'
   stored={x['_id']:x['_source'] for x in scan(es,index=index)};assert set(stored)==set(corpus)
   for d in corpus:assert stored[d]['title']==corpus[d]['title'] and stored[d]['txt']==corpus[d]['text']
  es.indices.refresh(index=index);assert es.count(index=index)['count']==5183
  write(a.out/'index_settings.json',es.indices.get(index=index));status['stages']['index']={'seconds':time.perf_counter()-t,'reused':exists}
  t=time.perf_counter();raw=bm.search(corpus,queries,top_k=100);bmrun={q:{d:raw[q][d] for d in order(raw[q])[:100]} for q in ids};write(a.out/'bm25_raw.json',raw);write(a.out/'bm25.json',bmrun);status['stages']['bm25']={'seconds':time.perf_counter()-t}
  t=time.perf_counter();m=SentenceTransformer(str(mp),device='cpu',local_files_only=True,trust_remote_code=False);m.max_seq_length=256;m.eval();m.requires_grad_(False)
  dense=DenseRetrievalExactSearch(Encoder(m,a.out),batch_size=16,corpus_chunk_size=50000,show_progress_bar=False).search(corpus,queries,top_k=100,score_function='cos_sim')
  write(a.out/'dense.json',dense);status['stages']['dense']={'seconds':time.perf_counter()-t,'corpus':5183,'queries':len(ids),'dimensions':384,'max_tokens':256,'fresh_corpus_and_queries':True}
  trec,fusion=load_vendor()
  make=lambda run:trec.TrecRun.from_list([(q,'Q0',d,i,run[q][d],'111') for q in ids for i,d in enumerate(order(run[q]),1)])
  t=time.perf_counter();r=fusion.reciprocal_rank_fusion([make(bmrun),make(dense)],rrf_k=60,depth=100,k=None);rr={q:{} for q in ids}
  for row in r.run_data.itertuples(index=False):rr[row.topic][row.docid]=float(row.score)
  candidates={q:{d:rr[q][d] for d in order(rr[q])[:50]} for q in ids};assert all(len(x)==50 for x in candidates.values())
  write(a.out/'rrf_union.json',rr);write(a.out/'candidates.json',candidates);status['stages']['fusion']={'seconds':time.perf_counter()-t,'queries':len(ids),'pairs':50*len(ids)}
  identity={'config':cfg,'model':ml,'data':lock['data_files'],'bm25':read(ROOT/'bm25_config.json'),'code':{str(p.relative_to(ROOT)):sha(p) for p in list(ROOT.glob('*.py'))+list((ROOT/'vendor').glob('*.py'))},'query_ids':ids,'prompt':None,'decoding':None,'beir_commit':lock['beir_commit']};import hashlib
  identity['cache_key']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();write(a.out/'identity.json',identity)
  write(a.out/'server_resource_snapshot.json',es.nodes.stats(metric='jvm,process'));status['exit_code']=0
 except Exception:status['exit_code']=1;status['error']=traceback.format_exc();raise
 finally:
  status['elapsed_seconds']=time.perf_counter()-t0;status['python_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024);status['es_peak_rss']='not measured; separate snapshot';write(a.out/'status.json',status);print(json.dumps(status,indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--hostname',default='http://127.0.0.1:19282');main(p.parse_args())
