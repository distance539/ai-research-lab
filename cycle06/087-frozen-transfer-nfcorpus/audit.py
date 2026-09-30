"""Independent standard-library checks of real rankings, graded metrics and RRF."""
import argparse,collections,math
from pathlib import Path
from common import ROOT,read_json as read,write_json as write,ranked,qrels,metrics,sha

def audit(cache,out):
 lock=read(ROOT/'resources.lock.json')
 for p,h in lock['data_files'].items():assert sha(cache/p)==h
 gold=qrels(cache/'nfcorpus/qrels/test.tsv');runs=read(out/'runs.json');per=read(out/'per_query.json');full=read(out/'rrf_union.json');ids=read(out/'identity.json')['query_ids'];gold={q:gold[q] for q in ids};maxerr=0.;scores=0
 assert len(ids)==len(set(ids)) and set(runs)=={'bm25','dense','rrf60'}
 for n,r in runs.items():
  assert set(r)==set(ids)
  for q in ids:
   assert len(r[q])<=100
   expected=metrics(gold[q],ranked(r[q]));maxerr=max(maxerr,max(abs(expected[m]-v) for m,v in per[n][q].items()))
 for q in ids:
  expected={}
  for n in ['bm25','dense']:
   for i,d in enumerate(ranked(runs[n][q])[:100],1):expected[d]=expected.get(d,0)+1/(60+i)
  assert set(full[q])==set(expected)
  for d,v in expected.items():maxerr=max(maxerr,abs(v-full[q][d]));scores+=1
  assert ranked(full[q])[:100]==ranked(runs['rrf60'][q])
 summary=read(out/'summary.json')
 for n in runs:
  for m in ['ndcg_cut_10','recall_100','mrr_10']:maxerr=max(maxerr,abs(sum(per[n][q][m] for q in ids)/len(ids)-summary['target'][n][m]))
  delta=[per[n][q]['ndcg_cut_10']-per['bm25'][q]['ndcg_cut_10'] for q in ids]
  assert summary['target'][n]['win_tie_loss']==[sum(x>1e-12 for x in delta),sum(abs(x)<=1e-12 for x in delta),sum(x< -1e-12 for x in delta)]
  assert abs(sum(delta)/len(ids)-summary['target'][n]['delta_vs_bm25'])<1e-12
 assert maxerr<1e-10
 # Explicit graded fixture distinguishes linear from binary or exponential gain.
 fixture=metrics({'a':2,'b':1},['b','a'])['ndcg_cut_10'];assert abs(fixture-(1+2/math.log2(3))/(2+1/math.log2(3)))<1e-15
 missing=read(out/'bm25_missing.json')['missing_query_ids'];assert all(not runs['bm25'][q] and per['bm25'][q]['ndcg_cut_10']==0 for q in missing)
 rows=[__import__('json').loads(x) for x in (cache/'nfcorpus/corpus.jsonl').read_text().splitlines()];assert len(rows)==len({x['_id'] for x in rows})==3633
 qrows=[__import__('json').loads(x) for x in (cache/'nfcorpus/queries.jsonl').read_text().splitlines()];assert len(qrows)==len({x['_id'] for x in qrows});docs={x['_id'] for x in rows}
 assert all(set(r)<=docs for r in gold.values())
 report={'status':'passed','queries':len(ids),'methods':len(runs),'rrf_scores_recomputed':scores,'maximum_error':maxerr,'linear_gain_fixture':fixture,'empty_bm25_queries_retained':len(missing),'positive_qrels':sum(sum(v>0 for v in g.values()) for g in gold.values()),'duplicate_corpus_query_ids':False,'qrel_doc_integrity':True,'scope':'offline numerical audit; real inference separately recorded'}
 write(out/'audit.json',report);print(report)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();audit(a.cache,a.out)
