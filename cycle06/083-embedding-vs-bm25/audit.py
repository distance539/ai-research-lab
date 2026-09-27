"""Independent per-query audit; optional exhaustive saved-embedding score audit."""
import argparse,csv,math
from pathlib import Path
from common import ROOT,read_json,write_json,ranked,metrics,sha

def audit(run,embedding_audit=False,output=None):
    dense=read_json(run/'results.json');bm=read_json(ROOT/'baseline/results.json');gold=read_json(run/'qrels_used.json')
    assert gold==read_json(ROOT/'baseline/qrels_used.json')
    assert set(dense)==set(bm)==set(gold) and len(gold)==809
    rows=[]; per=read_json(run/'per_query.json');worst=0
    for q in sorted(gold,key=int):
        assert len(dense[q])==100 and len(bm[q])<=100
        ds,bs=ranked(dense[q]),ranked(bm[q]);dm,bm_=metrics(gold[q],ds),metrics(gold[q],bs)
        worst=max(worst,max(abs(dm[k]-per[q][k]) for k in dm))
        ranks={d:{'dense':ds.index(d)+1 if d in ds else None,'bm25':bs.index(d)+1 if d in bs else None} for d in gold[q]}
        rows.append({'query_id':q,'dense':dm,'bm25':bm_,'delta':{k:dm[k]-bm_[k] for k in dm},'relevant_ranks':ranks})
    assert worst<1e-10
    summary={}
    for k in rows[0]['delta']:
        values=[x['delta'][k] for x in rows]
        summary[k]={'dense':sum(x['dense'][k] for x in rows)/len(rows),'bm25':sum(x['bm25'][k] for x in rows)/len(rows),'delta':sum(values)/len(rows),'wins':sum(x>1e-12 for x in values),'ties':sum(abs(x)<=1e-12 for x in values),'losses':sum(x< -1e-12 for x in values),'positive_contribution':sum(x for x in values if x>0)/len(rows),'negative_contribution':sum(x for x in values if x<0)/len(rows)}
    patterns={}
    for x in rows:
        key=('dense_hit' if x['dense']['recall_100']>0 else 'dense_miss')+'__'+('bm25_hit' if x['bm25']['recall_100']>0 else 'bm25_miss')
        patterns[key]=patterns.get(key,0)+1
    ordered=sorted(rows,key=lambda x:(-x['delta']['ndcg_cut_10'],int(x['query_id'])))
    selected={'largest_gains':ordered[:3],'largest_losses':sorted(rows,key=lambda x:(x['delta']['ndcg_cut_10'],int(x['query_id'])))[:3]}
    result={'queries':len(rows),'metric_max_error':worst,'summary':summary,'hit100_patterns':patterns,'selected_cases':selected,'embedding_audit':'not requested','inference_rerun':False}
    if embedding_audit:
        import numpy as np
        q=np.load(run/'queries.npy').astype('float64');d=np.load(run/'corpus.npy').astype('float64');ids=read_json(run/'ids.json')
        q/=np.linalg.norm(q,axis=1,keepdims=True);d/=np.linalg.norm(d,axis=1,keepdims=True)
        maxerr=0.; minmargin=1.; mismatches=[]
        for i,qid in enumerate(ids['queries']):
            scores=q[i]@d.T
            ix=sorted(range(len(d)),key=lambda j:(float(scores[j]),ids['corpus'][j]),reverse=True)
            saved=ranked(dense[qid]);expected=[ids['corpus'][j] for j in ix[:100]]
            if expected!=saved:mismatches.append(qid)
            lookup={doc:j for j,doc in enumerate(ids['corpus'])}
            maxerr=max(maxerr,max(abs(dense[qid][doc]-scores[lookup[doc]]) for doc in saved))
            minmargin=min(minmargin,float(scores[ix[99]]-scores[ix[100]]))
        assert maxerr<1e-5
        # Float32/float64 can reorder near ties; retain differences instead of inventing equality.
        result['embedding_audit']={'shapes':[list(q.shape),list(d.shape)],'pair_scores':len(q)*len(d),'float64_max_score_error':maxerr,'queries_with_top100_order_difference':mismatches,'min_rank100_101_margin':minmargin,'input_sha256':{f:sha(run/f) for f in ['queries.npy','corpus.npy']}}
    dest=output or run;dest.mkdir(parents=True,exist_ok=True)
    write_json(dest/'paired.json',rows);write_json(dest/'comparison.json',result)
    with (dest/'paired.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['query_id','bm25_ndcg10','dense_ndcg10','delta_ndcg10','bm25_recall100','dense_recall100'])
        for x in rows:w.writerow([x['query_id'],x['bm25']['ndcg_cut_10'],x['dense']['ndcg_cut_10'],x['delta']['ndcg_cut_10'],x['bm25']['recall_100'],x['dense']['recall_100']])
    print(result['summary']['ndcg_cut_10']);print(result['hit100_patterns']);return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=ROOT/'results/development');p.add_argument('--embedding-audit',action='store_true');p.add_argument('--out',type=Path);a=p.parse_args();audit(a.run,a.embedding_audit,a.out)
