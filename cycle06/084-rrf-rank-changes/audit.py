"""Independent exact-rational fusion and metric audit; Python standard library only."""
import argparse,csv,gzip,json,math
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def order(r):return sorted(r,key=lambda d:(r[d],d),reverse=True)
def metrics(gold,ranking):
    rel={d for d,x in gold.items() if x>0}
    dcg=sum(1/math.log2(i+1) for i,d in enumerate(ranking[:10],1) if d in rel)
    ideal=sum(1/math.log2(i+1) for i in range(1,min(10,len(rel))+1))
    return {'ndcg_cut_10':dcg/ideal,'recall_100':len(set(ranking[:100])&rel)/len(rel),'mrr_10':next((1/i for i,d in enumerate(ranking[:10],1) if d in rel),0.)}
def audit(out):
    bm=read(ROOT/'inputs/bm25.json');dense=read(ROOT/'inputs/dense.json');gold=read(ROOT/'inputs/qrels.json');per=read(out/'per_query.json');summary=read(out/'summary.json')
    with gzip.open(out/'runs.json.gz','rt') as f:runs=json.load(f)
    maxerr=0.;pairs=0;rank_mismatches=[]
    for name,k,depth in summary['settings']:
        for q,saved in runs[name].items():
            scores={}
            ranks=[{d:i for i,d in enumerate(order(src[q])[:depth],1)} for src in [bm,dense]]
            for source in [bm,dense]:
                for i,d in enumerate(order(source[q])[:depth],1):scores[d]=scores.get(d,Fraction(0))+Fraction(1,k+i)
            expected=sorted(scores,key=lambda d:(float(scores[d]),d),reverse=True)[:100]
            actual=order(saved)
            # Rationally equal scores can round differently after two float additions.
            # Require official score agreement and audit actual finite-precision order separately.
            if expected!=actual:rank_mismatches.append({'method':name,'query_id':q,'metric_changes':{key:metrics(gold[q],expected)[key]-metrics(gold[q],actual)[key] for key in ['ndcg_cut_10','recall_100','mrr_10']}})
            direct={d:sum(1/(k+r[d]) for r in ranks if d in r) for d in scores}
            assert order(direct)[:100]==actual,(name,q)
            for d,v in saved.items():maxerr=max(maxerr,abs(v-float(scores[d])));pairs+=1
            assert maxerr<1e-14
    for name,run in runs.items():
        rows=[]
        for q,saved in run.items():
            actual=metrics(gold[q],order(saved));rows.append(actual)
            for key,v in actual.items():assert abs(v-per[name][q][key])<1e-12
        for key in rows[0]:assert abs(sum(x[key] for x in rows)/len(rows)-summary['methods'][name][key])<1e-12
    assert set(runs['rrf60_depth50'])==set(runs['alternate50'])
    assert all(set(runs['rrf60_depth50'][q])==set(runs['alternate50'][q]) for q in runs['alternate50'])
    trace_rows=0
    source_ranks={q:[{d:i for i,d in enumerate(order(src[q]),1)} for src in [bm,dense]] for q in runs['bm25']}
    with gzip.open(out/'trace.csv.gz','rt') as f:
        for row in csv.DictReader(f):
            q,d=row['query_id'],row['doc_id'];ranks=source_ranks[q]
            contributions=[1/(60+r[d]) if d in r else 0. for r in ranks]
            assert abs(sum(contributions)-float(row['rrf_score']))<1e-14
            assert [str(r.get(d,'')) for r in ranks]==[row['bm25_rank'],row['dense_rank']]
            assert int(row['relevant'])==int(gold[q].get(d,0)>0)
            assert int(row['output100'])==int(int(row['union_rank'])<=100)
            trace_rows+=1
    result={'passed':True,'queries':len(runs['bm25']),'methods':len(runs),'rrf_score_checks':pairs,'max_score_error':maxerr,'rational_rounding_rank_differences':rank_mismatches,'trace_rows_checked':trace_rows,'all_saved_metrics_independently_recomputed':True,'depth50_same_candidate_sets':True}
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);audit(p.parse_args().out)
