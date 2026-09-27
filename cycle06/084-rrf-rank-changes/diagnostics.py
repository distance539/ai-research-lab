"""Describe paired deltas, score ties and rank movement from saved real logs."""
import argparse,csv,gzip,json
from pathlib import Path
from audit import metrics
ROOT=Path(__file__).resolve().parent
def run(out):
    per=json.loads((out/'per_query.json').read_text());gold=json.loads((ROOT/'inputs/qrels.json').read_text())
    with gzip.open(out/'runs.json.gz','rt') as f:runs=json.load(f)
    def paired(a,b):
        ds=[per[a][q]['ndcg_cut_10']-per[b][q]['ndcg_cut_10'] for q in per[a]]
        return {'delta':sum(ds)/len(ds),'wins':sum(x>1e-12 for x in ds),'ties':sum(abs(x)<=1e-12 for x in ds),'losses':sum(x< -1e-12 for x in ds),'positive_contribution':sum(x for x in ds if x>0)/len(ds),'negative_contribution':sum(x for x in ds if x<0)/len(ds)}
    trace={}
    with gzip.open(out/'trace.csv.gz','rt') as f:
        for row in csv.DictReader(f):trace.setdefault(row['query_id'],[]).append(row)
    asc=[];changed=0;shared_top10=0;exclusive_top10=0;gained10=0;lost10=0;relevant_pair_count=0
    for q,rows in trace.items():
        rank=sorted(rows,key=lambda x:(-float(x['rrf_score']),x['doc_id']))
        value=metrics(gold[q],[x['doc_id'] for x in rank[:100]])['ndcg_cut_10'];asc.append(value)
        changed+=abs(value-per['rrf60'][q]['ndcg_cut_10'])>1e-12
        for x in rows:
            if int(x['union_rank'])<=10:
                shared_top10+=bool(x['bm25_rank'] and x['dense_rank']);exclusive_top10+=not bool(x['bm25_rank'] and x['dense_rank'])
            if x['relevant']=='1':
                relevant_pair_count+=1;was=bool(x['bm25_rank'] and int(x['bm25_rank'])<=10);now=int(x['union_rank'])<=10
                gained10+=now and not was;lost10+=was and not now
    d={'paired_rrf60_vs_bm25':paired('rrf60','bm25'),'paired_rrf60_vs_alternate100':paired('rrf60','alternate100'),'paired_depth50_rrf_vs_alternate':paired('rrf60_depth50','alternate50'),'tie_policy_primary':{'ascending_ndcg10':sum(asc)/len(asc),'descending_ndcg10':sum(x['ndcg_cut_10'] for x in per['rrf60'].values())/len(asc),'queries_with_metric_change':changed},'rank_movement':{'shared_top10_pairs':shared_top10,'exclusive_top10_pairs':exclusive_top10,'relevant_union_pairs':relevant_pair_count,'relevant_pairs_enter_top10_vs_bm25':gained10,'relevant_pairs_leave_top10_vs_bm25':lost10}}
    (out/'diagnostics.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
