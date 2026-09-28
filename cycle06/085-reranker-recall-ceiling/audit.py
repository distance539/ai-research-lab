"""Independent metric/oracle and pair-coverage audit (stdlib only)."""
import argparse,csv,math
from pathlib import Path
from common import ROOT,read,write,metric,order

def main(out):
 runs=read(out/'runs.json');per=read(out/'per_query.json');summary=read(out/'summary.json');gold=read(ROOT/'inputs/qrels.json');diag=read(out/'diagnostics.json');maxerr=0.
 for name,run in runs.items():
  for q,scores in run.items():
   assert set(scores)==set(runs['original'][q]);assert all(math.isfinite(s) for s in scores.values())
   m=metric(gold[q],order(scores))
   for key,v in m.items():maxerr=max(maxerr,abs(v-per[name][q][key]))
   assert abs(m['recall_50']-per['original'][q]['recall_50'])<1e-12
  for key in m:assert abs(sum(metric(gold[q],order(run[q]))[key] for q in run)/len(run)-summary[name][key])<1e-12
 for x in diag:
  q=x['query_id'];r={d for d,v in gold[q].items() if v>0};n=len(r&set(runs['original'][q]));idcg=sum(1/math.log2(i+1) for i in range(1,min(10,len(r))+1));upper=sum(1/math.log2(i+1) for i in range(1,min(10,n)+1))/idcg
  assert abs(upper-x['oracle'])<1e-12;assert abs(x['candidate_loss']+x['ordering_loss']-(1-x['reranked']))<1e-12
 rows=list(csv.DictReader((out/'pair_scores.csv').open()));pairs={(r['query_id'],r['doc_id']) for r in rows};assert len(pairs)==len(rows)==summary['pairs']
 for r in rows:
  q,d=r['query_id'],r['doc_id'];assert runs['reranked'][q][d]==float(r['logit']);assert order(runs['reranked'][q]).index(d)+1==int(r['reranked_rank']);assert order(runs['original'][q]).index(d)+1==int(r['original_rank']);assert int(r['relevant'])==int(gold[q].get(d,0)>0)
 assert maxerr<1e-12
 result={'passed':True,'queries':len(diag),'pairs':len(rows),'max_metric_error':maxerr,'same_candidates':True,'oracle_closed_form_passed':True,'loss_decomposition_passed':True,'all_metrics_recomputed':True,'pair_scores_and_ranks_checked':True}
 write(out/'audit.json',result);print(result)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
