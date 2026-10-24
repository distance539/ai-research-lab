"""Independent stdlib reconstruction from CSV; does not import ranking/metric helpers."""
import argparse,csv,json,math,hashlib
from pathlib import Path
from fractions import Fraction
ROOT=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_text())
def main(out,save=True):
 gold=load(ROOT/'inputs/qrels.json');cfg=load(ROOT/'config.json');assert cfg['fusion_constant']==60 and cfg['fusion_weights']==[1,1]
 rows=list(csv.DictReader((out/'pair_scores.csv').open()));runs=load(out/'runs.json');br=load(out/'blend_runs.json');per=load(out/'comparison_per_query.json');summary=load(out/'comparison_summary.json');budget=load(out/'budget_ledger.json');trace=load(out/'fusion_trace.json')
 candidates=load(ROOT/'inputs/candidates.json');qids=list(runs['original']);expected=cfg['query_ids'][:5] if load(out/'status.json')['smoke'] else cfg['query_ids'];assert set(qids)==set(expected)
 order=lambda s:sorted(s,key=lambda d:(s[d],d),reverse=True)
 def metrics(q,rank):
  rel={d for d,g in gold[q].items() if g>0};hits=[i for i,d in enumerate(rank,1) if d in rel];ideal=sum(1/math.log2(i+1) for i in range(1,min(10,len(rel))+1))
  return {'ndcg_cut_10':sum(1/math.log2(i+1) for i in hits if i<=10)/ideal,'recall_10':sum(i<=10 for i in hits)/len(rel),'recall_50':len(hits)/len(rel),'mrr_10':1/min(hits) if hits and min(hits)<=10 else 0}
 def close(x,y):assert math.isfinite(x) and math.isfinite(y) and abs(x-y)<1e-12,(x,y)
 seen=set();maxerr=0.;qmetrics={};ds={k:[] for k in ['blend_minus_ce','blend_minus_rrf']};records=[];tie_pairs=0
 for q in qids:
  selected=[x for x in rows if x['query_id']==q];scores={x['doc_id']:float(x['logit']) for x in selected};k=10 if len(expected)==5 else 50
  assert len(selected)==len(scores)==k and set(scores)==set(order(candidates[q])[:k])
  assert scores==runs['reranked'][q] and {d:candidates[q][d] for d in scores}==runs['original'][q]
  a=order(runs['original'][q]);b=order(scores);f={d:float(Fraction(1,60+a.index(d)+1)+Fraction(1,60+b.index(d)+1)) for d in a}
  assert set(br[q])==set(f)
  for d in f:close(f[d],br[q][d]);maxerr=max(maxerr,abs(f[d]-br[q][d]))
  # Audit intended operational float-sum tie rule, rational sums independently check scores.
  ff={d:1/(61+a.index(d))+1/(61+b.index(d)) for d in a};rank=order(ff);assert rank==order(br[q]);tie_pairs+=len(f)-len(set(ff.values()))
  m={n:metrics(q,r) for n,r in [('rrf',a),('ce',b),('blend',rank)]};qmetrics[q]=m
  p=next(x for x in per if x['query_id']==q);record={n:m[n]['ndcg_cut_10'] for n in m};records.append(record)
  for n in m:close(p[n],m[n]['ndcg_cut_10']);assert p['ranks'][n]==next((i for i,d in enumerate({'rrf':a,'ce':b,'blend':rank}[n],1) if gold[q].get(d,0)>0),None)
  for key,n in [('blend_minus_ce','ce'),('blend_minus_rrf','rrf')]:v=m['blend']['ndcg_cut_10']-m[n]['ndcg_cut_10'];close(p[key],v);ds[key].append(v)
  close(p['ce_minus_rrf'],m['ce']['ndcg_cut_10']-m['rrf']['ndcg_cut_10'])
  for x in selected:
   d=x['doc_id'];assert (q,d) not in seen;seen.add((q,d));assert int(x['original_rank'])==a.index(d)+1 and int(x['reranked_rank'])==b.index(d)+1;assert int(x['retained_tokens'])==min(512,int(x['raw_tokens']))
   tr=next(t for t in trace if t['query_id']==q and t['doc_id']==d);assert tr['rrf_rank']==a.index(d)+1 and tr['ce_rank']==b.index(d)+1;close(tr['blend_score'],ff[d])
 assert len(per)==len(qids) and len(trace)==len(rows)
 for n in ['rrf','ce','blend']:
  for key in qmetrics[qids[0]][n]:close(summary['methods'][n][key],sum(qmetrics[q][n][key] for q in qids)/len(qids))
  assert summary['top10_misses'][n]==sum(qmetrics[q][n]['recall_10']==0 for q in qids)
 for key,values in ds.items():
  close(summary['comparisons'][key]['mean'],sum(values)/len(values));assert summary['comparisons'][key]['win_tie_loss']==[sum(v>1e-12 for v in values),sum(abs(v)<=1e-12 for v in values),sum(v< -1e-12 for v in values)]
 losses=[x for x in records if x['ce']<x['rrf']-1e-12];gains=[x for x in records if x['ce']>x['rrf']+1e-12]
 assert summary['override_tradeoff']=={'old_ce_losses':len(losses),'improved_vs_ce':sum(x['blend']>x['ce']+1e-12 for x in losses),'recovered_to_rrf_or_better':sum(x['blend']>=x['rrf']-1e-12 for x in losses),'old_ce_gains':len(gains),'old_gains_reduced_by_blend':sum(x['blend']<x['ce']-1e-12 for x in gains),'new_harm_vs_rrf_on_nonlosses':sum(x['ce']>=x['rrf']-1e-12 and x['blend']<x['rrf']-1e-12 for x in records)}
 assert budget['candidate_pairs']==budget['ce_pairs']==budget['blend_ce_pairs']==len(rows)
 assert budget['ce_effective_tokens']==budget['blend_effective_tokens']==sum(int(x['retained_tokens']) for x in rows)
 assert budget['training_tokens']==budget['blend_extra_model_pairs']==budget['rrf_additional_ce_pairs']==0 and summary['confirmation_evaluated'] is False
 result={'passed':True,'queries':len(qids),'pairs':len(rows),'max_reciprocal_score_error':maxerr,'duplicate_fusion_scores':tie_pairs,'raw_pairs_sha256':hashlib.sha256((out/'pair_scores.csv').read_bytes()).hexdigest(),'independent_metrics_budgets_and_tradeoffs':True}
 if save:(out/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);print(main(p.parse_args().out))
