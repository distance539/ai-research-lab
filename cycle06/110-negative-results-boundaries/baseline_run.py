"""One real official CE pass, then three fixed-candidate ranking policies."""
import argparse,time,hashlib,json
from pathlib import Path
import ce_run
from common import ROOT,read,write,order,metric

def blend_report(out):
 cfg=read(ROOT/'config.json');runs=read(out/'runs.json');gold=read(ROOT/'inputs/qrels.json');qids=list(runs['original']);t=time.perf_counter()
 blend={};trace=[]
 for q in qids:
  a={d:i for i,d in enumerate(order(runs['original'][q]),1)};b={d:i for i,d in enumerate(order(runs['reranked'][q]),1)}
  blend[q]={d:1/(cfg['fusion_constant']+a[d])+1/(cfg['fusion_constant']+b[d]) for d in a}
  for d in a:trace.append({'query_id':q,'doc_id':d,'rrf_rank':a[d],'ce_rank':b[d],'blend_score':blend[q][d]})
  order(blend[q])
 seconds=time.perf_counter()-t
 methods={'rrf':runs['original'],'ce':runs['reranked'],'blend':blend};per={q:{m:metric(gold[q],order(v[q])) for m,v in methods.items()} for q in qids}
 paired=[]
 for q in qids:
  n={m:per[q][m]['ndcg_cut_10'] for m in methods};paired.append({'query_id':q,**n,'blend_minus_ce':n['blend']-n['ce'],'blend_minus_rrf':n['blend']-n['rrf'],'ce_minus_rrf':n['ce']-n['rrf'],'ranks':{m:next((i for i,d in enumerate(order(v[q]),1) if gold[q].get(d,0)>0),None) for m,v in methods.items()}})
 summary={'queries':len(qids),'methods':{m:{key:sum(per[q][m][key] for q in qids)/len(qids) for key in next(iter(per.values()))[m]} for m in methods},'comparisons':{},'top10_misses':{m:sum(per[q][m]['recall_10']==0 for q in qids) for m in methods},'confirmation_evaluated':False,'scope':'observed development only; no threshold success claim'}
 for k in ['blend_minus_ce','blend_minus_rrf']:
  ds=[x[k] for x in paired];summary['comparisons'][k]={'mean':sum(ds)/len(ds),'win_tie_loss':[sum(x>1e-12 for x in ds),sum(abs(x)<=1e-12 for x in ds),sum(x< -1e-12 for x in ds)]}
 oldloss=[x for x in paired if x['ce_minus_rrf']< -1e-12];oldgain=[x for x in paired if x['ce_minus_rrf']>1e-12]
 summary['override_tradeoff']={'old_ce_losses':len(oldloss),'improved_vs_ce':sum(x['blend_minus_ce']>1e-12 for x in oldloss),'recovered_to_rrf_or_better':sum(x['blend_minus_rrf']>=-1e-12 for x in oldloss),'old_ce_gains':len(oldgain),'old_gains_reduced_by_blend':sum(x['blend_minus_ce']< -1e-12 for x in oldgain),'new_harm_vs_rrf_on_nonlosses':sum(x['ce_minus_rrf']>=-1e-12 and x['blend_minus_rrf']< -1e-12 for x in paired)}
 tokens=read(out/'token_audit.json');status=read(out/'status.json');ledger={'candidate_pairs':sum(len(v) for v in blend.values()),'rrf_additional_ce_pairs':0,'ce_pairs':tokens['pairs'],'blend_ce_pairs':tokens['pairs'],'blend_extra_model_pairs':0,'ce_effective_tokens':tokens['retained_tokens'],'blend_effective_tokens':tokens['retained_tokens'],'training_tokens':0,'ce_inference_seconds':status['stages']['inference']['seconds'],'fusion_arithmetic_seconds':seconds,'fusion_timing_scope':'build both rank maps, sum reciprocals and sort; includes trace construction; excludes evaluation and JSON writes','shared_initial_retrieval_seconds':'not remeasured; inherited cached BM25+dense+RRF','cost_interpretation':'CE and blend share the same single model pass; pureRRF has zero incremental CE cost. Not equal total end-to-end costs. Shape check adds4 auxiliary model pairs outside scored experiment for all strategies.'}
 write(out/'blend_runs.json',blend);write(out/'fusion_trace.json',trace);write(out/'comparison_per_query.json',paired);write(out/'comparison_summary.json',summary);write(out/'budget_ledger.json',ledger)
 print(json.dumps({'comparison':summary,'budget':ledger},indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();ce_run.main(a.cache,a.out,a.smoke);blend_report(a.out)
