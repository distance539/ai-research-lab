"""Independent saved-output audit, standard library only."""
import argparse,csv,math
from pathlib import Path
from common import ROOT,read,write,metric,order

def main(out):
 r=read(out/'runs.json');p=read(out/'per_query.json');s=read(out/'summary.json');g=read(ROOT/'inputs/qrels.json');rows=list(csv.DictReader((out/'pairs.csv').open()));maxerr=0.
 for n,run in r.items():
  for q,ss in run.items():
   assert set(ss)==set(r['rrf'][q]);assert all(math.isfinite(x) for x in ss.values())
   for m,v in metric(g[q],order(ss)).items():maxerr=max(maxerr,abs(v-p[n][q][m]))
   assert abs(p[n][q]['recall_50']-p['rrf'][q]['recall_50'])<1e-12
  for m,v in s['metrics'][n].items():assert abs(v-sum(metric(g[q],order(run[q]))[m] for q in run)/len(run))<1e-12
 for n,stats in s['tokens'].items():
  rr=[x for x in rows if x['variant']==n];assert len(rr)==len({(x['query_id'],x['doc_id']) for x in rr})==s['pairs_per_variant'];limit=int(n.split('_')[-1])
  for x in rr:
   q,d=x['query_id'],x['doc_id'];assert float(x['logit'])==r[n][q][d];assert int(x['rank'])==order(r[n][q]).index(d)+1;assert int(x['relevant'])==int(g[q].get(d,0)>0)
   assert int(x['retained_tokens'])==min(int(x['raw_tokens']),limit);assert int(x['truncated'])==int(int(x['raw_tokens'])>limit)
   assert int(x['query_retained'])+int(x['doc_retained'])+int(x['special_tokens'])==int(x['retained_tokens']);assert int(x['title_retained'])+int(x['abstract_retained'])==int(x['doc_retained'])
   if n.startswith('abstract'):assert int(x['title_retained'])==0
  for k,v in stats.items():
   if k=='relevant_pairs':actual=sum(int(x['relevant']) for x in rr)
   elif k=='relevant_truncated':actual=sum(int(x['relevant'])*int(x['truncated']) for x in rr)
   elif k=='relevant_unique_docs_truncated':actual=len({x['doc_id'] for x in rr if int(x['relevant']) and int(x['truncated'])})
   else:actual=sum(int(x[k]) for x in rr)
   assert actual==v,(n,k)
 contrasts=read(out/'contrasts.json')
 for name,vals in contrasts.items():
  for q,v in vals.items():
   if name=='interaction':expected=(p['title_512'][q]['ndcg_cut_10']-p['abstract_512'][q]['ndcg_cut_10'])-(p['title_256'][q]['ndcg_cut_10']-p['abstract_256'][q]['ndcg_cut_10'])
   else:a,b=name.split('_minus_');expected=p[a][q]['ndcg_cut_10']-p[b][q]['ndcg_cut_10']
   assert abs(v-expected)<1e-12
  assert abs(s['effects'][name]['mean']-sum(vals.values())/len(vals))<1e-12
  assert s['effects'][name]['win_tie_loss']==[sum(v>1e-12 for v in vals.values()),sum(abs(v)<=1e-12 for v in vals.values()),sum(v< -1e-12 for v in vals.values())]
 assert maxerr<1e-12
 result={'passed':True,'queries':len(r['rrf']),'pair_rows':len(rows),'max_metric_error':maxerr,'candidate_identity':True,'tokens_and_ranks':True,'effects_recomputed':True};write(out/'audit.json',result);print(result)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);main(a.parse_args().out)
