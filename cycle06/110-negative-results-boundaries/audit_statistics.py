"""Independent stdlib audit: graph BFS and resampled raw scores, no stats helper imports."""
import argparse,csv,gzip,json,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
load=lambda p:json.loads(p.read_text())
def check(out,save=True):
 cfg=load(ROOT/'config.json')['statistics'];gold=load(ROOT/'inputs/qrels.json');rows=load(out/'comparison_per_query.json');byq={r['query_id']:r for r in rows};qids=sorted(byq);n=len(qids);report=load(out/'uncertainty.json');saved=load(out/'components.json');idx=json.loads(gzip.decompress((out/'bootstrap_indices.json.gz').read_bytes()));rep=json.loads(gzip.decompress((out/'bootstrap_replicates.json.gz').read_bytes()))
 assert report['seed']==cfg['seed']==110 and report['replicates']==cfg['resamples']==10000 and report['confirmation_evaluated'] is False
 rel={q:{d for d,v in gold[q].items() if v>0} for q in qids};unseen=set(qids);groups=[]
 while unseen:
  todo=[min(unseen)];component=set(todo);unseen-=component
  while todo:
   q=todo.pop();neighbors={r for r in unseen if rel[q]&rel[r]};component|=neighbors;unseen-=neighbors;todo+=sorted(neighbors)
  groups.append(sorted(component))
 groups.sort(key=lambda x:x[0]);assert groups==saved['groups'] and len(groups)==saved['count']==report['component_count'];assert saved['sizes']==sorted(map(len,groups),reverse=True)
 close=lambda a,b: math.isfinite(a) and math.isfinite(b) and abs(a-b)<1e-12
 def quantile(xs,p):
  s=sorted(xs);v=(len(s)-1)*p;i=int(v);return s[i]+(s[min(i+1,len(s)-1)]-s[i])*(v-i)
 for mode,units in [('component',groups),('query',[[q] for q in qids])]:
  assert len(idx[mode])==cfg['resamples'] and all(len(draw)==len(units) and all(type(i)==int and 0<=i<len(units) for i in draw) for draw in idx[mode])
  for key in ['blend_minus_ce','blend_minus_rrf','ce_minus_rrf']:
   assert len(rep[mode][key])==cfg['resamples'];sums=[sum(byq[q][key] for q in unit) for unit in units];sizes=list(map(len,units))
   actual=[sum(sums[i] for i in draw)/sum(sizes[i] for i in draw) for draw in idx[mode]]
   assert all(close(a,b) for a,b in zip(actual,rep[mode][key]))
   assert all(close(a,b) for a,b in zip([quantile(actual,.025),quantile(actual,.975)],report['comparisons'][key][mode+'_percentile_95']))
 for key,rec in report['comparisons'].items():
  total=sum(byq[q][key] for q in qids);assert close(rec['mean'],total/n)
  loo=[(total-sum(byq[q][key] for q in g))/(n-len(g)) for g in groups]
  assert all(close(a,b) for a,b in zip([min(loo),max(loo)],rec['leave_one_component_out_range']))
 for f,h in report['input_sha256'].items():assert hashlib.sha256((out/f).read_bytes()).hexdigest()==h
 failures=load(out/'failure_cases.json')['rows'];assert failures==sorted(rows,key=lambda x:(x['blend_minus_ce'],x['query_id']))[:3]
 # Pairwise partition invariant: every shared-positive-document pair must share a group.
 assert sum(map(len,groups))==n and len({q for g in groups for q in g})==n
 result={'passed':True,'independent_graph':'BFS shared positive qrels','resampled_means_checked':60000,'interval_endpoints_checked':12,'input_sha256':report['input_sha256'],'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 if save:(out/'statistics_audit.json').write_text(json.dumps(result,indent=2)+'\n')
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);print(check(p.parse_args().out))
