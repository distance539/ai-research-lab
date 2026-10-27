"""Paired component bootstrap; explicit indices and no hidden selection."""
import argparse,gzip,json,time
from pathlib import Path
import numpy as np
from common import ROOT,read,write,sha
KEYS=['blend_minus_ce','blend_minus_rrf','ce_minus_rrf']
def groups_from_gold(qids,gold):
 parent={q:q for q in qids}
 def find(q):
  while parent[q]!=q:parent[q]=parent[parent[q]];q=parent[q]
  return q
 bydoc={}
 for q in sorted(qids):
  for d,v in gold[q].items():
   if v>0:
    if d in bydoc:parent[find(q)]=find(bydoc[d])
    else:bydoc[d]=q
 groups={}
 for q in sorted(qids):groups.setdefault(find(q),[]).append(q)
 return sorted(groups.values(),key=lambda x:x[0])
def gzwrite(p,data):p.write_bytes(gzip.compress(json.dumps(data,separators=(',',':'),allow_nan=False).encode(),mtime=0))
def main(out):
 started=time.perf_counter();cfg=read(ROOT/'config.json')['statistics'];rows=read(out/'comparison_per_query.json');qids=sorted(x['query_id'] for x in rows);byq={r['query_id']:r for r in rows};gold=read(ROOT/'inputs/qrels.json');groups=groups_from_gold(qids,gold)
 if not read(out/'status.json')['smoke']:assert groups==read(ROOT/'inputs/confirmation_groups.json')['groups']
 write(out/'components.json',{'rule':cfg['primary_unit'],'groups':groups,'count':len(groups),'sizes':sorted(map(len,groups),reverse=True),'queries':len(qids)})
 rng=np.random.default_rng(cfg['seed']);result={'scope':'frozen confirmation resampling' if not read(out/'status.json')['smoke'] else 'development preflight','seed':cfg['seed'],'replicates':cfg['resamples'],'component_count':len(groups),'comparisons':{},'confirmation_evaluated':not read(out/'status.json')['smoke']};distributions={};draws={}
 for mode,units in [('component',groups),('query',[[q] for q in qids])]:
  ix=rng.integers(len(units),size=(cfg['resamples'],len(units)));draws[mode]=ix.tolist();sizes=np.array(list(map(len,units)));den=sizes[ix].sum(axis=1)
  distributions[mode]={}
  for key in KEYS:
   sums=np.array([sum(byq[q][key] for q in g) for g in units]);dist=sums[ix].sum(axis=1)/den;distributions[mode][key]=dist.tolist()
   rec=result['comparisons'].setdefault(key,{'mean':sum(byq[q][key] for q in qids)/len(qids)})
   rec[mode+'_percentile_95']=np.quantile(dist,[.025,.975],method='linear').tolist()
 for key in KEYS:
  total=sum(byq[q][key] for q in qids);loo=[(total-sum(byq[q][key] for q in g))/(len(qids)-len(g)) for g in groups] if len(groups)>1 else []
  result['comparisons'][key]['leave_one_component_out_range']=[min(loo),max(loo)] if loo else None
 gzwrite(out/'bootstrap_indices.json.gz',draws);gzwrite(out/'bootstrap_replicates.json.gz',distributions)
 failed=sorted(rows,key=lambda r:(r['blend_minus_ce'],r['query_id']))[:3];write(out/'failure_cases.json',{'rule':'three lowest blend-minusCE; descriptive ranking predicate, no new semantic labels','rows':failed})
 result['seconds']=time.perf_counter()-started;result['input_sha256']={f:sha(out/f) for f in ['comparison_per_query.json','components.json','bootstrap_indices.json.gz','bootstrap_replicates.json.gz']};write(out/'uncertainty.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
