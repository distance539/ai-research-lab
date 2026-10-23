"""Independent stdlib recomputation: does not import run/common/production metrics."""
import argparse,json,math,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(out):
 cfg=load(R/'config.json');mat=load(R/'materials.json');runs=load(out/'runs.json');pairs=load(out/'pairs.json');pq=load(out/'per_query.json');summary=load(out/'summary.json');qrels=load(R/'inputs/qrels.json');candidates=load(R/'inputs/candidates.json');status=load(out/'status.json')
 assert status['exit_code']==0 and summary['confirmation_evaluated'] is False
 assert summary['semantic_review'].startswith('pending'), 'No fabricated approval'
 assert set(runs)==set(cfg['conditions'])
 ident=load(out/'identity.json');assert ident['config']==cfg and ident['input_lock']==load(R/'input.lock.json')
 for f,h in load(R/'input.lock.json')['files'].items():assert digest(R/f)==h
 assert len(pairs)==sum(len(r) for run in runs.values() for r in run.values())
 keys={(x['condition'],x['query_id'],x['doc_id']) for x in pairs};assert len(keys)==len(pairs)
 pd={ (x['condition'],x['query_id'],x['doc_id']):x for x in pairs }
 recalculated=[]
 for item in mat['rows']:
  q=item['query_id'];target=item['doc_id'];gold={d for d,v in qrels[q].items() if v>0};assert gold=={target}
  expected=set(candidates[q]) if not status['smoke'] else set(sorted(candidates[q],key=lambda d:(candidates[q][d],d),reverse=True)[:2]+[target])
  vals={};rankvals={}
  for cond in cfg['conditions']:
   sc=runs[cond][q];assert set(sc)==expected
   ranked=sorted(sc,key=lambda d:(sc[d],d),reverse=True);val=sum((1/math.log2(i+2)) for i,d in enumerate(ranked[:10]) if d in gold)/sum(1/math.log2(i+2) for i in range(min(10,len(gold))))
   vals[cond]=val;rankvals[cond]=ranked.index(target)+1
   for d in sc:
    x=pd[cond,q,d];assert math.isfinite(sc[d]) and x['score']==sc[d]
    assert x['retained_tokens']==min(x['raw_tokens'],512)
    base=pd['original',q,d]
    if d!=target or cond=='identity':assert x['input_sha256']==base['input_sha256'] and sc[d]==runs['original'][q][d]
   # All full sentences retained as a permutation. This is not semantic validation.
   assert sorted(item['orders'][cond])==list(range(len(item['document']['abstract'])))
   v=item['visibility'][cond];assert pd[cond,q,target]['input_sha256']==v['input_ids_sha256']
   assert pd[cond,q,target]['raw_tokens']==v['raw_tokens']
  assert vals['original']==vals['identity'];stored=next(x for x in pq if x['query_id']==q)
  assert stored['ndcg']==vals and stored['rank']==rankvals
  assert stored['target_score']=={c:runs[c][q][target] for c in runs}
  delta=vals['front']-vals['tail'];assert abs(stored['front_minus_tail']-delta)<1e-12
  recalculated.append((item['group'],delta))
  if item['group']=='long':assert item['visibility']['front']['evidence_fully_visible'] and not item['visibility']['tail']['evidence_fully_visible']
  else:assert all(v['evidence_fully_visible'] and v['raw_tokens']<=512 for v in item['visibility'].values())
 means={g:sum(v for k,v in recalculated if k==g)/2 for g in ['long','short']}
 assert means==summary['group_delta'];assert abs(means['long']-means['short']-summary['difference_of_group_deltas'])<1e-12
 totals={c:sum(x['retained_tokens'] for x in pairs if x['condition']==c) for c in cfg['conditions']};assert totals==summary['token_totals'] and len(set(totals.values()))==1
 assert summary['pairs']==len(pairs) and summary['identity_max_error']==0
 return {'status':'passed','scope':'all scores, input identity, permutations, candidate sets, ranks, nDCG, paired differences, token totals; no human semantics','pairs':len(pairs),'group_delta':means,'input_sha256':{f:digest(out/f) for f in ['runs.json','pairs.json','per_query.json','summary.json','identity.json']},'script_sha256':digest(Path(__file__))}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=check(a.out);(a.out/'audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
