"""Independent stdlib reconstruction from raw pair scores; no run/common import."""
import argparse,json,math,hashlib
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def check(out):
 read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 cfg=read(ROOT/'config.json');mat=read(ROOT/'materials.json');gold=read(ROOT/'inputs/qrels.json');prior=read(ROOT/'inputs/candidates.json');pairs=read(out/'pairs.json');status=read(out/'status.json');s=read(out/'summary.json');budget=read(out/'budget_ledger.json');cells=read(out/'cells.json');raw=read(out/'raw_runs.json');pq=read(out/'per_query.json');trace=read(out/'fusion_trace.json')
 for f,h in read(ROOT/'input.lock.json')['files'].items():assert sha(ROOT/f)==h,'input identity'
 assert read(ROOT/'human_review_confirmation.json')['materials_sha256']==sha(ROOT/'materials.json')
 assert status['exit_code']==0 and s['confirmation_evaluated'] is False
 qids=[r['query_id'] for r in mat['rows']];assert qids==cfg['queries'] and [r['query_id'] for r in pq]==qids
 sort=lambda scores:sorted(scores,key=lambda d:(scores[d],d),reverse=True)
 expected={q:set(dict.fromkeys(sort(prior[q])[:2]+[next(r['doc_id'] for r in mat['rows'] if r['query_id']==q)])) if status['smoke'] else set(prior[q]) for q in qids}
 keys=[(x['condition'],x['query_id'],x['doc_id']) for x in pairs];assert len(keys)==len(set(keys))
 assert set(keys)=={(a,q,d) for a in ['original','front'] for q in qids for d in expected[q]},'pair inventory'
 by=dict(zip(keys,pairs));stats=[]
 def close(a,b):assert abs(a-b)<1e-12,(a,b)
 for q in qids:
  maps={a:{d:by[(a,q,d)]['score'] for d in expected[q]} for a in ['original','front']};pr=sort({d:prior[q][d] for d in expected[q]});rankings={};target=next(r['doc_id'] for r in mat['rows'] if r['query_id']==q)
  for a,cond in [('0','original'),('1','front')]:
   assert raw[cond][q]==maps[cond];ce=sort(maps[cond]);rankings[a+'0']=ce
   # Same float summation contract; exact rational values independently verify magnitudes, not ties.
   fused={d:1/(60+pr.index(d)+1)+1/(60+ce.index(d)+1) for d in expected[q]};rankings[a+'1']=sort(fused)
   assert cells[a+'0'][q]==maps[cond] and cells[a+'1'][q]==fused
   for d in expected[q]:close(fused[d],float(Fraction(1,60+pr.index(d)+1)+Fraction(1,60+ce.index(d)+1)))
  for d in expected[q]:
   x,y=by[('original',q,d)],by[('front',q,d)];assert x['raw_tokens']==y['raw_tokens'] and x['retained_tokens']==y['retained_tokens'];assert x['target']==y['target']==(d==target)
   if d!=target:assert x['input_sha256']==y['input_sha256'] and x['score']==y['score'],'negative invariance'
  rel={d for d,v in gold[q].items() if v>0};ideal=sum(1/math.log2(i+2) for i in range(min(10,len(rel))))
  values={k:sum(1/math.log2(i+2) for i,d in enumerate(ds[:10]) if d in rel)/ideal for k,ds in rankings.items()};x=next(x for x in pq if x['query_id']==q)
  for k,v in values.items():close(v,x['ndcg'][k]);assert x['rank'][k]==rankings[k].index(target)+1;close(x['recall50'][k],len(rel&set(rankings[k][:50]))/len(rel))
  e={'A_at_B0':values['10']-values['00'],'A_at_B1':values['11']-values['01'],'B_at_A0':values['01']-values['00'],'B_at_A1':values['11']-values['10']};e['interaction']=values['11']-values['10']-values['01']+values['00'];e['A_average']=(e['A_at_B0']+e['A_at_B1'])/2;e['B_average']=(e['B_at_A0']+e['B_at_A1'])/2
  for k,v in e.items():close(v,x['effects'][k])
  stats.append((x['group'],values,e))
 for k in cells:close(s['cell_means'][k],sum(v[k] for g,v,e in stats)/4)
 for k in stats[0][2]:close(s['mean_effects'][k],sum(e[k] for g,v,e in stats)/4)
 for g in ['long','short']:close(s['group_interaction'][g],sum(e['interaction'] for gg,v,e in stats if gg==g)/2)
 assert len(trace)==len(pairs)
 for x in trace:
  q,d,k=x['query_id'],x['doc_id'],x['cell'];pr=sort({z:prior[q][z] for z in expected[q]});cr=sort(raw['original' if k[0]=='0' else 'front'][q]);assert x['prior_rank']==pr.index(d)+1 and x['ce_rank']==cr.index(d)+1;close(x['score'],cells[k][q][d])
 totals={a:sum(x['retained_tokens'] for x in pairs if x['condition']==a) for a in ['original','front']};assert totals==s['token_totals']==budget['token_totals']
 assert budget['unique_scored_pairs']==len(pairs)==s['pairs']==status['stages']['inference']['pairs'];assert budget['cell_pair_uses']==2*len(pairs) and budget['extra_pairs_for_B']==0 and budget['training_tokens']==0
 close(budget['inference_seconds'],status['stages']['inference']['seconds']);assert s['unchanged_negative_max_score_error']==0
 report={'status':'passed','queries':4,'pairs':len(pairs),'input_identity':{f.name:sha(f) for f in [out/'pairs.json',ROOT/'input.lock.json',ROOT/'config.json',ROOT/'materials.json']},'script_sha256':sha(Path(__file__)),'independent_metrics':'stdlib DCG + exact Fraction fusion magnitude check; no inference code imported','mean_interaction':sum(e['interaction'] for g,v,e in stats)/4}
 return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=check(a.out);(a.out/'independent_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
