"""Predeclared paired corruption comparison; no inference or sample selection."""
import json,argparse,statistics,collections
from pathlib import Path

def analyze(records,patches):
 mean=statistics.mean;conditions={}
 for condition in ['s2_flip','global_swap']:
  rs=[r for r in records if r['condition']==condition];ps=[p for p in patches if p['condition']==condition];byid={r['id']:r for r in rs}
  gap=mean(r['batch_clean']['diff']-r['batch_corrupt']['diff'] for r in rs);assert gap>0.1
  ends=[p for p in ps if p['position']==byid[p['id']]['end_position']];ranking=[]
  for l in range(12):
   for h in range(12):
    pp=[p for p in ends if p['layer']==l and p['head']==h];g=mean(p['gain'] for p in pp)
    ranking.append(dict(layer=l,head=h,mean_gain=g,recovery_ratio=g/gap,mean_diff=mean(p['diff'] for p in pp),positive_gains=sum(p['gain']>0 for p in pp),n=len(pp)))
  ranking.sort(key=lambda p:(-p['mean_gain'],p['layer'],p['head']))
  for i,p in enumerate(ranking):p['rank']=i+1
  conditions[condition]=dict(n=len(rs),patch_count=len(ps),mean_clean_diff=mean(r['batch_clean']['diff'] for r in rs),mean_corrupt_diff=mean(r['batch_corrupt']['diff'] for r in rs),mean_gap=gap,clean_positive=sum(r['clean']['diff']>0 for r in rs),corrupt_positive_original=sum(r['corrupt']['diff']>0 for r in rs),ranking=ranking,negative_gains=sum(p['gain']<0 for p in ps),strong_negative_gains=sum(p['gain'] < -0.001 for p in ps),max_pre_change_gain=max(abs(p['gain']) for p in ps if p['position']<min(byid[p['id']]['changed_positions'])),top10_boundary_margin=ranking[9]['mean_gain']-ranking[10]['mean_gain'])
 a,b=[conditions[k]['ranking'] for k in conditions];da={(r['layer'],r['head']):r for r in a};db={(r['layer'],r['head']):r for r in b};keys=list(da);N=len(keys)
 rho=1-6*sum((da[k]['rank']-db[k]['rank'])**2 for k in keys)/(N*(N*N-1))
 ta=set(list(da)[:10]);tb=set(list(db)[:10]);movement=[dict(layer=k[0],head=k[1],rank_a=da[k]['rank'],rank_b=db[k]['rank'],rank_change=db[k]['rank']-da[k]['rank'],gain_a=da[k]['mean_gain'],gain_b=db[k]['mean_gain']) for k in keys];movement.sort(key=lambda p:(-abs(p['rank_change']),p['layer'],p['head']))
 focal=[]
 for r in records:
  if r['condition']!='s2_flip':continue
  pp={p['condition']:p for p in patches if p['id']==r['id'] and p['position']==r['end_position'] and p['layer']==8 and p['head']==10}
  focal.append(dict(id=r['id'],template_index=r['template_index'],order=r['order'],gain_a=pp['s2_flip']['gain'],gain_b=pp['global_swap']['gain'],difference=pp['global_swap']['gain']-pp['s2_flip']['gain']))
 clean=[tuple(r['clean_ids']) for r in records if r['condition']=='s2_flip'];multiset={k:collections.Counter(clean)==collections.Counter(tuple(r['corrupt_ids']) for r in records if r['condition']==k) for k in conditions}
 return dict(conditions=conditions,comparison=dict(ordinal_rank_correlation=rho,top_k=10,top10_intersection=[list(k) for k in sorted(ta&tb)],top10_overlap=len(ta&tb)/10,top10_jaccard=len(ta&tb)/len(ta|tb),rank_movement=movement,focal_L8H10=focal,focal_mean_paired_gain_change=mean(r['difference'] for r in focal),corrupt_multisets_equal_clean=multiset,exact_gain_ties={k:144-len(set(r['mean_gain'] for r in v['ranking'])) for k,v in conditions.items()},near_adjacent_rank_gaps_below_0_001={k:sum(abs(u['mean_gain']-v['mean_gain'])<.001 for u,v in zip(z['ranking'],z['ranking'][1:])) for k,z in conditions.items()}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--results',default='results');a=p.parse_args();d=Path(a.results);print(json.dumps(analyze(json.loads((d/'records.json').read_text()),json.loads((d/'patches.json').read_text())),indent=2))
