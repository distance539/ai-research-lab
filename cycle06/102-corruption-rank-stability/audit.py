"""Independent standard-library recomputation. Does not import run.py or analyze.py."""
import argparse,json,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate_condition(records,patches,summary):
 samples={r['id']:r for r in json.loads((ROOT/'samples.json').read_text())};parent={r['id']:r for r in json.loads((ROOT/'parent_baseline.json').read_text())}
 assert len({r['id'] for r in records})==len(records)
 ids={r['id']:r for r in records};seen=set();gains={};gap=0.;premax=0.;strongnegative=0
 def close(a,b,eps=1e-9):assert abs(a-b)<=eps,(a,b)
 for r in records:
  assert r['id'] in samples
  for k,v in samples[r['id']].items():assert r[k]==v,k
  x,y=r['clean_ids'],r['corrupt_ids'];assert len(x)==len(y)==r['end_position']+1
  changed=[i for i,(a,b) in enumerate(zip(x,y)) if a!=b];assert changed==r['changed_positions']
  expected=[r['s2_position']] if r['condition']=='s2_flip' else [i for i,q in enumerate(x) if q in [r['io_id'],r['s_id']]]
  assert changed==expected
  io,s=r['io_id'],r['s_id'];assert x.count(io)==1 and x.count(s)==2 and y.count(io)==2 and y.count(s)==1
  assert y==([io if i==r['s2_position'] else q for i,q in enumerate(x)] if r['condition']=='s2_flip' else [s if q==io else io if q==s else q for q in x])
  assert x[r['s2_position']]==s and y[r['s2_position']]==io and x[r['io_position']]==io
  assert r['z_shape']==[1,len(x),12,64]
  for k in ['clean','corrupt','full','self','batch_clean','batch_corrupt']:
   q=r[k];close(q['io_logit']-q['s_logit'],q['diff'],1e-6);assert 0<=q['p_io']<=1
  assert r['clean']==r['full'] and r['corrupt']==r['self']
  for k in ['full_max_error','self_max_error']:assert r[k]==0
  for k in ['official_readout_max_error','projected_delta_max_error','batch_max_error']:assert 0<=r[k]<0.001
  close(r['clean']['diff'],parent[r['id']]['logit_diff'],0.001)
  gap+=r['batch_clean']['diff']-r['batch_corrupt']['diff']
 for p in patches:
  r=ids[p['id']];key=(p['id'],p['layer'],p['position'],p['head']);assert key not in seen;seen.add(key)
  assert 0<=p['layer']<12 and 0<=p['head']<12 and 0<=p['position']<=r['end_position']
  assert all(math.isfinite(p[k]) for k in ['io_logit','s_logit','diff','gain','p_io'])
  close(p['io_logit']-p['s_logit'],p['diff'],1e-6);close(p['diff']-r['batch_corrupt']['diff'],p['gain']);assert 0<=p['p_io']<=1
  if p['position']==r['end_position']:gains.setdefault((p['layer'],p['head']),[]).append(p['gain'])
  if p['position']<min(r['changed_positions']):premax=max(premax,abs(p['gain']))
  strongnegative+=p['gain'] < -0.001
 expected=sum(len(r['clean_ids'])*144 for r in records);assert len(seen)==expected
 assert premax<0.001
 n=len(records);gap/=n;assert gap>0.1;close(gap,summary['mean_gap']);assert summary['n']==n and summary['patch_count']==expected
 for key,rs in gains.items():assert len(rs)==n
 ranking=sorted(gains,key=lambda k:(-math.fsum(gains[k])/n,*k))
 assert ranking==[(p['layer'],p['head']) for p in summary['ranking']]
 for p in summary['ranking']:
  g=math.fsum(gains[p['layer'],p['head']])/n;close(g,p['mean_gain']);close(g/gap,p['recovery_ratio'])
 for k,src in [('mean_clean_diff','batch_clean'),('mean_corrupt_diff','batch_corrupt')]:close(math.fsum(r[src]['diff'] for r in records)/n,summary[k])
 return {'passed':True,'rows':n,'grid_cells':expected,'mean_gap':gap,'top_end_head':list(ranking[0]),'strong_negative_gains_below_minus_0_001':strongnegative,'pre_corruption_max_gain':premax,'checks':['sample identities','explicit counterfactual token substitution','fixed candidate identities','full-state exact','selfpatch exact','official readout','projected delta equivalence','parent baseline agreement','complete unique grid','raw arithmetic','all END rankings and ratios','causal negative controls']}
def validate(records,patches,summary):
 import collections
 conditions=['s2_flip','global_swap'];assert set(r['condition'] for r in records)==set(conditions)
 assert set(p['condition'] for p in patches)==set(conditions)
 result={}
 ranks={};raw={};end={}
 for condition in conditions:
  rs=[r for r in records if r['condition']==condition];ps=[p for p in patches if p['condition']==condition]
  result[condition]=validate_condition(rs,ps,summary['conditions'][condition]);byid={r['id']:r for r in rs}
  raw[condition]={}
  for p in ps:
   if p['position']==byid[p['id']]['end_position']:
    k=(p['layer'],p['head']);raw[condition].setdefault(k,[]).append(p['gain']);end[condition,p['id'],k]=p['gain']
  means={k:math.fsum(v)/len(v) for k,v in raw[condition].items()};ordered=sorted(means,key=lambda k:(-means[k],*k));ranks[condition]={k:i+1 for i,k in enumerate(ordered)}
  for x in summary['conditions'][condition]['ranking']:assert x['rank']==ranks[condition][x['layer'],x['head']]
  clean=collections.Counter(tuple(r['clean_ids']) for r in rs);bad=collections.Counter(tuple(r['corrupt_ids']) for r in rs)
  assert summary['comparison']['corrupt_multisets_equal_clean'][condition]==(clean==bad)
  if len(rs)==16:assert clean==bad
  assert abs(summary['conditions'][condition]['top10_boundary_margin']-(means[ordered[9]]-means[ordered[10]]))<1e-9
 a,b=[ranks[k] for k in conditions];keys=sorted(a);center=72.5
 # Independent Pearson calculation on ordinal ranks (rather than squared rank-distance formula).
 rho=math.fsum((a[k]-center)*(b[k]-center) for k in keys)/math.sqrt(math.fsum((a[k]-center)**2 for k in keys)*math.fsum((b[k]-center)**2 for k in keys))
 comp=summary['comparison'];assert abs(rho-comp['ordinal_rank_correlation'])<1e-12
 ta={k for k in keys if a[k]<=10};tb={k for k in keys if b[k]<=10}
 assert comp['top10_overlap']==len(ta&tb)/10 and comp['top10_jaccard']==len(ta&tb)/len(ta|tb)
 assert comp['top10_intersection']==[list(k) for k in sorted(ta&tb)]
 ra={r['id']:r for r in records if r['condition']=='s2_flip'};rb={r['id']:r for r in records if r['condition']=='global_swap'};assert ra.keys()==rb.keys()
 for i in ra:assert ra[i]['clean']==rb[i]['clean'] and ra[i]['clean_ids']==rb[i]['clean_ids']
 diffs=[]
 for x in comp['focal_L8H10']:
  i=x['id'];ga=end['s2_flip',i,(8,10)];gb=end['global_swap',i,(8,10)]
  assert abs(x['gain_a']-ga)<1e-9 and abs(x['gain_b']-gb)<1e-9 and abs(x['difference']-(gb-ga))<1e-9;diffs.append(gb-ga)
 assert len(diffs)==len(ra) and abs(math.fsum(diffs)/len(diffs)-comp['focal_mean_paired_gain_change'])<1e-9
 for x in comp['rank_movement']:
  k=x['layer'],x['head'];assert x['rank_a']==a[k] and x['rank_b']==b[k] and x['rank_change']==b[k]-a[k]
 if len(ra)==16:
  parent=json.loads((ROOT/'parent_summary.json').read_text())
  for x,y in zip(summary['conditions']['s2_flip']['ranking'],parent['end_head_ranking']):
   assert (x['layer'],x['head'])==(y['layer'],y['head']) and abs(x['mean_gain']-y['mean_gain'])<0.001
 return dict(passed=True,conditions=result,ordinal_rank_correlation=rho,top10_overlap=comp['top10_overlap'],parent101_reproduced=len(ra)==16)

def main():
 a=argparse.ArgumentParser();a.add_argument('--results',default=str(ROOT/'results'));a.add_argument('--out');args=a.parse_args();p=Path(args.results)
 result=validate(*[json.loads((p/f).read_text()) for f in ['records.json','patches.json','summary.json']]);result['inputs']={f:digest(p/f) for f in ['records.json','patches.json','summary.json']};result['script_sha256']=digest(__file__)
 if args.out:Path(args.out).write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
