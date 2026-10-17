"""Independent standard-library recomputation. Does not import run.py or analyze.py."""
import argparse,json,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate(records,patches,summary):
 samples={r['id']:r for r in json.loads((ROOT/'samples.json').read_text())};parent={r['id']:r for r in json.loads((ROOT/'parent_baseline.json').read_text())}
 assert len({r['id'] for r in records})==len(records)
 ids={r['id']:r for r in records};seen=set();gains={};gap=0.;premax=0.;strongnegative=0
 def close(a,b,eps=1e-9):assert abs(a-b)<=eps,(a,b)
 for r in records:
  assert r['id'] in samples
  for k,v in samples[r['id']].items():assert r[k]==v,k
  x,y=r['clean_ids'],r['corrupt_ids'];assert len(x)==len(y)==r['end_position']+1
  assert [i for i,(a,b) in enumerate(zip(x,y)) if a!=b]==[r['s2_position']]
  io,s=r['io_id'],r['s_id'];assert x.count(io)==1 and x.count(s)==2 and y.count(io)==2 and y.count(s)==1
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
  if p['position']<r['s2_position']:premax=max(premax,abs(p['gain']))
  strongnegative+=p['gain'] < -0.001
 expected=sum(len(r['clean_ids'])*144 for r in records);assert len(seen)==expected
 assert premax<0.001
 n=len(records);gap/=n;assert gap>0.1;close(gap,summary['mean_gap']);assert summary['n']==n and summary['patch_count']==expected
 for key,rs in gains.items():assert len(rs)==n
 ranking=sorted(gains,key=lambda k:(-math.fsum(gains[k])/n,*k))
 assert ranking==[(p['layer'],p['head']) for p in summary['end_head_ranking']]
 for p in summary['end_head_ranking']:
  g=math.fsum(gains[p['layer'],p['head']])/n;close(g,p['mean_gain']);close(g/gap,p['recovery_ratio'])
 for k,src in [('mean_clean_diff','batch_clean'),('mean_corrupt_diff','batch_corrupt')]:close(math.fsum(r[src]['diff'] for r in records)/n,summary[k])
 return {'passed':True,'rows':n,'grid_cells':expected,'mean_gap':gap,'top_end_head':list(ranking[0]),'strong_negative_gains_below_minus_0_001':strongnegative,'pre_corruption_max_gain':premax,'checks':['sample identities','one token corruption','fixed candidate identities','full-state exact','selfpatch exact','official readout','projected delta equivalence','parent baseline agreement','complete unique grid','raw arithmetic','all END rankings and ratios','causal negative controls']}
def main():
 a=argparse.ArgumentParser();a.add_argument('--results',default=str(ROOT/'results'));a.add_argument('--out');args=a.parse_args();p=Path(args.results)
 result=validate(*[json.loads((p/f).read_text()) for f in ['records.json','patches.json','summary.json']]);result['inputs']={f:digest(p/f) for f in ['records.json','patches.json','summary.json']};result['script_sha256']=digest(__file__)
 if args.out:Path(args.out).write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
