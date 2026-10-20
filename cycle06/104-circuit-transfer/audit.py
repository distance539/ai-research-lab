"""Independent standard-library arithmetic/identity audit; no model/analyze imports."""
import json,math,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parent
K=['clean','zero','corrupt','restored']
def mean(a):return math.fsum(a)/len(a)
def close(a,b):assert abs(a-b)<1e-9,(a,b)
def audit(p):
 p=Path(p);rows=json.loads((p/'records.json').read_text());s=json.loads((p/'summary.json').read_text());identity=json.loads((p/'identity.json').read_text());cfg=json.loads((ROOT/'config.json').read_text());samples=json.loads((ROOT/'samples.json').read_text())
 sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
 assert identity['plan_sha256']==sha(ROOT/'plan.json') and identity['resources_sha256']==sha(ROOT/'resources.lock.json')
 expected=cfg['smoke_ids'] if identity['mode']=='smoke' else list(range(256));assert [r['id'] for r in rows]==expected==identity['sample_ids']
 assert cfg['heads']=={'A':[8,10],'B':[8,6],'C':[9,9]}
 for r in rows:
  for k,v in samples[r['id']].items():assert r[k]==v,(r['id'],k)
  ids=r['ids'];ci=r['corrupt_ids'];io=r['io_id'];su=r['s_id'];assert io!=su and ids.count(io)==1 and ids.count(su)==2
  assert len(ids)==len(ci) and r['end_position']==len(ids)-1
  assert r['s2_position']==max(j for j,q in enumerate(ids) if q==su)
  assert [j for j,(a,b) in enumerate(zip(ids,ci)) if a!=b]==[r['s2_position']];assert ci[r['s2_position']]==io
  assert r['z_shape']==[1,len(ids),12,64] and r['logits_shape']==[1,len(ids),50257]
  assert set(r['clean_z'])=={'A','B','C'} and all(len(x)==64 and all(math.isfinite(v) for v in x) for x in r['clean_z'].values())
  for k in K:
   v=r[k];assert all(math.isfinite(v[x]) for x in ['io_logit','s_logit','diff','p_io']);close(v['diff'],v['io_logit']-v['s_logit']);assert 0<=v['p_io']<=1 and 0<=v['top1_id']<50257
  assert all(r['controls'][k]==0 for k in ['self_error','corrupt_self_error','empty_error'])
  assert all(0<=r['controls'][k]<.001 for k in ['projected_zero_error','projected_restore_error'])
 groups={}
 for g in dict.fromkeys(r['group'] for r in rows):
  rr=[r for r in rows if r['group']==g];ss=s['groups'][g];assert len(rr)==ss['n'];assert len({r['paired_key'] for r in rr})==len(rr)
  means={k:mean([r[k]['io_logit']-r[k]['s_logit'] for r in rr]) for k in K}
  for k in K:
   close(ss['mean_diff'][k],means[k]);assert ss['positive_counts'][k]==sum(r[k]['diff']>0 for r in rr);assert ss['top1_counts'][k]==sum(r[k]['top1_id']==r['io_id'] for r in rr)
  drop=mean([r['clean']['diff']-r['zero']['diff'] for r in rr]);gain=mean([r['restored']['diff']-r['corrupt']['diff'] for r in rr]);gap=means['clean']-means['corrupt']
  for k,v in [('drop',drop),('gain',gain),('gap',gap)]:close(ss[k],v)
  if abs(gap)>1e-6:close(ss['recovery'],gain/gap)
  else:assert ss['recovery'] is None
  assert ss['drop_nonpositive_ids']==[r['id'] for r in rr if r['clean']['diff']<=r['zero']['diff']]
  assert ss['gain_nonpositive_ids']==[r['id'] for r in rr if r['restored']['diff']<=r['corrupt']['diff']]
  groups[g]=dict(drop=drop,gain=gain,clean=means['clean'])
 if identity['mode']=='formal':
  assert set(groups)==set(cfg['groups']) and all(s['groups'][g]['n']==64 for g in groups)
  assert all({r['paired_key'] for r in rows if r['group']==g}=={r['paired_key'] for r in rows if r['group']=='old_old'} for g in groups)
  for g in ['new_old','old_new','new_new']:
   for k in ['drop','gain']:close(s['contrasts'][g][k],groups[g][k]-groups['old_old'][k])
   close(s['contrasts'][g]['clean_diff'],groups[g]['clean']-groups['old_old']['clean'])
  for k in ['drop','gain']:close(s['contrasts']['factor_interaction'][k],groups['new_new'][k]-groups['new_old'][k]-groups['old_new'][k]+groups['old_old'][k])
 return dict(status='passed',rows=len(rows),conditions=4*len(rows),records_sha256=sha(p/'records.json'),summary_sha256=sha(p/'summary.json'),plan_sha256=identity['plan_sha256'],groups=groups,limits='Full-vocabulary argmax requires model replay; no human semantic annotation claimed.')
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('directory');a=ap.parse_args();print(json.dumps(audit(a.directory),indent=2))
