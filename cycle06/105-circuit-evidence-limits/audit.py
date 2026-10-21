"""Independent stdlib audit; deliberately does not import analysis or model code."""
import argparse,json,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
KLIST=['','A','B','C','AB','AC','BC','ABC']
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def avg(v):return math.fsum(v)/len(v)
def near(x,y):assert math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-10),(x,y)
def audit(p):
 p=Path(p);rows=json.loads((p/'records.json').read_text());summary=json.loads((p/'summary.json').read_text());identity=json.loads((p/'identity.json').read_text());plan=json.loads((ROOT/'plan.json').read_text());cfg=json.loads((ROOT/'config.json').read_text())
 assert identity['plan_sha256']==digest(ROOT/'plan.json') and identity['resources_sha256']==digest(ROOT/'resources.lock.json')
 for f,h in plan['files'].items():assert digest(ROOT/f)==h
 expected=json.loads((ROOT/'samples.json').read_text());expected=[r for r in expected if r['id'] in cfg['smoke_ids']] if identity['mode']=='smoke' else expected
 assert [r['id'] for r in rows]==[r['id'] for r in expected]==identity['sample_ids']
 for r,e in zip(rows,expected):
  for k,v in e.items():assert r[k]==v
  assert len(r['ids'])==r['end_position']+1 and r['ids'].count(r['io_id'])==1 and r['ids'].count(r['s_id'])==2
  assert r['z_shape']==[1,len(r['ids']),12,64] and r['logits_shape']==[1,len(r['ids']),50257]
  assert r['controls']['all_head_self_error']==r['controls']['empty_error']==0
  assert all(0<=e<.001 for e in r['controls']['projected_errors'].values())
  assert set(r['conditions'])=={'zero','mean_abc'}
  vals=[r['clean']]
  for mode in ['zero','mean_abc']:
   assert set(r['conditions'][mode])=={'full','retained'}
   for system in ['full','retained']:
    assert set(r['conditions'][mode][system])==set(KLIST)
    vals+=list(r['conditions'][mode][system].values())
   assert r['conditions'][mode]['full']['']==r['clean']
  for v in vals:
   assert all(math.isfinite(v[k]) for k in ['io_logit','s_logit','diff','p_io']);near(v['diff'],v['io_logit']-v['s_logit'])
   assert 0<=v['p_io']<=1 and isinstance(v['top1_id'],int) and 0<=v['top1_id']<50257
   if v['diff']<0:assert v['top1_id']!=r['io_id']
 groups=list(dict.fromkeys(r['group'] for r in rows));assert list(summary)==groups
 for g in groups:
  rr=[r for r in rows if r['group']==g];ss=summary[g];assert ss['n']==len(rr);near(ss['clean_mean'],avg([r['clean']['diff'] for r in rr]));assert ss['clean_top1']==sum(r['clean']['top1_id']==r['io_id'] for r in rr)
  for mode in ['zero','mean_abc']:
   gap={}
   for k in KLIST:
    s=ss['modes'][mode]['subsets'][k];fv=[r['conditions'][mode]['full'][k] for r in rr];cv=[r['conditions'][mode]['retained'][k] for r in rr];diff=[a['io_logit']-a['s_logit']-b['io_logit']+b['s_logit'] for a,b in zip(fv,cv)]
    for key,value in [('full_mean',avg([v['io_logit']-v['s_logit'] for v in fv])),('retained_mean',avg([v['io_logit']-v['s_logit'] for v in cv])),('signed_mean_gap',avg(diff)),('absolute_mean_gap',abs(avg(diff))),('row_mae',avg([abs(v) for v in diff]))]:near(s[key],value)
    for key,value in [('full_top1',sum(v['top1_id']==r['io_id'] for v,r in zip(fv,rr))),('retained_top1',sum(v['top1_id']==r['io_id'] for v,r in zip(cv,rr))),('retained_positive',sum(v['diff']>0 for v in cv)),('top1_disagreements',sum(a['top1_id']!=b['top1_id'] for a,b in zip(fv,cv)))]:assert s[key]==value
    gap[k]=abs(avg(diff))
   maxk=max(KLIST,key=gap.get);assert ss['modes'][mode]['max_gap_subset']==maxk;near(ss['modes'][mode]['max_gap'],gap[maxk])
   worst=sorted(rr,key=lambda r:(-abs(r['clean']['diff']-r['conditions'][mode]['retained']['']['diff']),r['id']))[0]
   assert ss['modes'][mode]['worst_row_id']==worst['id'];near(ss['modes'][mode]['worst_row_residual'],worst['clean']['diff']-worst['conditions'][mode]['retained']['']['diff'])
 means=json.loads((p/'means.json').read_text());refs=json.loads((p/'references.json').read_text());assert len(refs)==24 and set(means)=={'0','1','2','3'}
 for ti,mat in means.items():
  assert len(mat)==12 and all(len(row)==768 and all(math.isfinite(v) for v in row) for row in mat)
  subset=[r for r in refs if str(r['template_index'])==ti];assert len(subset)==6
  assert len({tuple(r['ids'][i] for i in r['name_positions']) for r in subset})==6
  for r in subset:assert len(set(r['ids'][i] for i in r['name_positions']))==3
 runtime=json.loads((p/'runtime.json').read_text());assert runtime['rows']==len(rows) and runtime['unique_scored_forwards']==31*len(rows) and runtime['reference_forwards']==24
 return {'status':'passed','rows':len(rows),'saved_score_cells':33*len(rows),'unique_scored_forwards':31*len(rows),'inputs':{f:digest(p/f) for f in ['records.json','summary.json','means.json','references.json','identity.json','runtime.json']},'checks':['input identity','every logit subtraction','eight subset keys','all group means/MAE/counts','exhaustive gap maximum','counterexample selection','reference structure','controls/budget counts'],'model_argmax_note':'arithmetic cannot prove unsaved-vocabulary argmax; replay.py independently executes model'}
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--results',default='results');ap.add_argument('--out');a=ap.parse_args();v=audit(a.results)
 if a.out:Path(a.out).write_text(json.dumps(v,indent=2)+'\n')
 print(json.dumps(v))
