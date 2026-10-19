"""Independent standard-library arithmetic/identity audit. Does not import run/analyze."""
import json,math,hashlib,argparse,itertools
from pathlib import Path

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b,tol=1e-6):assert math.isfinite(a) and math.isfinite(b) and abs(a-b)<=tol,(a,b,tol)
def avg(v):return math.fsum(v)/len(v)
def audit(root,out):
 cfg=json.loads((root/'config.json').read_text());plan=json.loads((root/'plan.json').read_text())
 for f,h in plan['files'].items():assert digest(root/f)==h,f
 rows=json.loads((out/'records.json').read_text());s=json.loads((out/'summary.json').read_text());bank=json.loads((out/'reference_bank.json').read_text());identity=json.loads((out/'identity.json').read_text())
 assert identity['plan_sha256']==digest(root/'plan.json') and identity['resources_sha256']==digest(root/'resources.lock.json')
 samples=json.loads((root/'samples.json').read_text());expected=[r for r in samples if r['id'] in cfg['smoke_ids']] if identity['mode']=='smoke' else samples
 assert [r['id'] for r in rows]==[r['id'] for r in expected]==identity['sample_ids'];assert len(bank)==12
 parent=json.loads((root/'parent_summary.json').read_text())['conditions']['s2_flip']['ranking'];assert list(cfg['heads'].values())==[[r['layer'],r['head']] for r in parent[:3]]
 for ti in (0,1):
  b=[r for r in bank if r['template_index']==ti];assert [r['donor_index'] for r in b]==list(range(6));assert len({tuple(r['ids']) for r in b})==6
  names=[b[0]['ids'][p] for p in b[0]['name_positions']];assert len(set(names))==3
  assert [tuple(r['ids'][p] for p in r['name_positions']) for r in b]==list(itertools.permutations(names))
 for r,ex in zip(rows,expected):
  for k,v in ex.items():assert r[k]==v
  assert r['end_position']==len(r['ids'])-1 and r['ids'].count(r['io_id'])==1 and r['ids'].count(r['s_id'])==2
  assert r['z_shape']==[1,len(r['ids']),12,64] and r['donor_index']==cfg['donor_indices'][str(r['id'])]
  assert r['controls']['self_error']==r['controls']['empty_error']==0
  assert r['controls']['official_readout_error']<.001 and max(r['controls']['projected_errors'].values())<.001
  b=[q for q in bank if q['template_index']==r['template_index']]
  assert set(r['interventions'])==set(cfg['modes'])
  for mode in cfg['modes']:
   assert set(r['interventions'][mode])==set(cfg['subsets'])
   for k in 'ABC':
    z=r['clean_z'][k];v=r['replacement_z'][mode][k];assert len(z)==len(v)==64
    target=[0.]*64 if mode=='zero' else b[r['donor_index']]['z'][k] if mode=='resample_abc' else [avg([q['z'][k][j] for q in b]) for j in range(64)]
    for a,c in zip(v,target):close(a,c,2e-6)
    for key,val in [('clean_norm',math.sqrt(math.fsum(q*q for q in z))),('replacement_norm',math.sqrt(math.fsum(q*q for q in v))),('distance_to_clean',math.sqrt(math.fsum((a-c)**2 for a,c in zip(z,v))))]:close(r['diagnostics'][mode][k][key],val,2e-6)
  for v in [r['clean']]+[v for m in r['interventions'].values() for v in m.values()]:
   close(v['io_logit']-v['s_logit'],v['diff']);assert 0<=v['p_io']<=1 and 0<=v['top1_id']<50257
 close(s['mean_clean_diff'],avg([r['clean']['diff'] for r in rows]));assert s['n']==len(rows)
 for mode in cfg['modes']:
  for subset in cfg['subsets']:
   v=[r['interventions'][mode][subset] for r in rows];q=s['conditions'][mode][subset]
   close(q['mean_diff'],avg([x['diff'] for x in v]));close(q['mean_drop'],avg([r['clean']['diff']-x['diff'] for r,x in zip(rows,v)]));close(q['mean_p_io'],avg([x['p_io'] for x in v]))
   assert q['positive_diff']==sum(x['diff']>0 for x in v);assert q['vocab_correct']==sum(x['top1_id']==r['io_id'] for r,x in zip(rows,v))
  vals=[]
  for r,e in zip(rows,s['interactions'][mode]['rows']):
   assert r['id']==e['id'];D=r['clean']['diff'];v={k:q['diff'] for k,q in r['interventions'][mode].items()}
   z={f'I_{a}{b}':v[a]+v[b]-v[a+b]-D for a,b in [('A','B'),('A','C'),('B','C')]}
   z.update(conditional_C_after_AB=v['AB']-v['ABC'],conditional_C_minus_single=v['AB']+v['C']-v['ABC']-D,triple_minus_singles=v['A']+v['B']+v['C']-v['ABC']-2*D)
   for k,x in z.items():close(e[k],x)
   vals.append(z)
  for k,v in s['interactions'][mode]['means'].items():close(v,avg([x[k] for x in vals]))
 return {'status':'passed','rows':len(rows),'interventions':len(rows)*21,'bank_rows':12,'input_sha256':{f:digest(out/f) for f in ['records.json','reference_bank.json','summary.json','identity.json']},'script_sha256':digest(Path(__file__)),'checks':['frozen inputs','selected heads','donor identity/permutations','all scalar candidate differences','all336 condition cells when formal','all vector means/norms/distances','paired interactions','aggregate means/counts','implementation controls'],'not_verified':'full-vocabulary argmax requires model replay; no human semantic audit claimed'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('out');a=p.parse_args();root=Path(__file__).resolve().parent;out=Path(a.out);r=audit(root,out);(out/'audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
