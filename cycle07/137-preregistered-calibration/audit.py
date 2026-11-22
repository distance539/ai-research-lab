"""Independent identity/count/softmax audit, no imports from analysis."""
import json,hashlib,math
from pathlib import Path
R=Path(__file__).resolve().parent
read=lambda n:json.loads((R/n).read_text())
p=read('pools.json');res=read('results.json');cfg=read('protocol.json');count=0
for name in ['exploration','calibration']:
 rows=read(name+'.json');assert len(rows)==1000
 assert set(r['id'] for r in rows)==set(r['id'] for r in p[name])
 assert not set(r['id'] for r in rows)&set(r['id'] for r in p['confirmation'])
 small=sum(int(r['retained_tokens'])<257 for r in rows)
 assert [small,1000-small]==res[name]['group_counts']
 orders=[]
 for t in [1,cfg['fixed_T'],cfg['target_T']]:
  vals=[]
  for r in rows:
   z=[x/t for x in r['logits']];m=max(z);v=[math.exp(x-m) for x in z];conf=max(v)/sum(v);vals.append((-conf,r['id']));count+=1
  orders.append([i for _,i in sorted(vals)])
 assert orders[0]==orders[1]==orders[2]
for n,d in res['inputs'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==d
out={'status':'passed','softmax_rows':count,'candidate_fits':0,'confirmation_predictions_read':False,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(R/'audit_result.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
