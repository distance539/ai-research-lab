"""Controlled one-file changes in temporary copies; no source mutation."""
import json,shutil,tempfile,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent
cases=[('label','calibration.json'),('logit','exploration.json'),('order','exploration.json'),('group','protocol.json'),('pool','pools.json'),('code','metrics.py'),('missing','run.py')]
results=[]
for kind,name in cases:
 with tempfile.TemporaryDirectory() as td:
  d=Path(td)/'code';shutil.copytree(R,d,ignore=shutil.ignore_patterns('__pycache__'))
  path=d/name
  if kind=='missing':path.unlink()
  elif kind=='code':path.write_text(path.read_text()+'\n# injected change\n')
  else:
   v=json.loads(path.read_text())
   if kind=='label':v[0]['label']=1-v[0]['label']
   elif kind=='logit':v[0]['logits'][0]+=.01
   elif kind=='order':v[0],v[1]=v[1],v[0]
   elif kind=='group':v['minimum_group_n']=49
   elif kind=='pool':v['confirmation'][0]['id']='0'*64
   path.write_text(json.dumps(v))
  r=subprocess.run([sys.executable,'freeze.py'],cwd=d,capture_output=True,text=True)
  expected='Frozen file mismatch: '+name
  assert r.returncode!=0 and expected in r.stderr
  results.append(dict(case=kind,file=name,exit_code=r.returncode,diagnostic=expected))
# Independent arithmetic checker must also catch damaged derived results without a hash gate.
with tempfile.TemporaryDirectory() as td:
 d=Path(td)/'code';shutil.copytree(R,d,ignore=shutil.ignore_patterns('__pycache__'))
 path=d/'predictions.json';v=json.loads(path.read_text());v['1000_global']['rows'][0]['loss']+=.01;path.write_text(json.dumps(v))
 r=subprocess.run([sys.executable,'audit.py'],cwd=d,capture_output=True,text=True)
 assert r.returncode!=0 and 'AssertionError' in r.stderr
 results.append(dict(case='derived_loss',file='predictions.json',exit_code=r.returncode,diagnostic='AssertionError'))
(R/'damage_result.json').write_text(json.dumps(dict(status='passed',cases=results),indent=2)+'\n')
print(json.dumps(results,indent=2))
